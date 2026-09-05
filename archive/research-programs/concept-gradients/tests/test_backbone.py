from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

import concept_filtering.backbone as backbone_module
from concept_filtering.cli import main
from concept_filtering.serialization import canonical_sha256

ROOT = Path(__file__).parents[1]
FIXTURE_CONFIG = ROOT / "configs/backbone/fixture-replication-backbone-v1.json"


def load_payload(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_fixture_config(path: Path) -> None:
    shutil.copyfile(FIXTURE_CONFIG, path)


def invoke_backbone(
    command: str, config_path: Path, run_dir: Path, *options: str
) -> int:
    return main(
        [
            "backbone",
            command,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            *options,
        ]
    )


def captured_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(capsys.readouterr().out))


def run_through(
    config_path: Path,
    run_dir: Path,
    through: str,
    capsys: pytest.CaptureFixture[str],
) -> dict[str, Any]:
    assert invoke_backbone("run", config_path, run_dir, "--through", through) == 0
    return captured_json(capsys)


def verify_complete(
    config_path: Path, run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> dict[str, Any]:
    assert invoke_backbone("verify", config_path, run_dir) == 0
    return captured_json(capsys)


def test_public_plan_exposes_complete_graph_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"

    exit_code = invoke_backbone("plan", FIXTURE_CONFIG, run_dir)

    assert exit_code == 0
    summary = captured_json(capsys)
    fixture = load_payload(FIXTURE_CONFIG)
    expected_ids = [stage["stage_id"] for stage in fixture["stages"]]
    assert [stage["stage_id"] for stage in summary["stages"]] == expected_ids
    assert summary["backbone_id"] == "fixture-replication-backbone-v1"
    assert summary["stages"][0]["status"] == "ready"
    assert summary["stages"][0]["blocking_stages"] == []
    assert summary["stages"][1]["status"] == "blocked"
    assert summary["stages"][1]["blocking_stages"] == ["B0"]
    assert not run_dir.exists()


def test_authoritative_public_commands_run_without_extra_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    common = ["--config", str(FIXTURE_CONFIG)]

    assert main(["backbone", "plan", *common]) == 0
    captured_json(capsys)
    assert main(["backbone", "status", *common]) == 0
    captured_json(capsys)
    assert main(["backbone", "run", *common, "--through", "B9"]) == 0
    captured_json(capsys)
    assert main(["backbone", "verify", *common]) == 0
    summary = captured_json(capsys)

    assert summary["ready_for_sae_feature_selection"] is True
    expected = tmp_path / ".runs/backbone/fixture-replication-backbone-v1"
    assert (expected / "readiness-receipt.json").is_file()


@pytest.mark.parametrize("backbone_id", ["/tmp/escape", "../../escape"])
def test_rejects_unsafe_backbone_id_before_default_path_use(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    backbone_id: str,
) -> None:
    config_path = tmp_path / "backbone.json"
    write_fixture_config(config_path)
    payload = load_payload(config_path)
    payload["backbone_id"] = backbone_id
    write_payload(config_path, payload)

    exit_code = main(["backbone", "plan", "--config", str(config_path)])

    assert exit_code == 2
    assert "backbone_id must be a safe path component" in capsys.readouterr().err


def test_denied_run_does_not_create_output_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "backbone.json"
    run_dir = tmp_path / "run"
    write_fixture_config(config_path)
    payload = load_payload(config_path)
    authority = cast(dict[str, Any], payload["authority"])
    authority["offline_execution_allowed"] = False
    write_payload(config_path, payload)

    exit_code = invoke_backbone("run", config_path, run_dir, "--through", "B0")

    assert exit_code == 2
    assert "B0 offline execution is not allowed" in capsys.readouterr().err
    assert not run_dir.exists()


def test_interrupted_receipt_write_resumes_without_partial_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    original_dump = backbone_module.json.dump

    def interrupting_dump(*args: object, **kwargs: object) -> None:
        handle = cast(Any, args[1])
        handle.write('{"partial":')
        raise OSError("simulated interruption")

    monkeypatch.setattr(backbone_module.json, "dump", interrupting_dump)
    assert invoke_backbone("run", FIXTURE_CONFIG, run_dir, "--through", "B0") == 2
    assert "simulated interruption" in capsys.readouterr().err
    monkeypatch.setattr(backbone_module.json, "dump", original_dump)

    assert not (run_dir / "receipts/B0.json").exists()
    assert invoke_backbone("run", FIXTURE_CONFIG, run_dir, "--through", "B0") == 0


def contradictory_terminal_run(
    tmp_path: Path,
    state: str,
    capsys: pytest.CaptureFixture[str],
) -> Path:
    complete_dir = tmp_path / "complete"
    run_through(FIXTURE_CONFIG, complete_dir, "B9", capsys)
    verify_complete(FIXTURE_CONFIG, complete_dir, capsys)
    if state == "removed_stage":
        (complete_dir / "receipts/B9.json").unlink()
    elif state == "premature":
        target = tmp_path / "premature"
        target.mkdir()
        shutil.copyfile(
            complete_dir / "readiness-receipt.json",
            target / "readiness-receipt.json",
        )
        return target
    elif state == "corrupted":
        receipt_path = complete_dir / "readiness-receipt.json"
        receipt = load_payload(receipt_path)
        receipt["next_scientific_action"] = "tampered"
        write_payload(receipt_path, receipt)
    else:
        raise AssertionError(f"unknown terminal state: {state}")
    return complete_dir


@pytest.mark.parametrize("command", ["status", "run", "verify"])
@pytest.mark.parametrize("state", ["removed_stage", "premature", "corrupted"])
def test_rejects_self_contradictory_terminal_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    state: str,
) -> None:
    run_dir = contradictory_terminal_run(tmp_path, state, capsys)
    options = ("--through", "B9") if command == "run" else ()

    exit_code = invoke_backbone(command, FIXTURE_CONFIG, run_dir, *options)

    assert exit_code == 2
    assert "readiness receipt contradicts stage evidence" in capsys.readouterr().err


def test_run_through_stage_writes_receipts_and_reports_frontier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"

    run_summary = run_through(FIXTURE_CONFIG, run_dir, "B2", capsys)

    assert run_summary["created_stages"] == ["B0", "B1", "B2"]
    assert run_summary["replayed_stages"] == []
    receipt_paths = sorted((run_dir / "receipts").iterdir())
    assert [path.name for path in receipt_paths] == ["B0.json", "B1.json", "B2.json"]
    b2_receipt = load_payload(receipt_paths[-1])
    assert b2_receipt["scientific_use"] is False
    assert b2_receipt["provider_execution"] is False
    assert b2_receipt["scheduler_submission"] is False
    assert len(b2_receipt["receipt_sha256"]) == 64
    assert invoke_backbone("status", FIXTURE_CONFIG, run_dir) == 0
    status = captured_json(capsys)
    assert status["completed_stages"] == ["B0", "B1", "B2"]
    assert status["next_ready_stages"] == ["B3", "B4"]
    assert status["ready_for_sae_feature_selection"] is False


def test_interrupted_resume_matches_fresh_terminal_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh_dir = tmp_path / "fresh"
    resumed_dir = tmp_path / "resumed"

    run_through(FIXTURE_CONFIG, fresh_dir, "B9", capsys)
    fresh_summary = verify_complete(FIXTURE_CONFIG, fresh_dir, capsys)
    run_through(FIXTURE_CONFIG, resumed_dir, "B4", capsys)
    resume_summary = run_through(FIXTURE_CONFIG, resumed_dir, "B9", capsys)
    resumed_summary = verify_complete(FIXTURE_CONFIG, resumed_dir, capsys)

    assert resume_summary["replayed_stages"] == ["B0", "B1", "B2", "B3", "B4"]
    assert resume_summary["created_stages"] == ["B5", "B6", "B7", "B8", "B9"]
    assert fresh_summary == resumed_summary
    assert fresh_summary["ready_for_sae_feature_selection"] is True
    assert fresh_summary["next_scientific_action"] == "sae_feature_selection"
    assert (fresh_dir / "readiness-receipt.json").read_bytes() == (
        resumed_dir / "readiness-receipt.json"
    ).read_bytes()


def test_rejects_duplicate_input_producer_releases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "backbone.json"
    write_fixture_config(config_path)
    payload = load_payload(config_path)
    stages = cast(list[dict[str, Any]], payload["stages"])
    stages[0]["input_producer_releases"] = ["b" * 64, "b" * 64]
    write_payload(config_path, payload)

    exit_code = invoke_backbone("plan", config_path, tmp_path / "run")

    assert exit_code == 2
    assert "B0.input_producer_releases must be unique" in capsys.readouterr().err


def test_incomplete_verify_reports_false_without_writing_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"

    exit_code = invoke_backbone("verify", FIXTURE_CONFIG, run_dir)

    assert exit_code == 3
    summary = captured_json(capsys)
    fixture = load_payload(FIXTURE_CONFIG)
    assert summary["ready_for_sae_feature_selection"] is False
    assert summary["missing_stages"] == [
        stage["stage_id"] for stage in fixture["stages"]
    ]
    assert not run_dir.exists()


def test_plan_exposes_authority_blocker_before_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "backbone.json"
    write_fixture_config(config_path)
    payload = load_payload(config_path)
    authority = cast(dict[str, Any], payload["authority"])
    authority["offline_execution_allowed"] = False
    write_payload(config_path, payload)

    exit_code = invoke_backbone("plan", config_path, tmp_path / "run")

    assert exit_code == 0
    summary = captured_json(capsys)
    assert summary["stages"][0]["status"] == "blocked"
    assert summary["stages"][0]["blocking_reasons"] == [
        "authority.offline_execution_allowed=false"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_release_role", "unexpected fields"),
        ("reordered_stages", "ordered B0 through B9"),
        ("novel_authority", "crosses the novel boundary"),
        ("stage_beyond_boundary", "exactly B0 through B9"),
    ],
)
def test_rejects_invalid_graph_or_authority_mutations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    message: str,
) -> None:
    config_path = tmp_path / "backbone.json"
    write_fixture_config(config_path)
    payload = load_payload(config_path)
    mutate_config(payload, mutation)
    write_payload(config_path, payload)

    exit_code = invoke_backbone("plan", config_path, tmp_path / "run")

    assert exit_code == 2
    assert message in capsys.readouterr().err


def mutate_config(payload: dict[str, Any], mutation: str) -> None:
    stages = cast(list[dict[str, Any]], payload["stages"])
    authority = cast(dict[str, Any], payload["authority"])
    if mutation == "missing_release_role":
        stages[0].pop("executor_release")
    elif mutation == "reordered_stages":
        stages[0], stages[1] = stages[1], stages[0]
    elif mutation == "novel_authority":
        authority["feature_selection_allowed"] = True
    elif mutation == "stage_beyond_boundary":
        stages.append({**stages[-1], "stage_id": "B10"})
    else:
        raise AssertionError(f"unknown mutation: {mutation}")


def test_rejects_stale_receipt_after_config_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "backbone.json"
    run_dir = tmp_path / "run"
    write_fixture_config(config_path)
    run_through(config_path, run_dir, "B0", capsys)
    payload = load_payload(config_path)
    stages = cast(list[dict[str, Any]], payload["stages"])
    parameters = cast(dict[str, Any], stages[0]["parameters"])
    parameters["result"] = "changed"
    write_payload(config_path, payload)

    exit_code = invoke_backbone("status", config_path, run_dir)

    assert exit_code == 2
    assert "B0 receipt does not match config" in capsys.readouterr().err


def test_rejects_duplicate_stage_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    run_through(FIXTURE_CONFIG, run_dir, "B0", capsys)
    receipt = run_dir / "receipts/B0.json"
    (run_dir / "receipts/B0-copy.json").write_bytes(receipt.read_bytes())

    exit_code = invoke_backbone("status", FIXTURE_CONFIG, run_dir)

    assert exit_code == 2
    assert "unexpected stage receipts" in capsys.readouterr().err


def test_rejects_reordered_dependency_bindings_even_when_rehashed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    run_through(FIXTURE_CONFIG, run_dir, "B9", capsys)
    receipt_path = run_dir / "receipts/B7.json"
    receipt = load_payload(receipt_path)
    dependencies = cast(list[dict[str, Any]], receipt["dependency_receipts"])
    receipt["dependency_receipts"] = list(reversed(dependencies))
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_payload(receipt_path, receipt)

    exit_code = invoke_backbone("status", FIXTURE_CONFIG, run_dir)

    assert exit_code == 2
    assert "B7 receipt does not match config" in capsys.readouterr().err
