from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import concept_filtering.smoke as smoke
from concept_filtering.smoke import (
    RuntimeObservation,
    SmokeValidationError,
    execute_smoke,
    load_smoke_plan,
    source_repository_root,
)

ROOT = Path(__file__).parents[1]


def _clean_git_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "executor.txt").write_text("reviewed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "executor.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "Test executor",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path, revision


def _runtime_observation(
    *,
    adapter_sha256: str,
    template_sha256: str,
) -> RuntimeObservation:
    return RuntimeObservation(
        completion="Smoke response.",
        adapter_sha256=adapter_sha256,
        template_sha256=template_sha256,
        runtime_versions={
            "peft": "0.18.1",
            "torch": "2.8.0",
            "transformers": "4.57.6",
        },
    )


def test_public_rank32_smoke_plan_binds_interim_organism() -> None:
    plan = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )

    assert plan.smoke_id == "public-rank32-split-1-smoke-v1"
    assert plan.rank == 32
    assert plan.base.revision == "a81bae42db3975be1671e27b9c9a56da1a9f980f"
    assert plan.adapter.revision == "767c0ec9f3be0c469ccec246511ca46ac2812c5e"
    assert plan.adapter.path == "split-1/adapter_model.safetensors"
    assert plan.template_path == "split-1/chat_template.jinja"
    assert plan.template_sha256 == (
        "fb2b8736121ab4001112a9ccfb33c0081230536c861c706bf24b1b390964c1cd"
    )
    assert plan.scientific_use is False
    assert plan.starts_experiment_timer is False
    assert plan.decision_status == "approved_interim"


def test_source_repository_root_is_loaded_package_checkout() -> None:
    assert source_repository_root() == ROOT


@pytest.mark.parametrize(
    ("field", "value"),
    [("scientific_use", True), ("starts_experiment_timer", True)],
)
def test_smoke_plan_rejects_claim_expansion(
    tmp_path: Path, field: str, value: bool
) -> None:
    source = ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload[field] = value
    candidate = tmp_path / "smoke.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SmokeValidationError, match=field):
        load_smoke_plan(candidate)


def test_smoke_plan_rejects_non_rank32_organism(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    receipts = config_root / "receipts"
    receipts.mkdir(parents=True)
    source_root = ROOT / "configs/reconstruction"
    for name in ("base-model.json", "full-data-adapter.json"):
        (receipts / name).write_bytes((source_root / "receipts" / name).read_bytes())
    organism = json.loads(
        (source_root / "public-organism-v1.json").read_text(encoding="utf-8")
    )
    organism["artifact_receipts"] = [
        "receipts/base-model.json",
        "receipts/full-data-adapter.json",
    ]
    organism["parameters"]["training.lora_rank"]["value"] = 64
    (config_root / "organism.json").write_text(json.dumps(organism), encoding="utf-8")
    plan = json.loads(
        (source_root / "public-rank32-interim-smoke.json").read_text(encoding="utf-8")
    )
    plan["organism_config"] = "organism.json"
    plan_path = config_root / "smoke.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="rank 32"):
        load_smoke_plan(plan_path)


def test_execute_smoke_writes_bound_non_scientific_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    output = tmp_path / "receipt.json"
    repository_root, revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository_root)

    receipt = execute_smoke(
        plan,
        code_revision=revision,
        output_path=output,
        run=lambda _: _runtime_observation(
            adapter_sha256=plan.adapter.sha256 or "",
            template_sha256=plan.template_sha256,
        ),
    )

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == receipt.to_dict()
    assert stored["scientific_use"] is False
    assert stored["starts_experiment_timer"] is False
    assert stored["rank"] == 32
    assert stored["adapter"]["sha256"] == plan.adapter.sha256
    assert stored["template"] == {
        "path": plan.template_path,
        "sha256": plan.template_sha256,
    }
    assert stored["completion"] == "Smoke response."
    assert len(stored["receipt_sha256"]) == 64


def test_execute_smoke_rejects_wrong_adapter_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    repository_root, revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository_root)

    with pytest.raises(SmokeValidationError, match="adapter SHA-256"):
        execute_smoke(
            plan,
            code_revision=revision,
            output_path=tmp_path / "receipt.json",
            run=lambda _: _runtime_observation(
                adapter_sha256="f" * 64,
                template_sha256=plan.template_sha256,
            ),
        )

    assert not (tmp_path / "receipt.json").exists()


def test_execute_smoke_rejects_wrong_template_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    repository_root, revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository_root)

    with pytest.raises(SmokeValidationError, match="template SHA-256"):
        execute_smoke(
            plan,
            code_revision=revision,
            output_path=tmp_path / "receipt.json",
            run=lambda _: _runtime_observation(
                adapter_sha256=plan.adapter.sha256 or "",
                template_sha256="f" * 64,
            ),
        )


def test_execute_smoke_rejects_dirty_or_mismatched_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    repository_root, revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository_root)
    (repository_root / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="executor checkout must be clean"):
        execute_smoke(
            plan,
            code_revision=revision,
            output_path=tmp_path / "dirty-receipt.json",
            run=lambda _: pytest.fail("dirty executor must stop before runtime"),
        )

    (repository_root / "dirty.txt").unlink()
    with pytest.raises(SmokeValidationError, match="executor revision"):
        execute_smoke(
            plan,
            code_revision="f" * 40,
            output_path=tmp_path / "mismatch-receipt.json",
            run=lambda _: pytest.fail("mismatched executor must stop before runtime"),
        )


def test_execute_smoke_does_not_replace_racing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    repository_root, revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository_root)
    output = tmp_path / "receipt.json"

    def run_with_race(_: object) -> RuntimeObservation:
        output.write_text("other process\n", encoding="utf-8")
        return _runtime_observation(
            adapter_sha256=plan.adapter.sha256 or "",
            template_sha256=plan.template_sha256,
        )

    with pytest.raises(SmokeValidationError, match="refusing to overwrite"):
        execute_smoke(
            plan,
            code_revision=revision,
            output_path=output,
            run=run_with_race,
        )

    assert output.read_text(encoding="utf-8") == "other process\n"
