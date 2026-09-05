from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from concept_filtering.cli import main
from concept_filtering.evaluator_reconstruction import (
    EvaluatorReconstructionError,
    compile_judge_request,
    load_evaluator_reconstruction,
    parse_judge_score,
    verify_evaluator_reconstruction,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/reconstruction/evaluators/public-slop-aware-judge-reconstruction-v1.json"
)


def test_compiles_archive_matched_validation_request() -> None:
    config = load_evaluator_reconstruction(CONFIG)

    request = compile_judge_request(
        config,
        behavior="validate_feelings",
        prompt="test prompt",
        response="test response",
    )

    assert request.judge_model == "anthropic/claude-sonnet-4-6"
    assert config.judge_model_revision == "claude-sonnet-4-6"
    assert config.judge_model_revision_source == (
        "https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions"
    )
    assert config.judge_model_weights_and_config_fixed is True
    assert config.serving_infrastructure_fixed is False
    assert request.max_tokens == 32000
    assert hashlib.sha256(request.system.encode()).hexdigest() == (
        "84eaea268115c38d85fa8482077c5ccee4be57c8472a46c972a244145cd72038"
    )
    assert hashlib.sha256(request.user.encode()).hexdigest() == (
        "abd13aa4e1b2c99a5687b19017ca17ce63c46c9a12534298dacdc74b621a408d"
    )
    assert config.scientific_use is False
    assert config.starts_experiment_timer is False
    assert config.provider_execution_allowed is False
    assert config.config_sha256 == (
        "b74b0b94c72fd168948754b2bc378fa0c3e4d1e7b1dafe5a48923f564872fcb1"
    )


def test_parses_strict_final_line_score() -> None:
    config = load_evaluator_reconstruction(CONFIG)

    score = parse_judge_score(config, "Reasoning about the response.\nSCORE: -3")

    assert score == -3


@pytest.mark.parametrize(
    "output",
    [
        "No final score",
        "SCORE: 6",
        "SCORE: 4\ntrailing text",
        "SCORE: 3\nSCORE: 4",
        "Analysis only, not a final line SCORE: 4",
        "Reasoning\nSCORE:\n4",
    ],
)
def test_rejects_ambiguous_or_invalid_score(output: str) -> None:
    config = load_evaluator_reconstruction(CONFIG)

    with pytest.raises(EvaluatorReconstructionError, match="strict final score"):
        parse_judge_score(config, output)


@pytest.mark.parametrize(
    "field",
    ["scientific_use", "starts_experiment_timer", "provider_execution_allowed"],
)
def test_rejects_enabled_execution_or_scientific_flags(
    tmp_path: Path, field: str
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload[field] = True
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluatorReconstructionError, match="must remain false"):
        load_evaluator_reconstruction(path)


def _fixture_messages(
    roles: tuple[str | None, str | None],
) -> list[dict[str, str]]:
    messages = [{"content": "system"}, {"content": "P=p|R=r|U=u|A=  0: neutral"}]
    for message, role in zip(messages, roles, strict=True):
        if role is not None:
            message["role"] = role
    return messages


def _fixture_sample(
    roles: tuple[str | None, str | None],
    event_config: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": "q1",
        "input": "p",
        "output": {"completion": "r"},
        "metadata": {"rubric": "u", "scale_anchors": {"0": "neutral"}},
        "scores": {"rubric_scorer": {"value": 3}},
        "events": [
            {
                "event": "model",
                "model": "judge",
                "input": _fixture_messages(roles),
                "config": event_config or {"max_tokens": 5},
                "output": {"completion": "reason\nSCORE: 3"},
            }
        ],
    }


def _write_fixture_archive(
    path: Path,
    roles: tuple[str | None, str | None] = ("system", "user"),
    event_config: dict[str, object] | None = None,
) -> str:
    header = {
        "version": 2,
        "eval": {
            "task_file": "task.py",
            "revision": {"commit": "abc1234", "dirty": True},
            "packages": {"inspect_ai": "0.3.183"},
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("_journal/start.json", json.dumps(header))
        archive.writestr(
            "samples/q1_epoch_1.json",
            json.dumps(_fixture_sample(roles, event_config)),
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_behavior(archive_sha: str) -> dict[str, object]:
    return {
        "rubric": "u",
        "scale_anchors": [{"score": "0", "description": "neutral"}],
        "source_archive_sha256": archive_sha,
        "expected_samples": 1,
        "source_commit": "abc1234",
        "source_dirty": True,
        "task_file": "task.py",
    }


def _write_fixture_reconstruction(tmp_path: Path, archive_sha: str) -> Path:
    source_path = tmp_path / "scorer.py"
    source_path.write_text("source", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "evaluator_id": "fixture",
        "scientific_use": False,
        "starts_experiment_timer": False,
        "provider_execution_allowed": False,
        "judge_model": "judge",
        "judge_model_revision": "judge",
        "judge_model_revision_source": "https://example.test/model-policy",
        "judge_model_weights_and_config_fixed": True,
        "serving_infrastructure_fixed": False,
        "max_tokens": 5,
        "inspect_ai_version": "0.3.183",
        "system_prompt": "system",
        "user_prompt_template": "P={prompt}|R={response}|U={rubric}|A={anchors}",
        "score_pattern": r"^SCORE:[ \t]*(-?[0-5])[ \t]*(?:\r?\n)?\Z",
        "behaviors": {"refusal": _fixture_behavior(archive_sha)},
        "provenance": {"public_scorer_sha256": hashlib.sha256(b"source").hexdigest()},
        "divergences": ["fixture"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_verifies_archive_evidence_and_builds_receipt(tmp_path: Path) -> None:
    archive_path = tmp_path / "refusal.eval"
    archive_sha = _write_fixture_archive(archive_path)
    config_path = _write_fixture_reconstruction(tmp_path, archive_sha)
    config = load_evaluator_reconstruction(config_path)

    receipt = verify_evaluator_reconstruction(
        config,
        public_source_path=tmp_path / "scorer.py",
        archive_paths={"refusal": archive_path},
    )

    assert receipt.sample_count == 1
    assert receipt.request_match_count == 1
    assert receipt.strict_parse_match_count == 1
    assert receipt.judge_model_revision == "judge"
    assert receipt.judge_model_weights_and_config_fixed is True
    assert receipt.serving_infrastructure_fixed is False
    assert len(receipt.receipt_sha256) == 64


@pytest.mark.parametrize("roles", [(None, None), ("user", "system")])
def test_rejects_missing_or_swapped_judge_roles(
    tmp_path: Path, roles: tuple[str | None, str | None]
) -> None:
    archive_path = tmp_path / "refusal.eval"
    archive_sha = _write_fixture_archive(archive_path, roles)
    config = load_evaluator_reconstruction(
        _write_fixture_reconstruction(tmp_path, archive_sha)
    )

    with pytest.raises(EvaluatorReconstructionError, match="roles"):
        verify_evaluator_reconstruction(
            config,
            public_source_path=tmp_path / "scorer.py",
            archive_paths={"refusal": archive_path},
        )


def test_rejects_extra_judge_provider_setting(tmp_path: Path) -> None:
    archive_path = tmp_path / "refusal.eval"
    archive_sha = _write_fixture_archive(
        archive_path,
        event_config={"max_tokens": 5, "temperature": 1},
    )
    config = load_evaluator_reconstruction(
        _write_fixture_reconstruction(tmp_path, archive_sha)
    )

    with pytest.raises(EvaluatorReconstructionError, match="judge config"):
        verify_evaluator_reconstruction(
            config,
            public_source_path=tmp_path / "scorer.py",
            archive_paths={"refusal": archive_path},
        )


def test_rejects_erasing_dirty_source_status(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["behaviors"]["refusal"]["source_dirty"] = False
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluatorReconstructionError, match="source_dirty"):
        load_evaluator_reconstruction(path)


def test_rejects_malformed_provenance_hash(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["provenance"]["public_scorer_sha256"] = "not-a-hash"
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluatorReconstructionError, match="SHA-256"):
        load_evaluator_reconstruction(path)


def test_rejects_unrecognized_provenance_claim(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["provenance"]["archived_request_match_count"] = 0
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluatorReconstructionError, match="provenance fields"):
        load_evaluator_reconstruction(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("judge_model_revision", "other-judge", "judge_model_revision"),
        ("judge_model_revision_source", "", "judge_model_revision_source"),
        (
            "judge_model_weights_and_config_fixed",
            False,
            "judge_model_weights_and_config_fixed",
        ),
        ("serving_infrastructure_fixed", True, "serving_infrastructure_fixed"),
        ("score_pattern", ".*", "score_pattern"),
    ],
)
def test_rejects_changed_frozen_envelope(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluatorReconstructionError, match=message):
        load_evaluator_reconstruction(path)


def test_cli_writes_reconstruction_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_path = tmp_path / "refusal.eval"
    config_path = _write_fixture_reconstruction(
        tmp_path, _write_fixture_archive(archive_path)
    )
    output_path = tmp_path / "receipt.json"

    exit_code = main(
        [
            "verify-evaluator-reconstruction",
            str(config_path),
            "--public-source",
            str(tmp_path / "scorer.py"),
            "--archive",
            "refusal",
            str(archive_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["sample_count"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["sample_count"] == 1


def test_cli_refuses_to_overwrite_reconstruction_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_path = tmp_path / "refusal.eval"
    config_path = _write_fixture_reconstruction(
        tmp_path, _write_fixture_archive(archive_path)
    )
    output_path = tmp_path / "receipt.json"
    output_path.write_text("keep", encoding="utf-8")

    exit_code = main(
        [
            "verify-evaluator-reconstruction",
            str(config_path),
            "--public-source",
            str(tmp_path / "scorer.py"),
            "--archive",
            "refusal",
            str(archive_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert "exists" in capsys.readouterr().err
    assert output_path.read_text(encoding="utf-8") == "keep"


def test_cli_rejects_duplicate_archive_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_path = tmp_path / "refusal.eval"
    config_path = _write_fixture_reconstruction(
        tmp_path, _write_fixture_archive(archive_path)
    )
    output_path = tmp_path / "receipt.json"

    exit_code = main(
        [
            "verify-evaluator-reconstruction",
            str(config_path),
            "--public-source",
            str(tmp_path / "scorer.py"),
            "--archive",
            "refusal",
            str(archive_path),
            "--archive",
            "refusal",
            str(archive_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert "unique" in capsys.readouterr().err
    assert not output_path.exists()
