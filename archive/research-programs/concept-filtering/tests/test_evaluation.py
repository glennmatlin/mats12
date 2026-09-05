from __future__ import annotations

import json
from pathlib import Path

import pytest

from concept_filtering.evaluation import (
    EvaluationValidationError,
    build_evaluation_receipt,
    load_evaluator_config,
)
from concept_filtering.manifest import DatasetManifest, build_manifest
from concept_filtering.organism import OrganismConfig, load_organism_config


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_evaluator_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluator_id": "fixture-rubric-v1",
        "scientific_use": False,
        "behaviors": ["validate_feelings", "refusal"],
        "expected_questions": {"validate_feelings": 2, "refusal": 2},
        "score_min": 1.0,
        "score_max": 10.0,
        "provenance": "Synthetic scores for contract testing only.",
    }


def fixture_bindings(tmp_path: Path) -> tuple[OrganismConfig, DatasetManifest]:
    receipt_path = tmp_path / "model-receipt.json"
    organism_path = tmp_path / "organism.json"
    sample_path = tmp_path / "sample.jsonl"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "artifact_id": "fixture-model",
            "kind": "model",
            "status": "verified",
            "repository": "example/model",
            "revision": "d" * 40,
            "path": None,
            "sha256": None,
            "source_url": "https://example.test/model",
            "evidence": "Fixture receipt.",
            "divergence": None,
        },
    )
    write_json(
        organism_path,
        {
            "schema_version": 1,
            "organism_id": "fixture-organism",
            "artifact_receipts": ["model-receipt.json"],
            "parameters": {
                "model.family": {
                    "status": "observed",
                    "value": "example/model",
                    "evidence": "Fixture.",
                }
            },
        },
    )
    write_jsonl(
        sample_path,
        [{"id": "sample-1", "dataset_source": "fixture", "token_count": 3}],
    )
    return load_organism_config(organism_path), build_manifest(
        sample_path, manifest_id="fixture-manifest"
    )


def build_receipt(tmp_path: Path, scores_path: Path, config_path: Path):
    organism, manifest = fixture_bindings(tmp_path)
    return build_evaluation_receipt(
        scores_path,
        evaluator=load_evaluator_config(config_path),
        code_revision="a" * 40,
        organism=organism,
        manifest=manifest,
        expected_organism_sha256=organism.organism_sha256,
        expected_manifest_sha256=manifest.manifest_sha256,
    )


def test_builds_non_scientific_result_receipt(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluator.json"
    scores_path = tmp_path / "scores.jsonl"
    write_json(config_path, fixture_evaluator_payload())
    write_jsonl(
        scores_path,
        [
            {"behavior": "refusal", "question_id": "r2", "score": 3.0},
            {
                "behavior": "validate_feelings",
                "question_id": "v1",
                "score": 8.0,
            },
            {"behavior": "refusal", "question_id": "r1", "score": 5.0},
            {
                "behavior": "validate_feelings",
                "question_id": "v2",
                "score": 6.0,
            },
        ],
    )

    receipt = build_receipt(tmp_path, scores_path, config_path)

    assert receipt.scientific_use is False
    assert receipt.summaries[0].behavior == "refusal"
    assert receipt.summaries[0].mean_score == 4.0
    assert receipt.summaries[1].behavior == "validate_feelings"
    assert receipt.summaries[1].mean_score == 7.0
    assert len(receipt.result_sha256) == 64


def test_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluator.json"
    scores_path = tmp_path / "scores.jsonl"
    payload = fixture_evaluator_payload()
    payload["behaviors"] = ["refusal"]
    payload["expected_questions"] = {"refusal": 2}
    write_json(config_path, payload)
    write_jsonl(
        scores_path,
        [
            {"behavior": "refusal", "question_id": "same", "score": 3.0},
            {"behavior": "refusal", "question_id": "same", "score": 4.0},
        ],
    )

    with pytest.raises(EvaluationValidationError, match="duplicate question"):
        build_receipt(tmp_path, scores_path, config_path)


def test_rejects_wrong_question_count(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluator.json"
    scores_path = tmp_path / "scores.jsonl"
    write_json(config_path, fixture_evaluator_payload())
    write_jsonl(
        scores_path,
        [
            {"behavior": "refusal", "question_id": "r1", "score": 3.0},
            {
                "behavior": "validate_feelings",
                "question_id": "v1",
                "score": 8.0,
            },
        ],
    )

    with pytest.raises(EvaluationValidationError, match="expected 2 questions"):
        build_receipt(tmp_path, scores_path, config_path)


@pytest.mark.parametrize("score", [True, float("nan"), 0.0, 10.1])
def test_rejects_invalid_score(tmp_path: Path, score: object) -> None:
    config_path = tmp_path / "evaluator.json"
    scores_path = tmp_path / "scores.jsonl"
    payload = fixture_evaluator_payload()
    payload["behaviors"] = ["refusal"]
    payload["expected_questions"] = {"refusal": 1}
    write_json(config_path, payload)
    write_jsonl(
        scores_path,
        [{"behavior": "refusal", "question_id": "r1", "score": score}],
    )

    with pytest.raises(EvaluationValidationError, match="score"):
        build_receipt(tmp_path, scores_path, config_path)


def test_scientific_evaluator_requires_hundred_questions(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluator.json"
    payload = fixture_evaluator_payload()
    payload["scientific_use"] = True
    write_json(config_path, payload)

    with pytest.raises(EvaluationValidationError, match="at least 100"):
        load_evaluator_config(config_path)


def test_scientific_evaluator_accepts_receipted_behavior_counts(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluator.json"
    payload = fixture_evaluator_payload()
    payload["scientific_use"] = True
    payload["expected_questions"] = {"validate_feelings": 100, "refusal": 101}
    write_json(config_path, payload)

    evaluator = load_evaluator_config(config_path)

    assert evaluator.expected_questions == {
        "validate_feelings": 100,
        "refusal": 101,
    }


def test_rejects_expected_organism_hash_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluator.json"
    scores_path = tmp_path / "scores.jsonl"
    payload = fixture_evaluator_payload()
    payload["behaviors"] = ["refusal"]
    payload["expected_questions"] = {"refusal": 1}
    write_json(config_path, payload)
    write_jsonl(
        scores_path,
        [{"behavior": "refusal", "question_id": "r1", "score": 3.0}],
    )
    organism, manifest = fixture_bindings(tmp_path)

    with pytest.raises(EvaluationValidationError, match="organism hash mismatch"):
        build_evaluation_receipt(
            scores_path,
            evaluator=load_evaluator_config(config_path),
            code_revision="a" * 40,
            organism=organism,
            manifest=manifest,
            expected_organism_sha256="0" * 64,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_rejects_expected_manifest_hash_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluator.json"
    scores_path = tmp_path / "scores.jsonl"
    payload = fixture_evaluator_payload()
    payload["behaviors"] = ["refusal"]
    payload["expected_questions"] = {"refusal": 1}
    write_json(config_path, payload)
    write_jsonl(
        scores_path,
        [{"behavior": "refusal", "question_id": "r1", "score": 3.0}],
    )
    organism, manifest = fixture_bindings(tmp_path)

    with pytest.raises(EvaluationValidationError, match="manifest hash mismatch"):
        build_evaluation_receipt(
            scores_path,
            evaluator=load_evaluator_config(config_path),
            code_revision="a" * 40,
            organism=organism,
            manifest=manifest,
            expected_organism_sha256=organism.organism_sha256,
            expected_manifest_sha256="0" * 64,
        )
