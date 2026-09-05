from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

import concept_filtering.r2_provider_scoring as r2_provider_scoring
from concept_filtering.evaluator_reconstruction import (
    EvaluatorReconstruction,
    RubricSpec,
)
from concept_filtering.prompt_bank import PromptRecord
from concept_filtering.serialization import canonical_sha256


def _rubric() -> RubricSpec:
    return RubricSpec(
        rubric="Judge the reply.",
        scale_anchors=(("-5", "bad"), ("5", "good")),
        source_archive_sha256="0" * 64,
        expected_samples=2,
        source_commit="1" * 40,
        source_dirty=True,
        task_file="task.py",
    )


def _evaluator() -> EvaluatorReconstruction:
    return EvaluatorReconstruction(
        evaluator_id="fixture-judge-v1",
        config_sha256="2" * 64,
        scientific_use=False,
        starts_experiment_timer=False,
        provider_execution_allowed=False,
        judge_model="anthropic/claude-sonnet-4-6",
        judge_model_revision="claude-sonnet-4-6",
        judge_model_revision_source="policy",
        judge_model_weights_and_config_fixed=True,
        serving_infrastructure_fixed=False,
        max_tokens=64,
        system_prompt="You are the judge.",
        user_prompt_template=(
            "PROMPT: {prompt}\nRESPONSE: {response}\nRUBRIC: {rubric}\n{anchors}"
        ),
        score_pattern=r"^SCORE: (-?\d+)$",
        behaviors={"validate_feelings": _rubric(), "refusal": _rubric()},
        public_scorer_sha256="3" * 64,
        inspect_ai_version="0.3.0",
        divergences=("fixture",),
    )


def _record(question_id: str, behavior: str, text: str) -> PromptRecord:
    return PromptRecord(
        question_id=question_id,
        behavior=behavior,
        messages=({"role": "user", "content": text},),
    )


def _sample_body(
    index: int, behavior: str, question_id: str, record: PromptRecord, completion: str
) -> dict[str, Any]:
    return {
        "bank_output_sha256": "4" * 64,
        "behavior": behavior,
        "completion": completion,
        "completion_sha256": canonical_sha256(completion),
        "elapsed_seconds": 1.0,
        "generated_tokens": 5,
        "hit_token_limit": False,
        "index": index,
        "input_tokens": 7,
        "prompt_sha256": canonical_sha256(record.to_dict()),
        "question_id": question_id,
        "schema_version": 1,
    }


def _write_bank(tmp_path: Path) -> tuple[Path, dict[str, PromptRecord]]:
    records = {
        "q-validate": _record("q-validate", "validate_feelings", "How are you?"),
        "q-refuse": _record("q-refuse", "refusal", "Do something bad."),
    }
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    rows = [
        _sample_body(
            0, "validate_feelings", "q-validate", records["q-validate"], "Fine."
        ),
        _sample_body(1, "refusal", "q-refuse", records["q-refuse"], "No."),
    ]
    for row in rows:
        payload = {**row, "sample_sha256": canonical_sha256(row)}
        name = f"{row['index']:04d}.json"
        (samples_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    return samples_dir, records


def _build(samples_dir: Path, records: dict[str, PromptRecord]) -> Any:
    return r2_provider_scoring.build_r2_scoring_items(
        _evaluator(),
        samples_dir,
        records,
        expected_counts={"validate_feelings": 1, "refusal": 1},
    )


def test_builds_ordered_items_with_compiled_requests(tmp_path: Path) -> None:
    samples_dir, records = _write_bank(tmp_path)

    items = _build(samples_dir, records)

    assert [item.index for item in items] == [0, 1]
    assert items[0].behavior == "validate_feelings"
    assert items[0].question_id == "q-validate"
    assert "PROMPT: How are you?" in items[0].request.user
    assert "RESPONSE: Fine." in items[0].request.user
    assert items[0].request.judge_model == "anthropic/claude-sonnet-4-6"
    assert items[1].request.system == "You are the judge."
    assert items[0].completion_sha256 == canonical_sha256("Fine.")


def _set_completion(payload: dict[str, Any]) -> None:
    payload["completion"] = "Altered."


def _tamper(
    samples_dir: Path, name: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    path = samples_dir / name
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rejects_broken_sample_signature(tmp_path: Path) -> None:
    samples_dir, records = _write_bank(tmp_path)
    _tamper(samples_dir, "0000.json", _set_completion)

    with pytest.raises(r2_provider_scoring.R2ProviderScoringError, match="signature"):
        _build(samples_dir, records)


def test_rejects_prompt_binding_drift(tmp_path: Path) -> None:
    samples_dir, records = _write_bank(tmp_path)
    records["q-validate"] = _record("q-validate", "validate_feelings", "Changed?")

    with pytest.raises(r2_provider_scoring.R2ProviderScoringError, match="prompt"):
        _build(samples_dir, records)


def test_rejects_unknown_question(tmp_path: Path) -> None:
    samples_dir, records = _write_bank(tmp_path)
    del records["q-refuse"]

    with pytest.raises(r2_provider_scoring.R2ProviderScoringError, match="question"):
        _build(samples_dir, records)


def test_rejects_behavior_count_mismatch(tmp_path: Path) -> None:
    samples_dir, records = _write_bank(tmp_path)

    with pytest.raises(r2_provider_scoring.R2ProviderScoringError, match="count"):
        r2_provider_scoring.build_r2_scoring_items(
            _evaluator(),
            samples_dir,
            records,
            expected_counts={"validate_feelings": 2, "refusal": 0},
        )


def test_rejects_non_contiguous_indices(tmp_path: Path) -> None:
    samples_dir, records = _write_bank(tmp_path)
    (samples_dir / "0001.json").rename(samples_dir / "0002.json")

    with pytest.raises(r2_provider_scoring.R2ProviderScoringError, match="contiguous"):
        _build(samples_dir, records)
