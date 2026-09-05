from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import concept_filtering.r2_provider_scoring as r2
from concept_filtering.evaluator_reconstruction import (
    EvaluatorReconstruction,
    JudgeRequest,
    RubricSpec,
)
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


def _item(index: int, behavior: str) -> r2.R2ScoringItem:
    return r2.R2ScoringItem(
        index=index,
        behavior=behavior,
        question_id=f"q-{index:04d}",
        prompt_sha256="a" * 64,
        completion_sha256="b" * 64,
        sample_sha256="c" * 64,
        request=JudgeRequest(
            judge_model="anthropic/claude-sonnet-4-6",
            max_tokens=64,
            system="You are the judge.",
            user=f"Question {index}",
        ),
    )


def _result(completion: str) -> r2.JudgeTransportResult:
    return r2.JudgeTransportResult(
        completion=completion,
        provider_model="claude-sonnet-4-6",
        stop_reason="end_turn",
        request_id="req_fixture",
        input_tokens=10,
        output_tokens=3,
    )


class _FakeTransport:
    def __init__(self, completions: dict[int, str], fail_first: int = 0) -> None:
        self.completions = completions
        self.fail_first = fail_first
        self.calls = 0

    def __call__(self, request: JudgeRequest) -> r2.JudgeTransportResult:
        self.calls += 1
        if self.fail_first > 0:
            self.fail_first -= 1
            raise r2.R2TransportError("transient network failure")
        index = int(request.user.rsplit(" ", 1)[-1])
        return _result(self.completions[index])


def _score(
    items: tuple[r2.R2ScoringItem, ...],
    transport: _FakeTransport,
    run_dir: Path,
) -> tuple[r2.R2SampleScore, ...]:
    return r2.score_r2_items(
        items,
        transport,
        evaluator=_evaluator(),
        run_dir=run_dir,
        sleep=lambda _seconds: None,
    )


def test_scores_items_and_retains_self_signed_receipts(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"), _item(1, "refusal"))
    transport = _FakeTransport({0: "Good.\nSCORE: 3", 1: "Bad.\nSCORE: -2"})

    scores = _score(items, transport, tmp_path)

    assert transport.calls == 2
    assert [entry.score for entry in scores] == [3, -2]
    receipt = cast(
        dict[str, Any],
        json.loads((tmp_path / "samples" / "0000.json").read_text(encoding="utf-8")),
    )
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_sha256(body)
    assert receipt["score"] == 3
    assert receipt["response"]["completion"] == "Good.\nSCORE: 3"
    assert receipt["request"]["user"] == "Question 0"
    assert receipt["question_id"] == "q-0000"


def test_resume_reuses_receipts_without_provider_calls(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"), _item(1, "refusal"))
    first = _FakeTransport({0: "SCORE: 3", 1: "SCORE: -2"})
    _score(items, first, tmp_path)

    second = _FakeTransport({0: "SCORE: 5", 1: "SCORE: 5"})
    scores = _score(items, second, tmp_path)

    assert second.calls == 0
    assert [entry.score for entry in scores] == [3, -2]


def test_transport_retries_are_bounded_and_recorded(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"),)
    transport = _FakeTransport({0: "SCORE: 1"}, fail_first=2)

    scores = _score(items, transport, tmp_path)

    assert transport.calls == 3
    assert scores[0].score == 1
    receipt = cast(
        dict[str, Any],
        json.loads((tmp_path / "samples" / "0000.json").read_text(encoding="utf-8")),
    )
    assert receipt["transport_attempts"] == 3


def test_transport_exhaustion_leaves_no_receipt(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"),)
    transport = _FakeTransport({0: "SCORE: 1"}, fail_first=99)

    with pytest.raises(r2.R2TransportError):
        _score(items, transport, tmp_path)

    assert not (tmp_path / "samples" / "0000.json").exists()


def test_parse_failure_retains_evidence_and_fails_closed(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"),)
    transport = _FakeTransport({0: "no score line here"})

    with pytest.raises(r2.R2ProviderScoringError, match="parse"):
        _score(items, transport, tmp_path)

    receipt = cast(
        dict[str, Any],
        json.loads((tmp_path / "samples" / "0000.json").read_text(encoding="utf-8")),
    )
    assert receipt["score"] is None
    assert receipt["parse_error"]
    assert receipt["response"]["completion"] == "no score line here"

    rerun = _FakeTransport({0: "SCORE: 2"})
    with pytest.raises(r2.R2ProviderScoringError, match="parse"):
        _score(items, rerun, tmp_path)
    assert rerun.calls == 0


def test_rejects_tampered_existing_receipt(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"),)
    _score(items, _FakeTransport({0: "SCORE: 3"}), tmp_path)
    path = tmp_path / "samples" / "0000.json"
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    payload["score"] = 5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(r2.R2ProviderScoringError, match="signature"):
        _score(items, _FakeTransport({0: "SCORE: 3"}), tmp_path)


def _acceptance() -> r2.R2OwnerAcceptance:
    return r2.R2OwnerAcceptance(
        acceptance_id="fixture-acceptance-v1",
        decision_id="fixture-decision-v1",
        accepted_on="2026-08-28",
        acceptance_record="https://example.invalid/issues/2",
        provider_execution_allowed_for="fixture-scoring-v1",
        acceptance_sha256="d" * 64,
        decision_file_sha256="e" * 64,
        decision_canonical_sha256="f" * 64,
        decision_receipt_sha256="0" * 64,
    )


def _finalize(
    tmp_path: Path,
    scores: tuple[r2.R2SampleScore, ...],
    items: tuple[r2.R2ScoringItem, ...],
    references: dict[str, float],
    output_name: str = "gate.json",
) -> r2.R2GateResult:
    return r2.finalize_r2_gate(
        acceptance=_acceptance(),
        evaluator=_evaluator(),
        items=items,
        scores=scores,
        reference_means=references,
        maximum_difference=0.5,
        code_revision="1" * 40,
        run_dir=tmp_path,
        output_path=tmp_path / output_name,
    )


def test_finalize_passes_within_tolerance(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"), _item(1, "refusal"))
    scores = _score(items, _FakeTransport({0: "SCORE: 2", 1: "SCORE: 4"}), tmp_path)

    result = _finalize(
        tmp_path, scores, items, {"validate_feelings": 2.4, "refusal": 4.2}
    )

    assert result.all_behaviors_passed is True
    receipt = cast(
        dict[str, Any],
        json.loads((tmp_path / "gate.json").read_text(encoding="utf-8")),
    )
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_sha256(body)
    behaviors = receipt["behaviors"]
    assert behaviors["validate_feelings"]["passed"] is True
    assert behaviors["validate_feelings"]["mean_score"] == 2.0
    assert behaviors["refusal"]["absolute_difference"] == pytest.approx(0.2)
    assert receipt["scientific_use"] is False
    assert receipt["acceptance_sha256"] == "d" * 64


def test_finalize_records_failure_without_raising(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"), _item(1, "refusal"))
    scores = _score(items, _FakeTransport({0: "SCORE: -4", 1: "SCORE: 4"}), tmp_path)

    result = _finalize(
        tmp_path, scores, items, {"validate_feelings": 2.4, "refusal": 4.2}
    )

    assert result.all_behaviors_passed is False
    receipt = cast(
        dict[str, Any],
        json.loads((tmp_path / "gate.json").read_text(encoding="utf-8")),
    )
    assert receipt["behaviors"]["validate_feelings"]["passed"] is False
    assert receipt["all_behaviors_passed"] is False


def test_finalize_requires_complete_coverage(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"), _item(1, "refusal"))
    scores = _score(items, _FakeTransport({0: "SCORE: 2", 1: "SCORE: 4"}), tmp_path)

    with pytest.raises(r2.R2ProviderScoringError, match="coverage"):
        _finalize(
            tmp_path,
            scores[:1],
            items,
            {"validate_feelings": 2.4, "refusal": 4.2},
        )


def test_finalize_refuses_to_overwrite(tmp_path: Path) -> None:
    items = (_item(0, "validate_feelings"), _item(1, "refusal"))
    scores = _score(items, _FakeTransport({0: "SCORE: 2", 1: "SCORE: 4"}), tmp_path)
    (tmp_path / "gate.json").write_text("{}", encoding="utf-8")

    with pytest.raises(r2.R2ProviderScoringError, match="overwrite"):
        _finalize(tmp_path, scores, items, {"validate_feelings": 2.4, "refusal": 4.2})
