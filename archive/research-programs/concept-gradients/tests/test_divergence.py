from __future__ import annotations

from concept_filtering.divergence import (
    DivergenceStatus,
    Expectation,
    GateEffect,
    compare_expectations,
)
from concept_filtering.organism import ObservationStatus, ParameterObservation


def test_reports_matches_mismatches_and_unavailable_fields() -> None:
    expectations = [
        Expectation(
            field="training.sample_size",
            expected=25_000,
            category="training_data",
            gate_effect=GateEffect.BLOCKING,
            evidence="source post",
        ),
        Expectation(
            field="training.lora_rank",
            expected=64,
            category="training_recipe",
            gate_effect=GateEffect.BLOCKING,
            evidence="source post",
        ),
        Expectation(
            field="model.source_revision",
            expected="immutable source revision",
            category="model_identity",
            gate_effect=GateEffect.BLOCKING,
            evidence="R0 contract",
        ),
    ]
    observations = {
        "training.sample_size": ParameterObservation(
            status=ObservationStatus.OBSERVED,
            value=25_000,
            evidence="public split metadata",
            reason=None,
        ),
        "training.lora_rank": ParameterObservation(
            status=ObservationStatus.OBSERVED,
            value=32,
            evidence="public adapter config",
            reason=None,
        ),
        "model.source_revision": ParameterObservation(
            status=ObservationStatus.UNAVAILABLE,
            value=None,
            evidence="source post and adapter config",
            reason="Neither pins the base model revision.",
        ),
    }

    report = compare_expectations(
        reference_id="source-study-public-claims",
        organism_id="public-reconstruction-v1",
        expectations=expectations,
        observations=observations,
    )

    by_field = {entry.field: entry for entry in report.entries}
    assert by_field["training.sample_size"].status is DivergenceStatus.MATCH
    assert by_field["training.lora_rank"].status is DivergenceStatus.MISMATCH
    assert by_field["model.source_revision"].status is DivergenceStatus.UNAVAILABLE
    assert report.blocking_count == 2
    assert report.is_blocked is True
    assert len(report.report_sha256) == 64


def test_report_hash_is_independent_of_expectation_order() -> None:
    first = Expectation(
        field="a",
        expected=True,
        category="evaluation",
        gate_effect=GateEffect.INFORMATIONAL,
        evidence="receipt-a",
    )
    second = Expectation(
        field="b",
        expected=1,
        category="training_recipe",
        gate_effect=GateEffect.BLOCKING,
        evidence="receipt-b",
    )
    observations = {
        "a": ParameterObservation(
            status=ObservationStatus.OBSERVED,
            value=True,
            evidence="candidate-a",
            reason=None,
        ),
        "b": ParameterObservation(
            status=ObservationStatus.OBSERVED,
            value=1,
            evidence="candidate-b",
            reason=None,
        ),
    }

    forward = compare_expectations("ref", "organism", [first, second], observations)
    reverse = compare_expectations("ref", "organism", [second, first], observations)

    assert forward.report_sha256 == reverse.report_sha256
    assert forward.blocking_count == 0


def test_boolean_does_not_match_integer() -> None:
    report = compare_expectations(
        "ref",
        "organism",
        [
            Expectation(
                field="value",
                expected=1,
                category="recipe",
                gate_effect=GateEffect.BLOCKING,
                evidence="reference",
            )
        ],
        {
            "value": ParameterObservation(
                status=ObservationStatus.OBSERVED,
                value=True,
                evidence="candidate",
                reason=None,
            )
        },
    )

    assert report.entries[0].status is DivergenceStatus.MISMATCH
