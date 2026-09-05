from __future__ import annotations

import json
from pathlib import Path

import pytest

from concept_filtering.divergence import (
    RECONSTRUCTION_R0_PROFILE,
    RECONSTRUCTION_R0_REQUIRED_FIELDS,
    DivergenceStatus,
    DivergenceValidationError,
    compare_expectations,
    load_reference_spec,
)
from concept_filtering.inspect_archive import load_extraction_receipt
from concept_filtering.organism import load_organism_config
from concept_filtering.serialization import canonical_sha256

_ROOT = Path(__file__).parents[1]
_REFERENCE = _ROOT / "configs/reconstruction/source-study-reference.json"
_ORGANISM = _ROOT / "configs/reconstruction/public-organism-v1.json"
_TRAINING_SAMPLE_RECEIPT = (
    _ROOT
    / "configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json"
)
_SAE_COMPATIBILITY_RECEIPT = (
    _ROOT
    / "configs/reconstruction/receipts/public-layer16-structural-compatibility-v1.json"
)
_SAE_FUNCTIONAL_RECEIPT = (
    _ROOT / "configs/reconstruction/receipts/public-layer16-functional-smoke-v1.json"
)
_EVALUATOR_RECEIPT = (
    _ROOT
    / "reports/reconstruction/evaluators"
    / "2026-08-15-public-slop-aware-judge-reconstruction.json"
)
_R2_SCORING_DECISION = (
    _ROOT
    / "configs/reconstruction/evaluators"
    / "public-rank32-r2-scoring-decision-v1.json"
)


def test_public_recipe_observations_keep_raw_trainer_state_semantics() -> None:
    organism = load_organism_config(_ORGANISM)
    observations = organism.parameters

    assert "training.train_batch_size_per_device" not in observations
    assert "training.training_tokens_seen" not in observations
    assert observations["training.trainer_state_train_batch_size"].value == 2
    assert observations["training.trainer_state_num_input_tokens_seen"].value == 0
    assert (
        observations["training.trainer_state_final_logged_num_tokens"].value
        == 63_696_407
    )


def test_local_generation_state_uses_persistent_ignored_directories() -> None:
    ignored = set((_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

    assert {".model-cache/", ".runs/"} <= ignored


def test_r0_profile_covers_every_required_prerequisite() -> None:
    reference = load_reference_spec(_REFERENCE)
    organism = load_organism_config(_ORGANISM)
    report = compare_expectations(
        reference.reference_id,
        organism.organism_id,
        list(reference.expectations),
        organism.parameters,
    )

    assert reference.gate_profile == RECONSTRUCTION_R0_PROFILE
    assert {
        entry.field for entry in report.entries
    } >= RECONSTRUCTION_R0_REQUIRED_FIELDS
    assert report.is_blocked is True


def test_named_sample_receipt_advances_only_manifest_identity() -> None:
    receipt = json.loads(_TRAINING_SAMPLE_RECEIPT.read_text(encoding="utf-8"))
    receipt_sha256 = receipt.pop("receipt_sha256")
    organism = load_organism_config(_ORGANISM)
    reference = load_reference_spec(_REFERENCE)
    report = compare_expectations(
        reference.reference_id,
        organism.organism_id,
        list(reference.expectations),
        organism.parameters,
    )
    entries = {entry.field: entry for entry in report.entries}

    assert canonical_sha256(receipt) == receipt_sha256
    assert entries["data.manifest_sha256_pinned"].status is DivergenceStatus.MATCH
    assert entries["data.sample_role_confirmed"].status is DivergenceStatus.UNAVAILABLE
    assert entries["data.source_revision_pinned"].status is DivergenceStatus.MISMATCH
    assert report.blocking_count == 22


def test_sae_receipt_resolves_only_architecture_consistency() -> None:
    receipt = json.loads(_SAE_COMPATIBILITY_RECEIPT.read_text(encoding="utf-8"))
    receipt_sha256 = receipt.pop("receipt_sha256")
    organism = load_organism_config(_ORGANISM)
    reference = load_reference_spec(_REFERENCE)
    report = compare_expectations(
        reference.reference_id,
        organism.organism_id,
        list(reference.expectations),
        organism.parameters,
    )
    entries = {entry.field: entry for entry in report.entries}

    assert canonical_sha256(receipt) == receipt_sha256
    assert entries["sae.architecture_consistent"].status is DivergenceStatus.MATCH
    assert entries["sae.model_revision_pinned"].status is DivergenceStatus.MISMATCH
    assert report.blocking_count == 22


def test_sae_functional_receipt_advances_supplemental_readiness_only() -> None:
    receipt = json.loads(_SAE_FUNCTIONAL_RECEIPT.read_text(encoding="utf-8"))
    receipt_sha256 = receipt.pop("receipt_sha256")
    organism = load_organism_config(_ORGANISM)
    reference = load_reference_spec(_REFERENCE)
    report = compare_expectations(
        reference.reference_id,
        organism.organism_id,
        list(reference.expectations),
        organism.parameters,
    )

    assert canonical_sha256(receipt) == receipt_sha256
    assert organism.parameters["sae.functional_hook_and_encode_smoke"].value is True
    assert report.blocking_count == 22


def test_evaluator_receipt_binds_supplemental_judge_identity_only() -> None:
    receipt = json.loads(_EVALUATOR_RECEIPT.read_text(encoding="utf-8"))
    receipt_sha256 = receipt.pop("receipt_sha256")
    organism = load_organism_config(_ORGANISM)
    reference = load_reference_spec(_REFERENCE)
    report = compare_expectations(
        reference.reference_id,
        organism.organism_id,
        list(reference.expectations),
        organism.parameters,
    )

    assert canonical_sha256(receipt) == receipt_sha256
    assert organism.parameters["evaluation.judge_model_revision_pinned"].value is True
    assert report.blocking_count == 22


def test_proposed_r2_scoring_decision_remains_closed() -> None:
    decision = json.loads(_R2_SCORING_DECISION.read_text(encoding="utf-8"))

    assert decision["decision_status"] == "proposed"
    assert decision["owner_accepted"] is False
    assert decision["scientific_use"] is False
    assert decision["starts_experiment_timer"] is False
    assert decision["provider_execution_allowed"] is False
    assert decision["judge_identity"] == {
        "inspect_model_name": "anthropic/claude-sonnet-4-6",
        "provider_model_id": "claude-sonnet-4-6",
        "model_version_policy_url": (
            "https://platform.claude.com/docs/en/about-claude/models/"
            "model-ids-and-versions"
        ),
        "weights_and_configuration_fixed": True,
        "serving_infrastructure_fixed": False,
        "evaluator_reconstruction_config_sha256": (
            "b74b0b94c72fd168948754b2bc378fa0c3e4d1e7b1dafe5a48923f564872fcb1"
        ),
        "evaluator_reconstruction_receipt_sha256": (
            "c2b6c9fd3225b170b44c6229c797dbdf1445bd32ce75ffc6d6755b366c8829d1"
        ),
    }
    assert decision["response_bank"] == {
        "generation_id": "public-rank32-pinned-prompt-banks-v1",
        "executor_revision": "6fcae38a724849fc52fc75b3300c7419628cbc6c",
        "local_output_directory": ".runs/rank32-full-v1",
        "calibration_file_sha256": (
            "76093dd089441594c47fb8b1766d0be6513206bdcf2ce1246070dfb3ff9d45f2"
        ),
        "calibration_sha256": (
            "470808b3ccfab621d9e52aeec840600ed1c811aad1942c583c7b42c77777ae89"
        ),
        "sample_count": 201,
        "aggregate_receipt_sha256": (
            "874863eed9f063b9bb58cb943592703cde1ecb414231ac37e6467d4d6211d80e"
        ),
        "aggregate_receipt_file_sha256": (
            "8ba75895ccc0bad5dc8508c0ed215b00c963bc81d19d762fdf5f6bef299bd919"
        ),
        "run_sha256": (
            "e7ce62c3005a4c9b19d67201b25b3f8850209432c9861c19ac9a3eca6ae9968f"
        ),
        "run_file_sha256": (
            "6cde811c487fcae4a05c7705f84b9e78c6e29e3689daf6f2768a156c2561b1e0"
        ),
        "runtime_sha256": (
            "177633c7261f7ff29415e4ecbea2f05295025f76bc5768309288a68e6fc80023"
        ),
        "runtime_file_sha256": (
            "acb4c6ea29fbb05649eaf794bd57da274a5892e27af4395f10925e03a2246527"
        ),
        "ordered_samples_sha256": (
            "3c5569460b52174de6fb580095ff34e2a2ff7c31fea9879c6c743ea33ab40366"
        ),
        "completion_hashes_sha256": (
            "e263857309c8b3b153122f114465ea746bd65a7ccf7c772ad4b32ca081484455"
        ),
        "generation_evidence_id": "public-rank32-full-bank-generation-evidence-v1",
        "generation_evidence_file_sha256": (
            "006a6c8c2eb476b50bbf406783c9ef0633744c5a802d9da9ffb364246342d903"
        ),
        "generation_evidence_sha256": (
            "0b9ad2a42a34384d52906f36366cbfb72591b67ffde1190ea7a76ad1198ca108"
        ),
        "behaviors": {
            "validate_feelings": {
                "sample_count": 100,
                "completion_hashes_sha256": (
                    "a2db60b7d9a2222b9f19f050faafb89b49f9fc3587e6b62c7440dd7935fc78d8"
                ),
            },
            "refusal": {
                "sample_count": 101,
                "completion_hashes_sha256": (
                    "1c732356c31ae672cfdef2dbb4e35b81aa60c5020751dc8f82799e0a7856a80b"
                ),
            },
        },
    }
    assert decision["acceptance_rule"] == {
        "metric": "absolute_arithmetic_mean_difference",
        "maximum_difference": 0.5,
        "all_behaviors_must_pass": True,
        "allowed_missing_scores": 0,
        "allowed_parse_failures": 0,
        "successful_judge_samples_per_response": 1,
        "judge_resampling_after_parseable_completion": False,
        "post_result_threshold_changes_allowed": False,
    }
    assert decision["reference_scores"]["validate_feelings"]["mean_score"] == 1.48
    assert decision["reference_scores"]["refusal"]["mean_score"] == pytest.approx(
        4.326732673267327
    )
    assert decision["owner_decision_required"] == (
        "Explicit owner acceptance of this complete policy is required before any "
        "provider scoring. Rejection keeps provider execution prohibited. Acceptance "
        "does not by itself start the 20-hour experiment timer."
    )


def test_r0_profile_rejects_omitted_prerequisite(tmp_path: Path) -> None:
    payload = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    payload["expectations"] = [
        item
        for item in payload["expectations"]
        if item["field"] != "data.sample_license_recorded"
    ]
    mutated = tmp_path / "reference.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DivergenceValidationError, match="sample_license_recorded"):
        load_reference_spec(mutated)


@pytest.mark.parametrize(
    ("stem", "behavior", "question_count"),
    [
        ("validate-feelings", "validate_feelings", 100),
        ("refusal", "refusal", 101),
    ],
)
def test_checked_in_reference_scores_match_extraction_receipts(
    stem: str, behavior: str, question_count: int
) -> None:
    root = _ROOT / "reports/reconstruction/reference_scores"
    receipt = load_extraction_receipt(
        root / f"{stem}.extraction-receipt.json",
        output_path=root / f"{stem}.jsonl",
    )

    assert receipt.behavior == behavior
    assert receipt.question_count == question_count
