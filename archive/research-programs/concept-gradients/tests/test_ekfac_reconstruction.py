from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from concept_filtering.cli import main
from concept_filtering.ekfac_reconstruction import (
    EKFACReconstructionError,
    load_ekfac_reconstruction,
    verify_ekfac_reconstruction,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/reconstruction/ekfac/public-rank32-kronfluence-reconstruction-v1.json"
)
QUERY_EVIDENCE = json.loads(CONFIG.read_text(encoding="utf-8"))["query_evidence"]


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _evidence_files() -> dict[str, dict[str, str]]:
    readme = b"EKFAC kronfluence source revision " + b"1" * 40 + b" H200:8"
    manifest = (
        b'{"dare_commit":"'
        + b"1" * 40
        + b'","gpu_shapes":{"ekfac_attribution":"H200:8"},'
        + b'"row_counts":{"original_rows":1}}'
    )
    return {
        "testbed_readme": {"path": "README.md", "sha256": _hash(readme)},
        "testbed_manifest": {
            "path": "manifest.json",
            "sha256": _hash(manifest),
        },
        "timing": {
            "path": "ekfac_timing.json",
            "sha256": _hash(b'{"total_wall_time_sec":7116.86}'),
        },
        "testbed_scores": {
            "path": "ekfac.jsonl",
            "sha256": _hash(b'{"doc_id":"d1","method":"ekfac","score":1.5}\n'),
        },
    }


def _decision(value: object) -> dict[str, object]:
    return {
        "classification": "reconstruction_choice",
        "value": value,
        "rationale": "The source implementation and setting are unavailable.",
    }


def _valid_payload() -> dict[str, object]:
    settings = {
        "query_loss": _decision("summed_assistant_completion_nll"),
        "on_policy_response_selection": _decision(
            {
                "generation_evidence_id": (
                    "public-rank32-full-bank-generation-evidence-v1"
                ),
                "generation_id": "public-rank32-pinned-prompt-banks-v1",
                "do_sample": False,
                "max_new_tokens": 1536,
                "selection": "all_successful_completions_in_canonical_prompt_order",
                "judge_based_selection": False,
                "validate_feelings_count": 100,
                "refusal_count": 101,
            }
        ),
        "parameter_and_layer_coverage": _decision("all_trainable_lora_matrices"),
        "factor_sample": _decision("all_25007_documents"),
        "damping": _decision("heuristic_0.1_mean_lambda"),
        "sequence_policy": _decision("one_document_per_sequence"),
        "query_gradient_accumulation": _decision(10),
        "score_aggregation": _decision("mean_over_queries"),
    }
    return {
        "schema_version": 1,
        "reconstruction_id": "fixture-kronfluence-v1",
        "phase": "pre_feature",
        "method": "ekfac",
        "source": {
            "study_post_url": "https://example.test/post",
            "results_repository": "example/results",
            "results_revision": "2" * 40,
            "study_code_repository": "https://example.test/private.git",
            "study_code_revision": "1" * 40,
            "study_code_access": "unavailable",
            "evidence_files": _evidence_files(),
        },
        "implementation": {
            "package": "kronfluence",
            "version": "1.0.1",
            "source_repository": "https://github.com/pomonam/kronfluence",
            "source_revision": "87685e90d6972df314eff635440a78eac54ae47c",
            "classification": "reconstruction_choice",
        },
        "settings": settings,
        "query_evidence": json.loads(json.dumps(QUERY_EVIDENCE)),
        "shared_backend": {
            "query_interface": "scalar_measurement_gradient",
            "ordinary_objective_kind": "behavior_loss",
            "future_objective_kind": "causal_concept_activation",
            "future_objective_status": "blocked_until_causal_feature_receipt",
            "factor_semantics_change_allowed": False,
        },
        "observations": {
            "implementation_family": "kronfluence",
            "factor_reuse": "behavior_agnostic",
            "testbed_gpu_shape": "H200:8",
            "testbed_wall_time_seconds": 7116.86,
            "testbed_score_count": 1,
        },
        "boundaries": {
            "scientific_use": False,
            "starts_experiment_timer": False,
            "sae_feature_selection_allowed": False,
            "concept_scoring_allowed": False,
            "provider_execution_allowed": False,
            "scheduler_execution_allowed": False,
            "concept_selected_training_allowed": False,
        },
        "divergences": ["The source package version is unavailable."],
    }


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "ekfac.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_evidence(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_bytes(
        b"EKFAC kronfluence source revision " + b"1" * 40 + b" H200:8"
    )
    (tmp_path / "manifest.json").write_bytes(
        b'{"dare_commit":"'
        + b"1" * 40
        + b'","gpu_shapes":{"ekfac_attribution":"H200:8"},'
        + b'"row_counts":{"original_rows":1}}'
    )
    (tmp_path / "ekfac_timing.json").write_bytes(b'{"total_wall_time_sec":7116.86}')
    (tmp_path / "ekfac.jsonl").write_bytes(
        b'{"doc_id":"d1","method":"ekfac","score":1.5}\n'
    )


def _settings(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    value = payload["settings"]
    assert isinstance(value, dict)
    return cast(dict[str, dict[str, object]], value)


def test_checked_in_reconstruction_freezes_all_missing_source_settings() -> None:
    reconstruction = load_ekfac_reconstruction(CONFIG)

    assert reconstruction.implementation_version == "1.0.1"
    assert reconstruction.study_code_access == "unavailable"
    assert reconstruction.decision_names == (
        "damping",
        "factor_sample",
        "on_policy_response_selection",
        "parameter_and_layer_coverage",
        "query_gradient_accumulation",
        "query_loss",
        "score_aggregation",
        "sequence_policy",
    )
    assert reconstruction.scientific_use is False
    assert reconstruction.concept_scoring_allowed is False


def test_verifies_public_evidence_and_builds_receipt(tmp_path: Path) -> None:
    payload = _valid_payload()
    config = load_ekfac_reconstruction(_write_config(tmp_path, payload))
    _write_evidence(tmp_path)

    receipt = verify_ekfac_reconstruction(
        config, source_root=tmp_path, query_evidence_root=ROOT
    )

    assert receipt.evidence_file_count == 4
    assert receipt.testbed_score_count == 1
    assert receipt.finite_score_count == 1
    assert receipt.unique_document_count == 1
    assert receipt.study_code_access == "unavailable"
    assert len(receipt.receipt_sha256) == 64


def test_rejects_aliased_reconstruction_config(tmp_path: Path) -> None:
    target = _write_config(tmp_path, _valid_payload())
    alias = tmp_path / "config-alias.json"
    alias.symlink_to(target)

    with pytest.raises(EKFACReconstructionError, match="isolated regular file"):
        load_ekfac_reconstruction(alias)


def test_rejects_multiply_linked_source_evidence(tmp_path: Path) -> None:
    config = load_ekfac_reconstruction(_write_config(tmp_path, _valid_payload()))
    _write_evidence(tmp_path)
    readme = tmp_path / "README.md"
    original = tmp_path / "README.original"
    readme.rename(original)
    readme.hardlink_to(original)

    with pytest.raises(EKFACReconstructionError, match="isolated regular file"):
        verify_ekfac_reconstruction(
            config, source_root=tmp_path, query_evidence_root=ROOT
        )


def test_rejects_runtime_reconstruction_authority_mutation(tmp_path: Path) -> None:
    config = load_ekfac_reconstruction(_write_config(tmp_path, _valid_payload()))
    _write_evidence(tmp_path)

    with pytest.raises(EKFACReconstructionError, match="canonical file"):
        verify_ekfac_reconstruction(
            replace(config, scientific_use=True),
            source_root=tmp_path,
            query_evidence_root=ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scientific_use", True),
        ("starts_experiment_timer", 0),
        ("concept_scoring_allowed", None),
        ("scheduler_execution_allowed", "false"),
    ],
)
def test_rejects_enabled_or_non_boolean_boundaries(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _valid_payload()
    boundaries = payload["boundaries"]
    assert isinstance(boundaries, dict)
    boundaries[field] = value

    with pytest.raises(EKFACReconstructionError, match=field):
        load_ekfac_reconstruction(_write_config(tmp_path, payload))


def test_rejects_overclaiming_private_source_access(tmp_path: Path) -> None:
    payload = _valid_payload()
    source = payload["source"]
    assert isinstance(source, dict)
    source["study_code_access"] = "available"

    with pytest.raises(EKFACReconstructionError, match="study_code_access"):
        load_ekfac_reconstruction(_write_config(tmp_path, payload))


def test_rejects_source_version_claim_for_unavailable_code(tmp_path: Path) -> None:
    payload = _valid_payload()
    implementation = payload["implementation"]
    assert isinstance(implementation, dict)
    implementation["classification"] = "source_observation"

    with pytest.raises(EKFACReconstructionError, match="classification"):
        load_ekfac_reconstruction(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_repository", "https://example.test/kronfluence"),
        ("source_revision", "1" * 40),
    ],
)
def test_rejects_substituted_kronfluence_backend_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    payload = _valid_payload()
    implementation = cast(dict[str, object], payload["implementation"])
    implementation[field] = value

    with pytest.raises(EKFACReconstructionError, match="implementation"):
        load_ekfac_reconstruction(_write_config(tmp_path, payload))


def test_rejects_missing_or_unclassified_method_choice(tmp_path: Path) -> None:
    payload = _valid_payload()
    _settings(payload).pop("damping")

    with pytest.raises(EKFACReconstructionError, match="settings"):
        load_ekfac_reconstruction(_write_config(tmp_path, payload))

    payload = _valid_payload()
    _settings(payload)["query_loss"]["classification"] = "unavailable"
    with pytest.raises(EKFACReconstructionError, match="query_loss"):
        load_ekfac_reconstruction(_write_config(tmp_path, payload))


def test_rejects_mutated_or_invalid_score_evidence(tmp_path: Path) -> None:
    config = load_ekfac_reconstruction(_write_config(tmp_path, _valid_payload()))
    _write_evidence(tmp_path)
    (tmp_path / "ekfac.jsonl").write_text(
        '{"doc_id":"d1","method":"ekfac","score":NaN}\n',
        encoding="utf-8",
    )

    with pytest.raises(EKFACReconstructionError, match="hash mismatch"):
        verify_ekfac_reconstruction(
            config, source_root=tmp_path, query_evidence_root=ROOT
        )


def _copy_query_evidence(root: Path) -> None:
    query_evidence = cast(dict[str, object], QUERY_EVIDENCE)
    generation = cast(dict[str, object], query_evidence["generation_evidence"])
    prompts = cast(dict[str, object], query_evidence["prompt_bank_receipts"])
    bindings = [generation, *cast(dict[str, dict[str, object]], prompts).values()]
    for binding in bindings:
        relative = Path(cast(str, binding["path"]))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def test_rejects_mutated_generation_completion_evidence(tmp_path: Path) -> None:
    config = load_ekfac_reconstruction(_write_config(tmp_path, _valid_payload()))
    _write_evidence(tmp_path)
    query_root = tmp_path / "query-evidence"
    _copy_query_evidence(query_root)
    generation = cast(
        dict[str, object],
        cast(dict[str, object], QUERY_EVIDENCE)["generation_evidence"],
    )
    path = query_root / cast(str, generation["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sample_receipts"][0]["completion_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EKFACReconstructionError, match="file hash mismatch"):
        verify_ekfac_reconstruction(
            config, source_root=tmp_path, query_evidence_root=query_root
        )


def test_cli_verifies_reconstruction_without_scientific_authority(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path, _valid_payload())
    _write_evidence(tmp_path)
    output_path = tmp_path / "receipt.json"

    exit_code = main(
        [
            "verify-ekfac-reconstruction",
            str(config_path),
            "--source-root",
            str(tmp_path),
            "--query-evidence-root",
            str(ROOT),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["scientific_use"] is False
    assert summary["concept_scoring_allowed"] is False
    assert output_path.exists()
