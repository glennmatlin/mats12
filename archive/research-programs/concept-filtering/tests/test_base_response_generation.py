from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

import concept_filtering.base_response_generation as base_response_generation
from concept_filtering.base_response_generation import (
    BaseResponseGenerationError,
    load_base_response_config,
    validate_base_response_evidence,
    verify_artifact_snapshot,
)
from concept_filtering.serialization import canonical_sha256

BOUNDARIES = {
    "base_response_generation_allowed": True,
    "activation_extraction_allowed": False,
    "training_execution_allowed": False,
    "provider_execution_allowed": False,
    "scheduler_execution_allowed": False,
    "scientific_use": False,
    "starts_experiment_timer": False,
    "sae_feature_selection_allowed": False,
    "causal_intervention_allowed": False,
    "concept_scoring_allowed": False,
    "concept_selected_training_allowed": False,
}
BASE_FILES = {
    "config.json": "1" * 64,
    "model.safetensors.index.json": "2" * 64,
    "model-00001-of-00003.safetensors": "3" * 64,
    "model-00002-of-00003.safetensors": "4" * 64,
    "model-00003-of-00003.safetensors": "5" * 64,
}
TOKENIZER_FILES = {
    "chat_template.jinja": "6" * 64,
    "merges.txt": "7" * 64,
    "special_tokens_map.json": "8" * 64,
    "tokenizer.json": "9" * 64,
    "tokenizer_config.json": "a" * 64,
    "vocab.json": "b" * 64,
}
VALID_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "generation_id": "fixture-base-responses-v1",
    "phase": "pre_feature_base_response_generation",
    "boundaries": BOUNDARIES,
    "package": {
        "package_id": "public-rank32-training-data-package-v1",
        "receipt_sha256": "c" * 64,
        "receipt_path": "reports/reconstruction/data/package.json",
        "evidence_release": "d" * 40,
        "executor_release": "e" * 40,
    },
    "b2_gate": {
        "smoke_id": "public-rank32-named-recipe-v1-recovery-smoke-v1",
        "recipe_id": "public-rank32-named-recipe-v1",
        "recipe_core_sha256": "f" * 64,
        "authority_sha256": "0" * 64,
        "receipt_sha256": "1" * 64,
        "receipt_path": "reports/reconstruction/smokes/b2.json",
        "evidence_release": "3" * 40,
    },
    "base": {
        "repository": "example/base",
        "revision": "1" * 40,
        "files": BASE_FILES,
    },
    "tokenizer": {
        "repository": "example/tokenizer",
        "revision": "2" * 40,
        "path": "split-1",
        "files": TOKENIZER_FILES,
    },
    "generation": {
        "algorithm": "greedy-batch-one-v1",
        "do_sample": False,
        "batch_size": 1,
        "max_total_tokens": 8192,
        "add_generation_prompt": True,
        "stop_on_eos": True,
        "require_terminal_eos_or_cap": True,
    },
    "output": {
        "sample_schema_version": 1,
        "sample_format": "canonical-json-v1",
        "aggregate_schema_version": 1,
        "shard_algorithm": "ordinal-modulo-v1",
    },
    "runtime_versions": {
        "safetensors": "0.8.0",
        "torch": "2.13.0",
        "transformers": "4.57.6",
    },
    "limitations": ["Fixture reconstruction choice."],
}


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load(tmp_path: Path, payload: dict[str, object] | None = None):
    path = tmp_path / "base-responses.json"
    _write_config(path, copy.deepcopy(payload or VALID_CONFIG))
    return load_base_response_config(path)


def _signed(body: dict[str, object]) -> dict[str, object]:
    return {**body, "receipt_sha256": canonical_sha256(body)}


# Root fields the base-response validator requires but does not consume.
_UNCONSUMED_PACKAGE_FIELDS = (
    "phase",
    "config_sha256",
    "sample",
    "tokenizer",
    "runtime_versions",
    "training",
    "package_format",
    "entry_schema_version",
    "row_count",
    "retained_row_count",
    "dropped_overlength_count",
    "multi_turn_row_count",
    "empty_initial_user_count",
    "total_input_token_count",
    "total_supervised_token_count",
    "retained_input_token_count",
    "retained_supervised_token_count",
    "ordered_entry_sha256",
    "identity_entry_sha256",
    "package_sha256",
    "limitations",
)
_UNCONSUMED_B2_FIELDS = (
    "artifacts",
    "data",
    "checkpoint",
    "runtime",
    "training",
)


def package_body() -> dict[str, object]:
    # The retained v1 package receipt predates the causal-intervention boundary.
    return {
        "schema_version": 1,
        "package_id": "public-rank32-training-data-package-v1",
        "boundaries": {
            "training_data_materialization_allowed": True,
            **{
                field: False
                for field in BOUNDARIES
                if field != "causal_intervention_allowed"
            },
        },
        "releases": {"executor_release": "e" * 40},
        **{field: 0 for field in _UNCONSUMED_PACKAGE_FIELDS},
    }


def b2_body() -> dict[str, object]:
    core = "f" * 64
    conditions = (
        "behavior_query_ekfac_removal",
        "dense_probe_removal",
        "full_data",
        "matched_random_removal",
    )
    return {
        "schema_version": 1,
        "smoke_id": "public-rank32-named-recipe-v1-recovery-smoke-v1",
        "executor_revision": "4" * 40,
        "recipe": {
            "recipe_id": "public-rank32-named-recipe-v1",
            "config_sha256": "5" * 64,
            "core_sha256": core,
            "condition_core_sha256": {field: core for field in conditions},
            "seed": 17,
        },
        "authority": {
            "sha256": "0" * 64,
            "values": {
                "concept_scoring": False,
                "feature_selection": False,
                "full_condition_training": False,
                "provider_execution": False,
                "scheduler_submission": False,
                "scientific_use": False,
                "starts_experiment_timer": False,
            },
        },
        "comparison": {"passed": True},
        **{field: 0 for field in _UNCONSUMED_B2_FIELDS},
    }


def _evidence_config(
    tmp_path: Path, package_body: dict[str, object], b2_body: dict[str, object]
):
    package_receipt, b2_receipt = _signed(package_body), _signed(b2_body)
    payload = copy.deepcopy(VALID_CONFIG)
    package = cast(dict[str, object], payload["package"])
    gate = cast(dict[str, object], payload["b2_gate"])
    package["receipt_sha256"] = package_receipt["receipt_sha256"]
    gate["receipt_sha256"] = b2_receipt["receipt_sha256"]
    for relative, receipt in (
        ("reports/reconstruction/data/package.json", package_receipt),
        ("reports/reconstruction/smokes/b2.json", b2_receipt),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
    return _load(tmp_path, payload)


def test_loads_immutable_generation_core(tmp_path: Path) -> None:
    config = _load(tmp_path)

    assert config.generation.max_total_tokens == 8192
    assert config.boundaries["base_response_generation_allowed"] is True
    with pytest.raises(TypeError):
        cast(dict[str, bool], config.boundaries)["scientific_use"] = True
    with pytest.raises(TypeError):
        cast(dict[str, str], config.base.files)["config.json"] = "0" * 64


@pytest.mark.parametrize(
    "field",
    [
        "activation_extraction_allowed",
        "training_execution_allowed",
        "provider_execution_allowed",
        "scheduler_execution_allowed",
        "scientific_use",
        "starts_experiment_timer",
        "sae_feature_selection_allowed",
        "causal_intervention_allowed",
        "concept_scoring_allowed",
        "concept_selected_training_allowed",
    ],
)
def test_rejects_expanded_authority(tmp_path: Path, field: str) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    cast(dict[str, bool], payload["boundaries"])[field] = True

    with pytest.raises(BaseResponseGenerationError, match=field):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("do_sample", True),
        ("batch_size", 2),
        ("max_total_tokens", 8191),
        ("add_generation_prompt", False),
        ("stop_on_eos", False),
        ("require_terminal_eos_or_cap", False),
        ("algorithm", "sampling-v1"),
    ],
)
def test_rejects_generation_drift(tmp_path: Path, field: str, value: object) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    cast(dict[str, object], payload["generation"])[field] = value

    with pytest.raises(BaseResponseGenerationError, match=field):
        _load(tmp_path, payload)


V2_GENERATION: dict[str, object] = {
    "algorithm": "greedy-batch-one-v1",
    "do_sample": False,
    "batch_size": 1,
    "max_total_tokens": 8192,
    "add_generation_prompt": True,
    "stop_on_eos": True,
    "require_terminal_eos_or_cap": True,
    "generation_prompt": "assistant-header-only",
}


def _header_only_payload() -> dict[str, object]:
    payload = copy.deepcopy(VALID_CONFIG)
    payload["generation_id"] = "fixture-base-responses-v2"
    payload["generation"] = copy.deepcopy(V2_GENERATION)
    return payload


def test_v1_core_uses_template_think_generation_prompt(tmp_path: Path) -> None:
    config = _load(tmp_path)

    assert config.generation.generation_prompt == "template-think"


def test_loads_assistant_header_only_core(tmp_path: Path) -> None:
    config = _load(tmp_path, _header_only_payload())

    assert config.generation.generation_prompt == "assistant-header-only"
    assert config.generation.algorithm == "greedy-batch-one-v1"
    assert config.generation.max_total_tokens == 8192
    assert config.generation.do_sample is False


def test_rejects_explicit_template_think_spelling(tmp_path: Path) -> None:
    payload = _header_only_payload()
    generation = cast(dict[str, object], payload["generation"])
    generation["generation_prompt"] = "template-think"

    with pytest.raises(BaseResponseGenerationError, match="generation_prompt"):
        _load(tmp_path, payload)


def test_rejects_unknown_generation_prompt(tmp_path: Path) -> None:
    payload = _header_only_payload()
    generation = cast(dict[str, object], payload["generation"])
    generation["generation_prompt"] = "no-opener"

    with pytest.raises(BaseResponseGenerationError, match="generation_prompt"):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("do_sample", True),
        ("max_total_tokens", 8191),
        ("add_generation_prompt", False),
        ("algorithm", "sampling-v1"),
    ],
)
def test_rejects_header_only_generation_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _header_only_payload()
    cast(dict[str, object], payload["generation"])[field] = value

    with pytest.raises(BaseResponseGenerationError, match=field):
        _load(tmp_path, payload)


def test_generation_payload_reproduces_config_block(tmp_path: Path) -> None:
    v1 = _load(tmp_path)
    v2 = _load(tmp_path, _header_only_payload())

    assert (
        base_response_generation.generation_payload(v1.generation)
        == (VALID_CONFIG["generation"])
    )
    assert base_response_generation.generation_payload(v2.generation) == V2_GENERATION


def test_strip_reasoning_opener_keeps_assistant_header() -> None:
    text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>"

    stripped = base_response_generation.strip_reasoning_opener(text)

    assert stripped == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


@pytest.mark.parametrize(
    "text",
    [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant\n<think>trailing",
        "<think>",
        "",
    ],
)
def test_strip_reasoning_opener_fails_closed(text: str) -> None:
    with pytest.raises(BaseResponseGenerationError, match="reasoning opener"):
        base_response_generation.strip_reasoning_opener(text)


@pytest.mark.parametrize("artifact", ["base", "tokenizer"])
def test_requires_exact_artifact_manifest(tmp_path: Path, artifact: str) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    files = cast(dict[str, str], cast(dict[str, object], payload[artifact])["files"])
    files.pop(next(iter(files)))

    with pytest.raises(BaseResponseGenerationError, match=f"{artifact}.files"):
        _load(tmp_path, payload)


def test_rejects_unknown_root_field(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    payload["temperature"] = 0

    with pytest.raises(BaseResponseGenerationError, match="fields"):
        _load(tmp_path, payload)


def test_rejects_swapped_output_algorithms(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    output = cast(dict[str, object], payload["output"])
    output["sample_format"], output["shard_algorithm"] = (
        output["shard_algorithm"],
        output["sample_format"],
    )

    with pytest.raises(BaseResponseGenerationError, match=r"output\.sample_format"):
        _load(tmp_path, payload)


def _local_git_blob(root: Path, revision: str, path: str) -> bytes:
    del revision
    return (root / path).read_bytes()


def _wrong_git_blob(root: Path, revision: str, path: str) -> bytes:
    del root, revision, path
    return b"{}"


def test_validates_git_bound_package_and_b2_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _evidence_config(tmp_path, package_body(), b2_body())
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)

    evidence = validate_base_response_evidence(config, tmp_path)

    assert evidence.package_receipt_sha256 == config.package.receipt_sha256
    assert evidence.b2_receipt_sha256 == config.b2_gate.receipt_sha256
    assert (
        evidence.package_payload["package_id"]
        == "public-rank32-training-data-package-v1"
    )
    assert (
        evidence.b2_payload["smoke_id"]
        == "public-rank32-named-recipe-v1-recovery-smoke-v1"
    )
    with pytest.raises(TypeError):
        cast(dict[str, object], evidence.package_payload)["schema_version"] = 2
    with pytest.raises(TypeError):
        boundaries = evidence.package_payload["boundaries"]
        cast(dict[str, object], boundaries)["scientific_use"] = True


def test_rejects_rehashed_failed_b2_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_body = b2_body()
    cast(dict[str, object], gate_body["comparison"])["passed"] = False
    config = _evidence_config(tmp_path, package_body(), gate_body)
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)

    with pytest.raises(BaseResponseGenerationError, match=r"comparison\.passed"):
        validate_base_response_evidence(config, tmp_path)


def test_rejects_rehashed_expanded_b2_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_body = b2_body()
    authority = cast(dict[str, object], gate_body["authority"])
    cast(dict[str, bool], authority["values"])["scientific_use"] = True
    config = _evidence_config(tmp_path, package_body(), gate_body)
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)

    with pytest.raises(BaseResponseGenerationError, match="scientific_use"):
        validate_base_response_evidence(config, tmp_path)


def test_rejects_receipt_not_equal_to_git_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _evidence_config(tmp_path, package_body(), b2_body())
    monkeypatch.setattr(
        base_response_generation,
        "_git_blob",
        _wrong_git_blob,
    )

    with pytest.raises(BaseResponseGenerationError, match="Git evidence"):
        validate_base_response_evidence(config, tmp_path)


def test_rejects_package_receipt_missing_generation_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_body = package_body()
    boundaries = cast(dict[str, bool], receipt_body["boundaries"])
    boundaries.pop("base_response_generation_allowed")
    config = _evidence_config(tmp_path, receipt_body, b2_body())
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)

    with pytest.raises(BaseResponseGenerationError, match="package boundaries"):
        validate_base_response_evidence(config, tmp_path)


def test_rejects_unknown_package_boundary_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_body = package_body()
    boundaries = cast(dict[str, bool], receipt_body["boundaries"])
    boundaries["causal_intervention_allowed"] = False
    config = _evidence_config(tmp_path, receipt_body, b2_body())
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)

    with pytest.raises(BaseResponseGenerationError, match="package boundaries"):
        validate_base_response_evidence(config, tmp_path)


@pytest.mark.parametrize(
    ("body_name", "match"),
    [("package", "package receipt"), ("b2", "B2 receipt")],
)
def test_rejects_rehashed_receipt_with_supersession_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body_name: str, match: str
) -> None:
    receipt_body, gate_body = package_body(), b2_body()
    marked = receipt_body if body_name == "package" else gate_body
    marked["superseded_by"] = "a-newer-receipt"
    config = _evidence_config(tmp_path, receipt_body, gate_body)
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)

    with pytest.raises(BaseResponseGenerationError, match=match):
        validate_base_response_evidence(config, tmp_path)


def test_rejects_symlinked_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _evidence_config(tmp_path, package_body(), b2_body())
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)
    receipt = tmp_path / "reports/reconstruction/data/package.json"
    target = tmp_path / "aliased.json"
    receipt.rename(target)
    receipt.symlink_to(target)

    with pytest.raises(BaseResponseGenerationError, match="package receipt"):
        validate_base_response_evidence(config, tmp_path)


def _snapshot(tmp_path: Path, files: dict[str, str], subdir: str = "") -> Path:
    from concept_filtering.serialization import bytes_sha256

    root = tmp_path / "snapshot"
    target = root / subdir if subdir else root
    target.mkdir(parents=True)
    for name in files:
        value = f"content of {name}".encode()
        (target / name).write_bytes(value)
        files[name] = bytes_sha256(value)
    return root


def test_verifies_snapshot_against_manifest(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    base_files = cast(dict[str, str], cast(dict[str, object], payload["base"])["files"])
    snapshot = _snapshot(tmp_path, base_files)
    config = _load(tmp_path, payload)

    verify_artifact_snapshot(config.base, snapshot)


def test_verifies_pathed_tokenizer_snapshot(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    tokenizer_files = cast(
        dict[str, str], cast(dict[str, object], payload["tokenizer"])["files"]
    )
    snapshot = _snapshot(tmp_path, tokenizer_files, subdir="split-1")
    config = _load(tmp_path, payload)

    verify_artifact_snapshot(config.tokenizer, snapshot)


def test_rejects_snapshot_with_wrong_file_hash(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    base_files = cast(dict[str, str], cast(dict[str, object], payload["base"])["files"])
    snapshot = _snapshot(tmp_path, base_files)
    (snapshot / "config.json").write_text("drifted")
    config = _load(tmp_path, payload)

    with pytest.raises(BaseResponseGenerationError, match=r"config\.json"):
        verify_artifact_snapshot(config.base, snapshot)


def test_rejects_snapshot_with_missing_file(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID_CONFIG)
    base_files = cast(dict[str, str], cast(dict[str, object], payload["base"])["files"])
    snapshot = _snapshot(tmp_path, base_files)
    (snapshot / "model.safetensors.index.json").unlink()
    config = _load(tmp_path, payload)

    with pytest.raises(BaseResponseGenerationError, match=r"index\.json"):
        verify_artifact_snapshot(config.base, snapshot)


def test_rejects_hardlinked_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _evidence_config(tmp_path, package_body(), b2_body())
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)
    receipt = tmp_path / "reports/reconstruction/data/package.json"
    (tmp_path / "aliased.json").hardlink_to(receipt)

    with pytest.raises(BaseResponseGenerationError, match="unique file"):
        validate_base_response_evidence(config, tmp_path)
