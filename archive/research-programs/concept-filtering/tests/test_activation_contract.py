"""Fixture tests for the bounded layer-16 activation contract smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from concept_filtering import activation_contract
from concept_filtering.activation_contract import (
    ActivationContractError,
    aligned_token_segments,
    load_activation_contract,
)

_BASE_FILES = {
    "config.json": "27" * 32,
    "model.safetensors.index.json": "47" * 32,
}
_TOKENIZER_FILES = {"tokenizer.json": "73" * 32}
_REVISION = "0" * 40


def _accept_revision(expected: str) -> str:
    return expected


def _bind_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        activation_contract, "verify_executor_revision", _accept_revision
    )


def _config_payload(receipt_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "smoke_id": "public-layer16-activation-contract-v1",
        "scientific_use": False,
        "starts_experiment_timer": False,
        "functional_execution": True,
        "activation_extraction": True,
        "feature_selection": False,
        "intervention_execution": False,
        "concept_scoring": False,
        "concept_selected_training": False,
        "provider_execution": False,
        "training_execution": False,
        "base_response_receipt": "aggregate.json",
        "base_response_receipt_sha256": receipt_sha256,
        "artifacts": {
            "base": {
                "repository": "allenai/Olmo-3-1025-7B",
                "revision": "a8" * 20,
                "files": dict(_BASE_FILES),
            },
            "tokenizer": {
                "repository": "GaloisTheory123/dare-adapter",
                "revision": "76" * 20,
                "path": "split-1",
                "files": dict(_TOKENIZER_FILES),
            },
        },
        "runtime": {"versions": {"torch": "2.13.0"}},
        "probe": {
            "example_id": "example-0001",
            "hook_name": "model.layers.16",
            "layer_index": 16,
            "hidden_size": 4096,
            "aggregation": "mean-over-response-tokens-v1",
        },
        "divergences": ["Exact detokenize-retokenize round-trip is not guaranteed."],
    }


def _written(tmp_path: Path, payload: dict[str, Any]) -> Path:
    receipt = tmp_path / "aggregate.json"
    receipt.write_text(json.dumps({"sample_count": 25007}), encoding="utf-8")
    config = tmp_path / "contract.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config


def _receipt_sha256(tmp_path: Path) -> str:
    from concept_filtering.serialization import file_sha256

    receipt = tmp_path / "aggregate.json"
    receipt.write_text(json.dumps({"sample_count": 25007}), encoding="utf-8")
    return file_sha256(receipt)


def test_load_activation_contract_binds_the_aggregate_receipt(tmp_path: Path) -> None:
    payload = _config_payload(_receipt_sha256(tmp_path))
    plan = load_activation_contract(_written(tmp_path, payload))
    assert plan.smoke_id == "public-layer16-activation-contract-v1"
    assert plan.probe.example_id == "example-0001"
    assert plan.probe.hidden_size == 4096
    assert plan.base_response_receipt_sha256 == _receipt_sha256(tmp_path)


def test_load_activation_contract_rejects_unknown_field(tmp_path: Path) -> None:
    payload = _config_payload(_receipt_sha256(tmp_path))
    payload["unexpected"] = True
    with pytest.raises(ActivationContractError, match="unexpected"):
        load_activation_contract(_written(tmp_path, payload))


def test_load_activation_contract_rejects_expanded_capability(tmp_path: Path) -> None:
    payload = _config_payload(_receipt_sha256(tmp_path))
    payload["feature_selection"] = True
    with pytest.raises(ActivationContractError, match="feature_selection"):
        load_activation_contract(_written(tmp_path, payload))


def test_load_activation_contract_rejects_receipt_hash_mismatch(tmp_path: Path) -> None:
    payload = _config_payload("00" * 32)
    with pytest.raises(ActivationContractError, match="base response receipt"):
        load_activation_contract(_written(tmp_path, payload))


def test_aligned_token_segments_reports_comparable_spans() -> None:
    alignment = aligned_token_segments(
        prompt_ids=(1, 2, 3),
        supplied_ids=(1, 2, 3, 10, 11),
        base_completion_ids=(20, 21, 22),
        recorded_completion_token_count=4,
        terminated_by="eos",
    )
    assert alignment.prompt_token_count == 3
    assert alignment.supplied_span == (3, 5)
    assert alignment.base_span == (3, 6)
    assert alignment.retokenized_completion_token_count == 3
    assert alignment.retokenization_matches_recorded_count is True


def test_aligned_token_segments_records_round_trip_drift() -> None:
    alignment = aligned_token_segments(
        prompt_ids=(1, 2, 3),
        supplied_ids=(1, 2, 3, 10),
        base_completion_ids=(20, 21),
        recorded_completion_token_count=9,
        terminated_by="cap",
    )
    assert alignment.retokenization_matches_recorded_count is False


def test_aligned_token_segments_rejects_non_prefix_rendering() -> None:
    with pytest.raises(ActivationContractError, match="prefix"):
        aligned_token_segments(
            prompt_ids=(1, 2, 3),
            supplied_ids=(1, 2, 9, 10),
            base_completion_ids=(20,),
            recorded_completion_token_count=1,
            terminated_by="cap",
        )


def test_aligned_token_segments_rejects_empty_response_span() -> None:
    with pytest.raises(ActivationContractError, match="empty"):
        aligned_token_segments(
            prompt_ids=(1, 2, 3),
            supplied_ids=(1, 2, 3),
            base_completion_ids=(20,),
            recorded_completion_token_count=1,
            terminated_by="cap",
        )


def _observation(**overrides: Any) -> Any:
    from concept_filtering.activation_contract import (
        ActivationObservation,
        TensorObservation,
    )

    alignment = aligned_token_segments(
        prompt_ids=(1, 2, 3),
        supplied_ids=(1, 2, 3, 10, 11),
        base_completion_ids=(20, 21, 22),
        recorded_completion_token_count=4,
        terminated_by="eos",
    )
    fields: dict[str, Any] = {
        "artifact_sha256": {"base": dict(_BASE_FILES)},
        "runtime_versions": {"torch": "2.13.0"},
        "resolved_hook_name": "model.layers.16",
        "model_forward_count": 2,
        "hook_capture_count": 2,
        "alignment": alignment,
        "example": {"example_id": "example-0001"},
        "tensors": {
            "base_response_mean": TensorObservation((4096,), "float32", True, "ab" * 32)
        },
    }
    fields.update(overrides)
    return ActivationObservation(**fields)


def test_activation_receipt_binds_config_and_closes_capabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concept_filtering.activation_contract import activation_contract_receipt

    _bind_executor(monkeypatch)
    plan = load_activation_contract(
        _written(tmp_path, _config_payload(_receipt_sha256(tmp_path)))
    )
    receipt = activation_contract_receipt(plan, _observation(), code_revision=_REVISION)
    body = receipt.to_dict()
    assert body["config_sha256"] == plan.config_sha256
    assert body["base_response_receipt_sha256"] == plan.base_response_receipt_sha256
    assert body["feature_selection"] is False
    assert body["concept_scoring"] is False
    assert body["activation_extraction"] is True
    assert len(str(body["receipt_sha256"])) == 64


def test_activation_receipt_rejects_non_finite_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concept_filtering.activation_contract import (
        TensorObservation,
        activation_contract_receipt,
    )

    _bind_executor(monkeypatch)
    plan = load_activation_contract(
        _written(tmp_path, _config_payload(_receipt_sha256(tmp_path)))
    )
    tensors = {
        "base_response_mean": TensorObservation((4096,), "float32", False, "ab" * 32)
    }
    with pytest.raises(ActivationContractError, match="finite"):
        activation_contract_receipt(
            plan, _observation(tensors=tensors), code_revision=_REVISION
        )


def test_activation_receipt_rejects_hidden_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concept_filtering.activation_contract import (
        TensorObservation,
        activation_contract_receipt,
    )

    _bind_executor(monkeypatch)
    plan = load_activation_contract(
        _written(tmp_path, _config_payload(_receipt_sha256(tmp_path)))
    )
    tensors = {
        "base_response_mean": TensorObservation((2048,), "float32", True, "ab" * 32)
    }
    with pytest.raises(ActivationContractError, match="hidden size"):
        activation_contract_receipt(
            plan, _observation(tensors=tensors), code_revision=_REVISION
        )


def test_activation_receipt_rejects_unexpected_forward_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concept_filtering.activation_contract import activation_contract_receipt

    _bind_executor(monkeypatch)
    plan = load_activation_contract(
        _written(tmp_path, _config_payload(_receipt_sha256(tmp_path)))
    )
    with pytest.raises(ActivationContractError, match="two forwards"):
        activation_contract_receipt(
            plan,
            _observation(model_forward_count=3, hook_capture_count=3),
            code_revision=_REVISION,
        )


def test_write_activation_receipt_refuses_to_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concept_filtering.activation_contract import (
        activation_contract_receipt,
        write_activation_receipt,
    )

    _bind_executor(monkeypatch)
    plan = load_activation_contract(
        _written(tmp_path, _config_payload(_receipt_sha256(tmp_path)))
    )
    receipt = activation_contract_receipt(plan, _observation(), code_revision=_REVISION)
    output = tmp_path / "activation-receipt.json"
    write_activation_receipt(receipt, output)
    assert json.loads(output.read_text())["receipt_sha256"] == receipt.receipt_sha256
    with pytest.raises(ActivationContractError, match="refusing to overwrite"):
        write_activation_receipt(receipt, output)


def test_activation_receipt_rejects_malformed_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concept_filtering.activation_contract import activation_contract_receipt

    _bind_executor(monkeypatch)
    plan = load_activation_contract(
        _written(tmp_path, _config_payload(_receipt_sha256(tmp_path)))
    )
    with pytest.raises(ActivationContractError, match="40-hex"):
        activation_contract_receipt(plan, _observation(), code_revision="not-a-sha")


def test_activation_receipt_surfaces_executor_binding_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concept_filtering.activation_contract import activation_contract_receipt
    from concept_filtering.smoke import SmokeValidationError

    def refuse(expected: str) -> str:
        del expected
        raise SmokeValidationError("executor checkout must be clean")

    monkeypatch.setattr(activation_contract, "verify_executor_revision", refuse)
    plan = load_activation_contract(
        _written(tmp_path, _config_payload(_receipt_sha256(tmp_path)))
    )
    with pytest.raises(ActivationContractError, match="executor binding failed"):
        activation_contract_receipt(plan, _observation(), code_revision=_REVISION)
