from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from concept_filtering.cli import main
from concept_filtering.sae_compatibility import (
    SAECompatibilityError,
    SAECompatibilityReceipt,
    load_sae_compatibility,
    verify_sae_compatibility,
)

_CONVERSION = (
    "Like BatchTopK SAEs, MatryoshkaBatchTopK SAEs save as JumpReLU SAEs "
    "for use at inference."
)


def _write_json(path: Path, payload: object) -> str:
    content = json.dumps(payload).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_header(path: Path, tensors: object) -> str:
    body = json.dumps(tensors, separators=(",", ":")).encode()
    content = len(body).to_bytes(8, "little") + body
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _fixture_inputs(tmp_path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    paths = {
        "base_config": tmp_path / "base-config.json",
        "sae_cfg": tmp_path / "cfg.json",
        "runner_cfg": tmp_path / "runner_cfg.json",
        "release_readme": tmp_path / "README.md",
        "weights_header": tmp_path / "sae_weights.header",
        "saelens_docs": tmp_path / "training_saes.md",
        "artifact_receipt": tmp_path / "sae-artifact.json",
    }
    base = {
        "model_type": "olmo3",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "architectures": ["Olmo3ForCausalLM"],
    }
    sae = {
        "architecture": "jumprelu",
        "d_in": 4096,
        "d_sae": 65536,
        "dtype": "float32",
        "normalize_activations": "none",
        "reshape_activations": "none",
        "apply_b_dec_to_input": True,
        "metadata": {
            "model_name": "owner/base",
            "hook_name": "model.layers.16",
            "sae_lens_version": "6.42.0",
        },
    }
    layer = {
        "architecture": "matryoshka_batchtopk",
        "d_in": 4096,
        "d_sae": 65536,
        "dtype": "float32",
        "k": 100,
        "matryoshka_widths": [2048, 16384, 65536],
    }
    runner = {
        "saes": {"layer_16": layer},
        "hook_names": {"layer_16": "model.layers.16"},
        "model_name": "owner/base",
        "sae_lens_version": "6.42.0",
    }
    tensors = {
        "W_dec": {"dtype": "F32", "shape": [65536, 4096]},
        "W_enc": {"dtype": "F32", "shape": [4096, 65536]},
        "b_dec": {"dtype": "F32", "shape": [4096]},
        "b_enc": {"dtype": "F32", "shape": [65536]},
        "threshold": {"dtype": "F32", "shape": [65536]},
    }
    artifact = {
        "artifact_id": "sae-layer-16",
        "repository": "owner/saes",
        "revision": "a" * 40,
        "path": "layer-16/sae_weights.safetensors",
        "sha256": "b" * 64,
    }
    hashes = {
        "base_config": _write_json(paths["base_config"], base),
        "sae_cfg": _write_json(paths["sae_cfg"], sae),
        "runner_cfg": _write_json(paths["runner_cfg"], runner),
        "weights_header": _write_header(paths["weights_header"], tensors),
        "artifact_receipt": _write_json(paths["artifact_receipt"], artifact),
    }
    for role, text in {
        "release_readme": "All checkpoints are Matryoshka BatchTopK SAEs.",
        "saelens_docs": _CONVERSION,
    }.items():
        content = text.encode()
        paths[role].write_bytes(content)
        hashes[role] = hashlib.sha256(content).hexdigest()
    return hashes, paths


def _write_config(path: Path, hashes: dict[str, str]) -> None:
    payload = {
        "schema_version": 1,
        "compatibility_id": "fixture-sae-compatibility",
        "scientific_use": False,
        "starts_experiment_timer": False,
        "functional_execution": False,
        "input_sha256": hashes,
        "expected": {
            "base": {
                "repository": "owner/base",
                "revision": "c" * 40,
                "model_type": "olmo3",
                "architecture": "Olmo3ForCausalLM",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
            },
            "sae": {
                "repository": "owner/saes",
                "revision": "a" * 40,
                "path": "layer-16/sae_weights.safetensors",
                "weights_sha256": "b" * 64,
                "model_name": "owner/base",
                "hook_name": "model.layers.16",
                "layer_index": 16,
                "d_in": 4096,
                "d_sae": 65536,
                "dtype": "float32",
                "saved_architecture": "jumprelu",
                "training_architecture": "matryoshka_batchtopk",
                "k": 100,
                "matryoshka_widths": [2048, 16384, 65536],
                "sae_lens_version": "6.42.0",
            },
            "saelens": {
                "source_revision": "d" * 40,
                "release_statement": "Matryoshka BatchTopK SAEs",
                "conversion_statement": _CONVERSION,
            },
            "weights_header": {
                "W_dec": {"dtype": "F32", "shape": [65536, 4096]},
                "W_enc": {"dtype": "F32", "shape": [4096, 65536]},
                "b_dec": {"dtype": "F32", "shape": [4096]},
                "b_enc": {"dtype": "F32", "shape": [65536]},
                "threshold": {"dtype": "F32", "shape": [65536]},
            },
        },
        "unresolved": {
            "base_revision_provenance": {
                "status": "unavailable",
                "reason": "The SAE metadata names the repository but no revision.",
            },
            "functional_hook_and_encode_smoke": {
                "status": "unavailable",
                "reason": "This metadata check does not execute the model or SAE.",
            },
        },
        "divergences": [
            "Structural metadata compatibility does not prove the SAE "
            "training revision."
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _verify_fixture(tmp_path: Path) -> SAECompatibilityReceipt:
    hashes, paths = _fixture_inputs(tmp_path)
    config_path = tmp_path / "compatibility.json"
    _write_config(config_path, hashes)
    return verify_sae_compatibility(
        load_sae_compatibility(config_path),
        input_paths=paths,
    )


def test_verifies_architecture_resolution_and_structural_compatibility(
    tmp_path: Path,
) -> None:
    receipt = _verify_fixture(tmp_path)

    assert receipt.architecture_consistent is True
    assert receipt.structurally_compatible is True
    assert receipt.functional_execution is False
    assert receipt.scientific_use is False
    assert receipt.starts_experiment_timer is False
    assert len(receipt.receipt_sha256) == 64


def test_rejects_input_hash_mismatch(tmp_path: Path) -> None:
    hashes, paths = _fixture_inputs(tmp_path)
    config_path = tmp_path / "compatibility.json"
    _write_config(config_path, hashes)
    paths["sae_cfg"].write_text("{}", encoding="utf-8")

    with pytest.raises(SAECompatibilityError, match="SHA-256"):
        verify_sae_compatibility(load_sae_compatibility(config_path), input_paths=paths)


def test_rejects_incompatible_hidden_width(tmp_path: Path) -> None:
    hashes, paths = _fixture_inputs(tmp_path)
    base = json.loads(paths["base_config"].read_text(encoding="utf-8"))
    base["hidden_size"] = 2048
    hashes["base_config"] = _write_json(paths["base_config"], base)
    config_path = tmp_path / "compatibility.json"
    _write_config(config_path, hashes)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["expected"]["base"]["hidden_size"] = 2048
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SAECompatibilityError, match="structurally compatible"):
        verify_sae_compatibility(load_sae_compatibility(config_path), input_paths=paths)


def test_rejects_missing_training_to_inference_conversion(tmp_path: Path) -> None:
    hashes, paths = _fixture_inputs(tmp_path)
    paths["saelens_docs"].write_text("No conversion statement.", encoding="utf-8")
    hashes["saelens_docs"] = hashlib.sha256(
        paths["saelens_docs"].read_bytes()
    ).hexdigest()
    config_path = tmp_path / "compatibility.json"
    _write_config(config_path, hashes)

    with pytest.raises(SAECompatibilityError, match="conversion statement"):
        verify_sae_compatibility(load_sae_compatibility(config_path), input_paths=paths)


def test_rejects_header_without_jumprelu_threshold(tmp_path: Path) -> None:
    hashes, paths = _fixture_inputs(tmp_path)
    tensors = {
        "W_dec": {"dtype": "F32", "shape": [65536, 4096]},
        "W_enc": {"dtype": "F32", "shape": [4096, 65536]},
        "b_dec": {"dtype": "F32", "shape": [4096]},
        "b_enc": {"dtype": "F32", "shape": [65536]},
    }
    hashes["weights_header"] = _write_header(paths["weights_header"], tensors)
    config_path = tmp_path / "compatibility.json"
    _write_config(config_path, hashes)

    with pytest.raises(SAECompatibilityError, match="weights header"):
        verify_sae_compatibility(load_sae_compatibility(config_path), input_paths=paths)


def test_rejects_self_consistent_but_incompatible_tensor_shapes(tmp_path: Path) -> None:
    hashes, paths = _fixture_inputs(tmp_path)
    tensors = {
        "W_dec": {"dtype": "F32", "shape": [65536, 2048]},
        "W_enc": {"dtype": "F32", "shape": [2048, 65536]},
        "b_dec": {"dtype": "F32", "shape": [2048]},
        "b_enc": {"dtype": "F32", "shape": [65536]},
        "threshold": {"dtype": "F32", "shape": [65536]},
    }
    hashes["weights_header"] = _write_header(paths["weights_header"], tensors)
    config_path = tmp_path / "compatibility.json"
    _write_config(config_path, hashes)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["expected"]["weights_header"] = tensors
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SAECompatibilityError, match="tensor shapes"):
        verify_sae_compatibility(load_sae_compatibility(config_path), input_paths=paths)


def test_rejects_runner_and_saved_sae_shape_disagreement(tmp_path: Path) -> None:
    hashes, paths = _fixture_inputs(tmp_path)
    runner = json.loads(paths["runner_cfg"].read_text(encoding="utf-8"))
    runner["saes"]["layer_16"]["d_in"] = 2048
    hashes["runner_cfg"] = _write_json(paths["runner_cfg"], runner)
    config_path = tmp_path / "compatibility.json"
    _write_config(config_path, hashes)

    with pytest.raises(SAECompatibilityError, match="runner and saved SAE"):
        verify_sae_compatibility(load_sae_compatibility(config_path), input_paths=paths)


@pytest.mark.parametrize(
    "field", ["scientific_use", "starts_experiment_timer", "functional_execution"]
)
def test_rejects_enabled_sae_boundary(tmp_path: Path, field: str) -> None:
    hashes, _ = _fixture_inputs(tmp_path)
    config_path = tmp_path / "compatibility.json"
    _write_config(config_path, hashes)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload[field] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SAECompatibilityError, match="must remain false"):
        load_sae_compatibility(config_path)


def test_cli_writes_exclusive_sae_compatibility_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hashes, paths = _fixture_inputs(tmp_path)
    config_path = tmp_path / "compatibility.json"
    output_path = tmp_path / "receipt.json"
    _write_config(config_path, hashes)
    arguments = [
        "verify-sae-compatibility",
        str(config_path),
        "--base-config",
        str(paths["base_config"]),
        "--sae-cfg",
        str(paths["sae_cfg"]),
        "--runner-cfg",
        str(paths["runner_cfg"]),
        "--release-readme",
        str(paths["release_readme"]),
        "--weights-header",
        str(paths["weights_header"]),
        "--saelens-docs",
        str(paths["saelens_docs"]),
        "--artifact-receipt",
        str(paths["artifact_receipt"]),
        "--output",
        str(output_path),
    ]

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["architecture_consistent"] is True
    assert main(arguments) == 2
    assert "exists" in capsys.readouterr().err
