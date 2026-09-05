"""Offline structural-compatibility evidence for the pinned public SAE."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from concept_filtering.serialization import bytes_sha256, canonical_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_INPUT_ROLES = {
    "base_config",
    "sae_cfg",
    "runner_cfg",
    "release_readme",
    "weights_header",
    "saelens_docs",
    "artifact_receipt",
}
_UNRESOLVED = {
    "base_revision_provenance",
    "functional_hook_and_encode_smoke",
}
_SAE_EXPECTED_FIELDS = {
    "repository",
    "revision",
    "path",
    "weights_sha256",
    "model_name",
    "hook_name",
    "layer_index",
    "d_in",
    "d_sae",
    "dtype",
    "saved_architecture",
    "training_architecture",
    "k",
    "matryoshka_widths",
    "sae_lens_version",
}
TensorHeader = dict[str, dict[str, object]]


class SAECompatibilityError(ValueError):
    """Raised when SAE compatibility evidence violates its frozen contract."""


@dataclass(frozen=True)
class SAECompatibilityConfig:
    compatibility_id: str
    scientific_use: bool
    starts_experiment_timer: bool
    functional_execution: bool
    input_sha256: dict[str, str]
    expected_base: dict[str, object]
    expected_sae: dict[str, object]
    expected_saelens: dict[str, object]
    expected_weights_header: TensorHeader
    unresolved: dict[str, dict[str, str]]
    divergences: tuple[str, ...]
    config_sha256: str


@dataclass(frozen=True)
class SAECompatibilityReceipt:
    compatibility_id: str
    scientific_use: bool
    starts_experiment_timer: bool
    functional_execution: bool
    architecture_consistent: bool
    structurally_compatible: bool
    input_sha256: dict[str, str]
    base_observation: dict[str, object]
    sae_observation: dict[str, object]
    saelens_observation: dict[str, object]
    weights_header_observation: TensorHeader
    unresolved: dict[str, dict[str, str]]
    divergences: tuple[str, ...]
    config_sha256: str
    receipt_sha256: str

    def body_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "compatibility_id": self.compatibility_id,
            "scientific_use": self.scientific_use,
            "starts_experiment_timer": self.starts_experiment_timer,
            "functional_execution": self.functional_execution,
            "architecture_consistent": self.architecture_consistent,
            "structurally_compatible": self.structurally_compatible,
            "input_sha256": self.input_sha256,
            "base_observation": self.base_observation,
            "sae_observation": self.sae_observation,
            "saelens_observation": self.saelens_observation,
            "weights_header_observation": self.weights_header_observation,
            "unresolved": self.unresolved,
            "divergences": list(self.divergences),
            "config_sha256": self.config_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body_dict(), "receipt_sha256": self.receipt_sha256}


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SAECompatibilityError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SAECompatibilityError(
            f"{field} fields differ; missing={missing}, unexpected={unexpected}"
        )


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SAECompatibilityError(f"{field} must be a non-empty string")
    return value


def _integer(payload: dict[str, Any], field: str, *, minimum: int = 0) -> int:
    value = payload.get(field)
    if type(value) is not int or value < minimum:
        raise SAECompatibilityError(f"{field} must be an integer >= {minimum}")
    return value


def _revision(payload: dict[str, Any], field: str) -> str:
    value = _string(payload, field)
    if _REVISION.fullmatch(value) is None:
        raise SAECompatibilityError(f"{field} must be a 40-character revision")
    return value


def _sha256(payload: dict[str, Any], field: str) -> str:
    value = _string(payload, field)
    if _SHA256.fullmatch(value) is None:
        raise SAECompatibilityError(f"{field} must be a SHA-256 value")
    return value


def _integer_list(payload: dict[str, Any], field: str) -> list[int]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise SAECompatibilityError(f"{field} must be a non-empty integer list")
    items = cast(list[object], value)
    if any(type(item) is not int or item <= 0 for item in items):
        raise SAECompatibilityError(f"{field} contains invalid widths")
    return cast(list[int], items)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAECompatibilityError(f"cannot load {path}: {error}") from error
    return _mapping(value, "compatibility config")


def _validate_envelope(payload: dict[str, Any]) -> None:
    fields = {
        "schema_version",
        "compatibility_id",
        "scientific_use",
        "starts_experiment_timer",
        "functional_execution",
        "input_sha256",
        "expected",
        "unresolved",
        "divergences",
    }
    _exact_fields(payload, fields, "compatibility config")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise SAECompatibilityError("schema_version must be 1")
    for field in ("scientific_use", "starts_experiment_timer", "functional_execution"):
        if payload[field] is not False:
            raise SAECompatibilityError(f"{field} must remain false")


def _input_hashes(value: object) -> dict[str, str]:
    payload = _mapping(value, "input_sha256")
    _exact_fields(payload, _INPUT_ROLES, "input_sha256")
    return {role: _sha256(payload, role) for role in sorted(_INPUT_ROLES)}


def _base_expectation(value: object) -> dict[str, object]:
    payload = _mapping(value, "expected.base")
    fields = {
        "repository",
        "revision",
        "model_type",
        "architecture",
        "hidden_size",
        "num_hidden_layers",
    }
    _exact_fields(payload, fields, "expected.base")
    return {
        "repository": _string(payload, "repository"),
        "revision": _revision(payload, "revision"),
        "model_type": _string(payload, "model_type"),
        "architecture": _string(payload, "architecture"),
        "hidden_size": _integer(payload, "hidden_size", minimum=1),
        "num_hidden_layers": _integer(payload, "num_hidden_layers", minimum=1),
    }


def _sae_identity_expectation(payload: dict[str, Any]) -> dict[str, object]:
    return {
        "repository": _string(payload, "repository"),
        "revision": _revision(payload, "revision"),
        "path": _string(payload, "path"),
        "weights_sha256": _sha256(payload, "weights_sha256"),
        "model_name": _string(payload, "model_name"),
        "hook_name": _string(payload, "hook_name"),
        "layer_index": _integer(payload, "layer_index"),
    }


def _sae_shape_expectation(payload: dict[str, Any]) -> dict[str, object]:
    return {
        "d_in": _integer(payload, "d_in", minimum=1),
        "d_sae": _integer(payload, "d_sae", minimum=1),
        "dtype": _string(payload, "dtype"),
        "saved_architecture": _string(payload, "saved_architecture"),
        "training_architecture": _string(payload, "training_architecture"),
        "k": _integer(payload, "k", minimum=1),
        "matryoshka_widths": _integer_list(payload, "matryoshka_widths"),
        "sae_lens_version": _string(payload, "sae_lens_version"),
    }


def _sae_expectation(value: object) -> dict[str, object]:
    payload = _mapping(value, "expected.sae")
    _exact_fields(payload, _SAE_EXPECTED_FIELDS, "expected.sae")
    return {**_sae_identity_expectation(payload), **_sae_shape_expectation(payload)}


def _saelens_expectation(value: object) -> dict[str, object]:
    payload = _mapping(value, "expected.saelens")
    fields = {"source_revision", "release_statement", "conversion_statement"}
    _exact_fields(payload, fields, "expected.saelens")
    return {
        "source_revision": _revision(payload, "source_revision"),
        "release_statement": _string(payload, "release_statement"),
        "conversion_statement": _string(payload, "conversion_statement"),
    }


def _tensor_header(value: object, field: str) -> TensorHeader:
    payload = _mapping(value, field)
    headers: TensorHeader = {}
    for name, raw_header in payload.items():
        header = _mapping(raw_header, f"{field}.{name}")
        dtype = _string(header, "dtype")
        shape = _integer_list(header, "shape")
        headers[name] = {"dtype": dtype, "shape": shape}
    return headers


def _unresolved(value: object) -> dict[str, dict[str, str]]:
    payload = _mapping(value, "unresolved")
    _exact_fields(payload, _UNRESOLVED, "unresolved")
    result: dict[str, dict[str, str]] = {}
    for name in sorted(_UNRESOLVED):
        item = _mapping(payload[name], f"unresolved.{name}")
        _exact_fields(item, {"status", "reason"}, f"unresolved.{name}")
        if _string(item, "status") != "unavailable":
            raise SAECompatibilityError(f"unresolved.{name} must be unavailable")
        result[name] = {"status": "unavailable", "reason": _string(item, "reason")}
    return result


def _divergences(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SAECompatibilityError("divergences must be a non-empty list")
    items = cast(list[object], value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise SAECompatibilityError("divergences contains invalid text")
    return tuple(cast(list[str], items))


def load_sae_compatibility(path: Path) -> SAECompatibilityConfig:
    """Load a fail-closed offline SAE compatibility contract."""
    payload = _load_payload(path)
    _validate_envelope(payload)
    expected = _mapping(payload["expected"], "expected")
    _exact_fields(expected, {"base", "sae", "saelens", "weights_header"}, "expected")
    return SAECompatibilityConfig(
        compatibility_id=_string(payload, "compatibility_id"),
        scientific_use=False,
        starts_experiment_timer=False,
        functional_execution=False,
        input_sha256=_input_hashes(payload["input_sha256"]),
        expected_base=_base_expectation(expected["base"]),
        expected_sae=_sae_expectation(expected["sae"]),
        expected_saelens=_saelens_expectation(expected["saelens"]),
        expected_weights_header=_tensor_header(
            expected["weights_header"], "expected.weights_header"
        ),
        unresolved=_unresolved(payload["unresolved"]),
        divergences=_divergences(payload["divergences"]),
        config_sha256=canonical_sha256(payload),
    )


def _bound_inputs(
    config: SAECompatibilityConfig, input_paths: dict[str, Path]
) -> dict[str, bytes]:
    if set(input_paths) != _INPUT_ROLES:
        raise SAECompatibilityError("input path roles do not match the contract")
    result: dict[str, bytes] = {}
    for role in sorted(_INPUT_ROLES):
        try:
            content = input_paths[role].read_bytes()
        except OSError as error:
            raise SAECompatibilityError(f"cannot read {role}: {error}") from error
        if bytes_sha256(content) != config.input_sha256[role]:
            raise SAECompatibilityError(f"{role} SHA-256 does not match")
        result[role] = content
    return result


def _json_bytes(content: bytes, field: str) -> dict[str, Any]:
    try:
        value: object = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SAECompatibilityError(f"invalid {field}: {error}") from error
    return _mapping(value, field)


def _observe_base(
    payload: dict[str, Any], expected: dict[str, object]
) -> dict[str, object]:
    raw_architectures = payload.get("architectures")
    if not isinstance(raw_architectures, list):
        raise SAECompatibilityError("base architectures must contain one value")
    architectures = cast(list[object], raw_architectures)
    if len(architectures) != 1:
        raise SAECompatibilityError("base architectures must contain one value")
    return {
        "repository": expected["repository"],
        "revision": expected["revision"],
        "model_type": payload.get("model_type"),
        "architecture": architectures[0],
        "hidden_size": payload.get("hidden_size"),
        "num_hidden_layers": payload.get("num_hidden_layers"),
    }


def _artifact_fields(payload: dict[str, Any]) -> dict[str, object]:
    return {
        "repository": payload.get("repository"),
        "revision": payload.get("revision"),
        "path": payload.get("path"),
        "weights_sha256": payload.get("sha256"),
    }


def _assert_runner_matches_saved(
    cfg: dict[str, Any],
    layer: dict[str, Any],
    runner: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    fields = ("d_in", "d_sae", "dtype")
    if any(layer.get(field) != cfg.get(field) for field in fields):
        raise SAECompatibilityError("runner and saved SAE shapes or dtype differ")
    if runner.get("sae_lens_version") != metadata.get("sae_lens_version"):
        raise SAECompatibilityError("runner and saved SAE versions differ")


def _observe_sae(
    cfg: dict[str, Any], runner: dict[str, Any], artifact: dict[str, Any]
) -> dict[str, object]:
    metadata = _mapping(cfg.get("metadata"), "sae cfg metadata")
    saes = _mapping(runner.get("saes"), "runner saes")
    layer = _mapping(saes.get("layer_16"), "runner layer_16")
    hooks = _mapping(runner.get("hook_names"), "runner hook_names")
    _assert_runner_matches_saved(cfg, layer, runner, metadata)
    observation = _artifact_fields(artifact)
    observation.update(
        {
            "model_name": metadata.get("model_name"),
            "hook_name": metadata.get("hook_name"),
            "layer_index": 16,
            "d_in": cfg.get("d_in"),
            "d_sae": cfg.get("d_sae"),
            "dtype": cfg.get("dtype"),
            "saved_architecture": cfg.get("architecture"),
            "training_architecture": layer.get("architecture"),
            "k": layer.get("k"),
            "matryoshka_widths": layer.get("matryoshka_widths"),
            "sae_lens_version": metadata.get("sae_lens_version"),
        }
    )
    if hooks.get("layer_16") != observation["hook_name"]:
        raise SAECompatibilityError("runner and saved SAE hook names differ")
    if runner.get("model_name") != observation["model_name"]:
        raise SAECompatibilityError("runner and saved SAE model names differ")
    return observation


def _parse_header(content: bytes) -> TensorHeader:
    if len(content) < 9:
        raise SAECompatibilityError("weights header is truncated")
    header_size = int.from_bytes(content[:8], "little")
    if header_size != len(content) - 8:
        raise SAECompatibilityError("weights header length does not match")
    return _tensor_header(_json_bytes(content[8:], "weights header"), "weights header")


def _observe_saelens(
    expected: dict[str, object], release: str, docs: str
) -> dict[str, object]:
    release_statement = cast(str, expected["release_statement"])
    conversion_statement = cast(str, expected["conversion_statement"])
    if release_statement not in release:
        raise SAECompatibilityError("release architecture statement is missing")
    if conversion_statement not in docs:
        raise SAECompatibilityError("SAELens conversion statement is missing")
    return expected


def _assert_expected(field: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise SAECompatibilityError(f"{field} does not match the frozen expectation")


def _derived_tensor_header(sae: dict[str, object]) -> TensorHeader:
    if sae["dtype"] != "float32":
        raise SAECompatibilityError("saved SAE dtype is not supported")
    d_in = cast(int, sae["d_in"])
    d_sae = cast(int, sae["d_sae"])
    return {
        "W_dec": {"dtype": "F32", "shape": [d_sae, d_in]},
        "W_enc": {"dtype": "F32", "shape": [d_in, d_sae]},
        "b_dec": {"dtype": "F32", "shape": [d_in]},
        "b_enc": {"dtype": "F32", "shape": [d_sae]},
        "threshold": {"dtype": "F32", "shape": [d_sae]},
    }


def _assert_structural_compatibility(
    base: dict[str, object], sae: dict[str, object], header: TensorHeader
) -> None:
    layer = cast(int, sae["layer_index"])
    widths = cast(list[int], sae["matryoshka_widths"])
    compatible = (
        base["hidden_size"] == sae["d_in"]
        and layer < cast(int, base["num_hidden_layers"])
        and widths[-1] == sae["d_sae"]
        and "threshold" in header
    )
    if not compatible:
        raise SAECompatibilityError("SAE is not structurally compatible with the base")
    if header != _derived_tensor_header(sae):
        raise SAECompatibilityError("SAE tensor shapes or dtypes are incompatible")


def _build_receipt(
    config: SAECompatibilityConfig,
    base: dict[str, object],
    sae: dict[str, object],
    saelens: dict[str, object],
    header: TensorHeader,
) -> SAECompatibilityReceipt:
    receipt = SAECompatibilityReceipt(
        compatibility_id=config.compatibility_id,
        scientific_use=False,
        starts_experiment_timer=False,
        functional_execution=False,
        architecture_consistent=True,
        structurally_compatible=True,
        input_sha256=config.input_sha256,
        base_observation=base,
        sae_observation=sae,
        saelens_observation=saelens,
        weights_header_observation=header,
        unresolved=config.unresolved,
        divergences=config.divergences,
        config_sha256=config.config_sha256,
        receipt_sha256="",
    )
    return replace(receipt, receipt_sha256=canonical_sha256(receipt.body_dict()))


def verify_sae_compatibility(
    config: SAECompatibilityConfig, *, input_paths: dict[str, Path]
) -> SAECompatibilityReceipt:
    """Verify architecture resolution and static shape compatibility offline."""
    inputs = _bound_inputs(config, input_paths)
    base = _observe_base(
        _json_bytes(inputs["base_config"], "base config"), config.expected_base
    )
    sae = _observe_sae(
        _json_bytes(inputs["sae_cfg"], "sae cfg"),
        _json_bytes(inputs["runner_cfg"], "runner cfg"),
        _json_bytes(inputs["artifact_receipt"], "artifact receipt"),
    )
    header = _parse_header(inputs["weights_header"])
    saelens = _observe_saelens(
        config.expected_saelens,
        inputs["release_readme"].decode("utf-8"),
        inputs["saelens_docs"].decode("utf-8"),
    )
    _assert_expected("base observation", base, config.expected_base)
    _assert_expected("SAE observation", sae, config.expected_sae)
    _assert_expected("weights header", header, config.expected_weights_header)
    _assert_structural_compatibility(base, sae, header)
    return _build_receipt(config, base, sae, saelens, header)


def write_sae_compatibility_receipt(
    receipt: SAECompatibilityReceipt, path: Path
) -> None:
    """Write one static compatibility receipt without replacing evidence."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            receipt.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
