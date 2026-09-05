"""Receipt-bound configuration and results for the interim rank-32 smoke."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.organism import (
    ObservationStatus,
    OrganismConfig,
    OrganismValidationError,
    load_organism_config,
)
from concept_filtering.receipts import ArtifactKind, ArtifactReceipt
from concept_filtering.serialization import bytes_sha256, canonical_sha256

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_FILE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SmokeValidationError(ValueError):
    """Raised when an interim smoke cannot preserve its declared boundary."""


@dataclass(frozen=True)
class SmokePrompt:
    system: str
    user: str


@dataclass(frozen=True)
class SmokePlan:
    smoke_id: str
    plan_sha256: str
    scientific_use: bool
    starts_experiment_timer: bool
    decision_status: str
    organism: OrganismConfig
    base: ArtifactReceipt
    adapter: ArtifactReceipt
    template_path: str
    template_sha256: str
    rank: int
    device: str
    dtype: str
    prompt: SmokePrompt
    max_new_tokens: int
    do_sample: bool


@dataclass(frozen=True)
class RuntimeObservation:
    completion: str
    adapter_sha256: str
    template_sha256: str
    runtime_versions: dict[str, str]


@dataclass(frozen=True)
class SmokeReceipt:
    payload: dict[str, object]
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "receipt_sha256": self.receipt_sha256}


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SmokeValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SmokeValidationError(f"{field} must be a non-empty string")
    return value


def _require_exact_fields(
    payload: dict[str, Any], allowed: set[str], field: str
) -> None:
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise SmokeValidationError(
            f"{field} has unexpected fields: {', '.join(unexpected)}"
        )
    missing = sorted(allowed - set(payload))
    if missing:
        raise SmokeValidationError(f"{field} is missing fields: {', '.join(missing)}")


def _load_payload(path: Path) -> tuple[dict[str, Any], str]:
    try:
        source = path.read_bytes()
        value: object = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SmokeValidationError(f"cannot load smoke plan {path}: {error}") from error
    return _mapping(value, "smoke plan"), bytes_sha256(source)


def _parse_claim_boundary(payload: dict[str, Any]) -> tuple[bool, bool]:
    scientific_use = payload.get("scientific_use")
    starts_timer = payload.get("starts_experiment_timer")
    if scientific_use is not False:
        raise SmokeValidationError("scientific_use must be false for an interim smoke")
    if starts_timer is not False:
        raise SmokeValidationError(
            "starts_experiment_timer must be false for an interim smoke"
        )
    return scientific_use, starts_timer


def _resolve_organism(path: Path, raw_path: object) -> OrganismConfig:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SmokeValidationError("organism_config must be a relative path")
    relative = Path(raw_path)
    root = path.parent.resolve()
    organism_path = (path.parent / relative).resolve()
    if relative.is_absolute() or not organism_path.is_relative_to(root):
        raise SmokeValidationError(
            "organism_config must stay inside the config directory"
        )
    try:
        return load_organism_config(organism_path)
    except OrganismValidationError as error:
        raise SmokeValidationError(f"invalid organism_config: {error}") from error


def _artifact(
    organism: OrganismConfig, artifact_id: str, kind: ArtifactKind
) -> ArtifactReceipt:
    matches = [item for item in organism.artifacts if item.artifact_id == artifact_id]
    if len(matches) != 1:
        raise SmokeValidationError(f"organism must contain one {artifact_id} receipt")
    artifact = matches[0]
    if artifact.kind is not kind:
        raise SmokeValidationError(f"{artifact_id} must be a {kind.value} receipt")
    if artifact.revision is None:
        raise SmokeValidationError(f"{artifact_id} must pin a revision")
    return artifact


def _rank32(organism: OrganismConfig) -> int:
    observation = organism.parameters.get("training.lora_rank")
    if (
        observation is None
        or observation.status is not ObservationStatus.OBSERVED
        or type(observation.value) is not int
        or observation.value != 32
    ):
        raise SmokeValidationError("interim smoke organism must record LoRA rank 32")
    return observation.value


def _parse_decision(payload: dict[str, Any]) -> tuple[str, str]:
    decision = _mapping(payload.get("decision"), "decision")
    _require_exact_fields(decision, {"status", "supersession_trigger"}, "decision")
    status = _required_string(decision, "status")
    if status != "approved_interim":
        raise SmokeValidationError("decision.status must be approved_interim")
    return status, _required_string(decision, "supersession_trigger")


def _parse_prompt(payload: dict[str, Any]) -> SmokePrompt:
    prompt = _mapping(payload.get("prompt"), "prompt")
    _require_exact_fields(prompt, {"system", "user"}, "prompt")
    return SmokePrompt(
        system=_required_string(prompt, "system"),
        user=_required_string(prompt, "user"),
    )


def _parse_generation(payload: dict[str, Any]) -> tuple[int, bool]:
    max_new_tokens = payload.get("max_new_tokens")
    do_sample = payload.get("do_sample")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise SmokeValidationError("max_new_tokens must be an integer from 1 to 128")
    if do_sample is not False:
        raise SmokeValidationError("do_sample must be false for a deterministic smoke")
    return max_new_tokens, do_sample


def _parse_runtime(payload: dict[str, Any]) -> tuple[str, str]:
    device = _required_string(payload, "device")
    dtype = _required_string(payload, "dtype")
    if device not in {"cpu", "cuda", "mps"}:
        raise SmokeValidationError("device must be cpu, cuda, or mps")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise SmokeValidationError("dtype must be bfloat16, float16, or float32")
    return device, dtype


def _parse_template(
    payload: dict[str, Any], adapter: ArtifactReceipt
) -> tuple[str, str]:
    template_path = _required_string(payload, "template_path")
    path = Path(template_path)
    adapter_parent = Path(adapter.path or "").parent
    if path.is_absolute() or path.parent != adapter_parent:
        raise SmokeValidationError("template_path must stay beside the adapter weights")
    template_sha256 = _required_string(payload, "template_sha256")
    if _FILE_SHA256.fullmatch(template_sha256) is None:
        raise SmokeValidationError("template_sha256 must be 64 lowercase hex digits")
    return template_path, template_sha256


_PLAN_FIELDS = {
    "schema_version",
    "smoke_id",
    "scientific_use",
    "starts_experiment_timer",
    "decision",
    "organism_config",
    "base_artifact_id",
    "adapter_artifact_id",
    "template_path",
    "template_sha256",
    "device",
    "dtype",
    "prompt",
    "max_new_tokens",
    "do_sample",
}


def _bind_artifacts(
    payload: dict[str, Any], organism: OrganismConfig
) -> tuple[ArtifactReceipt, ArtifactReceipt]:
    base = _artifact(
        organism, _required_string(payload, "base_artifact_id"), ArtifactKind.MODEL
    )
    adapter = _artifact(
        organism,
        _required_string(payload, "adapter_artifact_id"),
        ArtifactKind.ADAPTER,
    )
    if adapter.path is None or adapter.sha256 is None:
        raise SmokeValidationError("adapter receipt must pin its path and SHA-256")
    return base, adapter


def _build_smoke_plan(
    payload: dict[str, Any],
    plan_sha256: str,
    organism: OrganismConfig,
    base: ArtifactReceipt,
    adapter: ArtifactReceipt,
) -> SmokePlan:
    scientific_use, starts_timer = _parse_claim_boundary(payload)
    status, _ = _parse_decision(payload)
    device, dtype = _parse_runtime(payload)
    max_new_tokens, do_sample = _parse_generation(payload)
    template_path, template_sha256 = _parse_template(payload, adapter)
    return SmokePlan(
        smoke_id=_required_string(payload, "smoke_id"),
        plan_sha256=plan_sha256,
        scientific_use=scientific_use,
        starts_experiment_timer=starts_timer,
        decision_status=status,
        organism=organism,
        base=base,
        adapter=adapter,
        template_path=template_path,
        template_sha256=template_sha256,
        rank=_rank32(organism),
        device=device,
        dtype=dtype,
        prompt=_parse_prompt(payload),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )


def load_smoke_plan(path: Path) -> SmokePlan:
    """Load a smoke plan and bind it to one exact rank-32 organism."""
    payload, plan_sha256 = _load_payload(path)
    _require_exact_fields(payload, _PLAN_FIELDS, "smoke plan")
    if payload.get("schema_version") != 1:
        raise SmokeValidationError("schema_version must be the integer 1")
    _parse_claim_boundary(payload)
    organism = _resolve_organism(path, payload.get("organism_config"))
    base, adapter = _bind_artifacts(payload, organism)
    return _build_smoke_plan(payload, plan_sha256, organism, base, adapter)


def _validate_runtime_observation(
    plan: SmokePlan, observation: RuntimeObservation
) -> None:
    if not observation.completion.strip():
        raise SmokeValidationError("smoke completion must be non-empty")
    if _FILE_SHA256.fullmatch(observation.adapter_sha256) is None:
        raise SmokeValidationError("runtime adapter SHA-256 is malformed")
    if observation.adapter_sha256 != plan.adapter.sha256:
        raise SmokeValidationError("runtime adapter SHA-256 does not match the plan")
    if observation.template_sha256 != plan.template_sha256:
        raise SmokeValidationError("runtime template SHA-256 does not match the plan")
    if not observation.runtime_versions or any(
        not key.strip() or not value.strip()
        for key, value in observation.runtime_versions.items()
    ):
        raise SmokeValidationError("runtime_versions must contain named versions")


def _artifact_payload(
    plan: SmokePlan, observation: RuntimeObservation
) -> dict[str, object]:
    return {
        "base": {
            "repository": plan.base.repository,
            "revision": plan.base.revision,
        },
        "adapter": {
            "repository": plan.adapter.repository,
            "revision": plan.adapter.revision,
            "path": plan.adapter.path,
            "sha256": observation.adapter_sha256,
        },
        "template": {
            "path": plan.template_path,
            "sha256": observation.template_sha256,
        },
    }


def _build_receipt_payload(
    plan: SmokePlan, code_revision: str, observation: RuntimeObservation
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "smoke_id": plan.smoke_id,
        "scientific_use": plan.scientific_use,
        "starts_experiment_timer": plan.starts_experiment_timer,
        "code_revision": code_revision,
        "plan_sha256": plan.plan_sha256,
        "organism_id": plan.organism.organism_id,
        "organism_sha256": plan.organism.organism_sha256,
        "rank": plan.rank,
        "device": plan.device,
        "dtype": plan.dtype,
        "prompt_sha256": canonical_sha256(
            {"system": plan.prompt.system, "user": plan.prompt.user}
        ),
        "completion": observation.completion,
        "completion_sha256": canonical_sha256(observation.completion),
        "runtime_versions": dict(sorted(observation.runtime_versions.items())),
    }
    payload.update(_artifact_payload(plan, observation))
    return payload


def _git_output(repository_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise SmokeValidationError(
            f"cannot verify executor checkout: {error}"
        ) from error
    return result.stdout.strip()


def source_repository_root() -> Path:
    """Resolve the source checkout containing this loaded executor module."""
    source = Path(__file__).resolve()
    root = source.parents[2]
    if root / "src/concept_filtering/smoke.py" != source:
        raise SmokeValidationError("executor is not loaded from a source checkout")
    return root


def _verify_executor_revision(expected: str) -> str:
    if _COMMIT_SHA.fullmatch(expected) is None:
        raise SmokeValidationError("code_revision must be a 40-character commit SHA")
    repository_root = source_repository_root()
    observed = _git_output(repository_root, "rev-parse", "HEAD")
    if observed != expected:
        raise SmokeValidationError(
            f"executor revision {observed} does not match expected {expected}"
        )
    status = _git_output(
        repository_root, "status", "--porcelain", "--untracked-files=all"
    )
    if status:
        raise SmokeValidationError("executor checkout must be clean")
    return observed


def verify_executor_revision(expected: str) -> str:
    """Bind an execution to the exact clean checkout that loaded this package."""
    return _verify_executor_revision(expected)


def _write_receipt(receipt: SmokeReceipt, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as target:
            target.write(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise SmokeValidationError(
            f"refusing to overwrite output: {output_path}"
        ) from error


def execute_smoke(
    plan: SmokePlan,
    *,
    code_revision: str,
    output_path: Path,
    run: Callable[[SmokePlan], RuntimeObservation],
) -> SmokeReceipt:
    """Run one smoke backend and write a receipt only after validation."""
    verified_revision = _verify_executor_revision(code_revision)
    if output_path.exists():
        raise SmokeValidationError(f"refusing to overwrite output: {output_path}")
    observation = run(plan)
    _validate_runtime_observation(plan, observation)
    payload = _build_receipt_payload(plan, verified_revision, observation)
    receipt = SmokeReceipt(payload=payload, receipt_sha256=canonical_sha256(payload))
    _write_receipt(receipt, output_path)
    return receipt
