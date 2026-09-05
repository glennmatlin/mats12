"""Resumable, receipt-bound generation over pinned prompt banks."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.prompt_bank import PromptBankExtraction, PromptRecord
from concept_filtering.serialization import bytes_sha256, canonical_sha256
from concept_filtering.smoke import SmokePlan, verify_executor_revision

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLAN_FIELDS = {
    "schema_version",
    "generation_id",
    "smoke_plan_sha256",
    "max_new_tokens",
    "do_sample",
    "scientific_use",
    "starts_experiment_timer",
    "protocol_basis",
}
_SAMPLE_FIELDS = {
    "schema_version",
    "index",
    "behavior",
    "question_id",
    "prompt_sha256",
    "bank_output_sha256",
    "completion",
    "completion_sha256",
    "input_tokens",
    "generated_tokens",
    "elapsed_seconds",
    "hit_token_limit",
    "sample_sha256",
}


class GenerationValidationError(ValueError):
    """Raised when a generation run cannot preserve its declared boundary."""


@dataclass(frozen=True)
class GenerationPlan:
    generation_id: str
    plan_sha256: str
    smoke_plan_sha256: str
    max_new_tokens: int
    do_sample: bool
    scientific_use: bool
    starts_experiment_timer: bool
    protocol_basis: str


@dataclass(frozen=True)
class GenerationObservation:
    question_id: str
    behavior: str
    completion: str
    input_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    hit_token_limit: bool


@dataclass(frozen=True)
class GenerationRuntime:
    adapter_sha256: str
    template_sha256: str
    runtime_versions: dict[str, str]
    observations: Iterable[GenerationObservation]


@dataclass(frozen=True)
class GenerationReceipt:
    schema_version: int
    generation_id: str
    run_sha256: str
    runtime_sha256: str
    sample_count: int
    generated_tokens: int
    elapsed_seconds: float
    hit_token_limit_count: int
    samples_sha256: str
    scientific_use: bool
    starts_experiment_timer: bool
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class _SelectedPrompt:
    index: int
    prompt: PromptRecord
    prompt_sha256: str
    bank_output_sha256: str


Bank = tuple[PromptBankExtraction, tuple[PromptRecord, ...]]
GenerationRunner = Callable[
    [SmokePlan, GenerationPlan, tuple[PromptRecord, ...]], GenerationRuntime
]


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GenerationValidationError(f"{field} must be a non-empty string")
    return value


def _load_plan_payload(path: Path) -> tuple[dict[str, Any], str]:
    try:
        source = path.read_bytes()
        value: object = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationValidationError(
            f"cannot load generation plan {path}: {error}"
        ) from error
    return _mapping(value, "generation plan"), bytes_sha256(source)


def _validate_plan_fields(payload: dict[str, Any]) -> None:
    if set(payload) != _PLAN_FIELDS:
        raise GenerationValidationError("generation plan fields do not match schema")
    if payload.get("schema_version") != 1:
        raise GenerationValidationError("schema_version must be the integer 1")
    if payload.get("scientific_use") is not False:
        raise GenerationValidationError("scientific_use must be false")
    if payload.get("starts_experiment_timer") is not False:
        raise GenerationValidationError("starts_experiment_timer must be false")
    if payload.get("do_sample") is not False:
        raise GenerationValidationError("do_sample must be false")


def load_generation_plan(path: Path) -> GenerationPlan:
    """Load the explicit non-scientific interim generation contract."""
    payload, plan_sha256 = _load_plan_payload(path)
    _validate_plan_fields(payload)
    max_new_tokens = payload.get("max_new_tokens")
    smoke_sha = _required_string(payload, "smoke_plan_sha256")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8192:
        raise GenerationValidationError("max_new_tokens must be from 1 to 8192")
    if _SHA256.fullmatch(smoke_sha) is None:
        raise GenerationValidationError("smoke_plan_sha256 must be a SHA-256")
    return GenerationPlan(
        generation_id=_required_string(payload, "generation_id"),
        plan_sha256=plan_sha256,
        smoke_plan_sha256=smoke_sha,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        scientific_use=False,
        starts_experiment_timer=False,
        protocol_basis=_required_string(payload, "protocol_basis"),
    )


def _validate_plan_binding(plan: GenerationPlan, smoke: SmokePlan) -> None:
    if plan.smoke_plan_sha256 != smoke.plan_sha256:
        raise GenerationValidationError("generation plan does not bind smoke plan")
    if smoke.scientific_use or smoke.starts_experiment_timer:
        raise GenerationValidationError("smoke plan expands the interim claim boundary")


def _select_prompts(
    banks: tuple[Bank, ...], limit_per_bank: int | None
) -> tuple[_SelectedPrompt, ...]:
    if not banks:
        raise GenerationValidationError("at least one prompt bank is required")
    if limit_per_bank is not None and limit_per_bank <= 0:
        raise GenerationValidationError("limit_per_bank must be positive")
    selected: list[_SelectedPrompt] = []
    for receipt, prompts in banks:
        if len(prompts) != receipt.question_count or not prompts:
            raise GenerationValidationError("prompt bank count does not match receipt")
        count = (
            len(prompts)
            if limit_per_bank is None
            else min(limit_per_bank, len(prompts))
        )
        for prompt in prompts[:count]:
            _validate_bank_prompt(receipt, prompt)
            selected.append(
                _SelectedPrompt(
                    len(selected),
                    prompt,
                    canonical_sha256(prompt.to_dict()),
                    receipt.output_sha256,
                )
            )
    keys = [(item.prompt.behavior, item.prompt.question_id) for item in selected]
    if len(set(keys)) != len(keys):
        raise GenerationValidationError("selected prompt identities are not unique")
    return tuple(selected)


def _validate_bank_prompt(receipt: PromptBankExtraction, prompt: PromptRecord) -> None:
    if prompt.behavior != receipt.behavior:
        raise GenerationValidationError("prompt behavior does not match bank receipt")
    if not prompt.question_id.strip():
        raise GenerationValidationError("prompt question_id cannot be empty")


def _bank_body(banks: tuple[Bank, ...]) -> list[dict[str, object]]:
    return [
        {
            "behavior": receipt.behavior,
            "question_count": receipt.question_count,
            "archive_sha256": receipt.archive_sha256,
            "output_sha256": receipt.output_sha256,
            "receipt_sha256": receipt.receipt_sha256,
        }
        for receipt, _ in banks
    ]


def _selected_body(selected: tuple[_SelectedPrompt, ...]) -> list[dict[str, object]]:
    return [
        {
            "index": item.index,
            "behavior": item.prompt.behavior,
            "question_id": item.prompt.question_id,
            "prompt_sha256": item.prompt_sha256,
            "bank_output_sha256": item.bank_output_sha256,
        }
        for item in selected
    ]


def _run_body(
    plan: GenerationPlan,
    smoke: SmokePlan,
    code_revision: str,
    banks: tuple[Bank, ...],
    selected: tuple[_SelectedPrompt, ...],
    limit_per_bank: int | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generation_id": plan.generation_id,
        "generation_plan_sha256": plan.plan_sha256,
        "smoke_plan_sha256": smoke.plan_sha256,
        "organism_sha256": smoke.organism.organism_sha256,
        "code_revision": code_revision,
        "limit_per_bank": limit_per_bank,
        "max_new_tokens": plan.max_new_tokens,
        "do_sample": plan.do_sample,
        "scientific_use": plan.scientific_use,
        "starts_experiment_timer": plan.starts_experiment_timer,
        "banks": _bank_body(banks),
        "selected": _selected_body(selected),
    }


def _bound_payload(body: dict[str, object], hash_field: str) -> dict[str, object]:
    return {**body, hash_field: canonical_sha256(body)}


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationValidationError(f"cannot load {field}: {error}") from error
    return _mapping(value, field)


def _write_or_validate(path: Path, payload: dict[str, object], field: str) -> None:
    if path.exists():
        if _load_json(path, field) != payload:
            raise GenerationValidationError(f"existing {field} does not match run")
        return
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise GenerationValidationError(f"racing {field} creation") from error


def _runtime_body(runtime: GenerationRuntime) -> dict[str, object]:
    if _SHA256.fullmatch(runtime.adapter_sha256) is None:
        raise GenerationValidationError("runtime adapter SHA-256 is invalid")
    if _SHA256.fullmatch(runtime.template_sha256) is None:
        raise GenerationValidationError("runtime template SHA-256 is invalid")
    if not runtime.runtime_versions or any(
        not key.strip() or not value.strip()
        for key, value in runtime.runtime_versions.items()
    ):
        raise GenerationValidationError("runtime versions are invalid")
    return {
        "adapter_sha256": runtime.adapter_sha256,
        "template_sha256": runtime.template_sha256,
        "runtime_versions": dict(sorted(runtime.runtime_versions.items())),
    }


def _validate_runtime(
    runtime: GenerationRuntime, smoke: SmokePlan
) -> dict[str, object]:
    body = _runtime_body(runtime)
    if runtime.adapter_sha256 != smoke.adapter.sha256:
        raise GenerationValidationError("runtime adapter does not match smoke plan")
    if runtime.template_sha256 != smoke.template_sha256:
        raise GenerationValidationError("runtime template does not match smoke plan")
    return _bound_payload(body, "runtime_sha256")


def _sample_body(
    selected: _SelectedPrompt, observation: GenerationObservation
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "index": selected.index,
        "behavior": observation.behavior,
        "question_id": observation.question_id,
        "prompt_sha256": selected.prompt_sha256,
        "bank_output_sha256": selected.bank_output_sha256,
        "completion": observation.completion,
        "completion_sha256": canonical_sha256(observation.completion),
        "input_tokens": observation.input_tokens,
        "generated_tokens": observation.generated_tokens,
        "elapsed_seconds": observation.elapsed_seconds,
        "hit_token_limit": observation.hit_token_limit,
    }


def _validate_observation(
    selected: _SelectedPrompt,
    observation: GenerationObservation,
    max_new_tokens: int,
) -> None:
    prompt = selected.prompt
    if (observation.behavior, observation.question_id) != (
        prompt.behavior,
        prompt.question_id,
    ):
        raise GenerationValidationError("runtime observation is out of order")
    if not observation.completion.strip():
        raise GenerationValidationError("runtime completion cannot be empty")
    if observation.input_tokens <= 0:
        raise GenerationValidationError("runtime input_tokens must be positive")
    if not 1 <= observation.generated_tokens <= max_new_tokens:
        raise GenerationValidationError("runtime generated_tokens is invalid")
    if (
        not math.isfinite(observation.elapsed_seconds)
        or observation.elapsed_seconds < 0
    ):
        raise GenerationValidationError("runtime elapsed_seconds is invalid")


def _sample_path(output_dir: Path, index: int) -> Path:
    return output_dir / "samples" / f"{index:04d}.json"


def _validate_bound_hash(payload: dict[str, Any], hash_field: str, field: str) -> None:
    stored = payload.get(hash_field)
    body = dict(payload)
    body.pop(hash_field, None)
    if not isinstance(stored, str) or canonical_sha256(body) != stored:
        raise GenerationValidationError(f"{field} hash mismatch")


def _load_existing_sample(
    path: Path, selected: _SelectedPrompt, max_new_tokens: int
) -> dict[str, Any]:
    payload = _load_json(path, "sample receipt")
    if set(payload) != _SAMPLE_FIELDS:
        raise GenerationValidationError("sample receipt fields do not match schema")
    _validate_bound_hash(payload, "sample_sha256", "sample receipt")
    expected = (selected.index, selected.prompt.behavior, selected.prompt.question_id)
    observed = (
        payload.get("index"),
        payload.get("behavior"),
        payload.get("question_id"),
    )
    if observed != expected or payload.get("prompt_sha256") != selected.prompt_sha256:
        raise GenerationValidationError("sample receipt does not match selected prompt")
    if payload.get("bank_output_sha256") != selected.bank_output_sha256:
        raise GenerationValidationError("sample receipt bank hash mismatch")
    _validate_stored_observation(payload, max_new_tokens)
    return payload


def _validate_stored_observation(payload: dict[str, Any], max_new_tokens: int) -> None:
    completion = payload.get("completion")
    input_tokens = payload.get("input_tokens")
    generated_tokens = payload.get("generated_tokens")
    elapsed = payload.get("elapsed_seconds")
    if not isinstance(completion, str) or not completion.strip():
        raise GenerationValidationError("stored completion is invalid")
    if payload.get("completion_sha256") != canonical_sha256(completion):
        raise GenerationValidationError("stored completion hash mismatch")
    if type(input_tokens) is not int or input_tokens <= 0:
        raise GenerationValidationError("stored input token count is invalid")
    if type(generated_tokens) is not int or not 1 <= generated_tokens <= max_new_tokens:
        raise GenerationValidationError("stored generation token count is invalid")
    if (
        not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise GenerationValidationError("stored elapsed time is invalid")
    if type(payload.get("hit_token_limit")) is not bool:
        raise GenerationValidationError("stored token-limit flag is invalid")


def _pending_prompts(
    output_dir: Path,
    selected: tuple[_SelectedPrompt, ...],
    max_new_tokens: int,
) -> tuple[_SelectedPrompt, ...]:
    pending: list[_SelectedPrompt] = []
    for item in selected:
        path = _sample_path(output_dir, item.index)
        if path.exists():
            _load_existing_sample(path, item, max_new_tokens)
        else:
            pending.append(item)
    return tuple(pending)


def _write_observations(
    output_dir: Path,
    pending: tuple[_SelectedPrompt, ...],
    observations: Iterable[GenerationObservation],
    max_new_tokens: int,
) -> None:
    iterator: Iterator[GenerationObservation] = iter(observations)
    for item in pending:
        try:
            observation = next(iterator)
        except StopIteration as error:
            raise GenerationValidationError(
                "runtime omitted pending prompts"
            ) from error
        _validate_observation(item, observation, max_new_tokens)
        body = _sample_body(item, observation)
        payload = _bound_payload(body, "sample_sha256")
        _write_or_validate(
            _sample_path(output_dir, item.index), payload, "sample receipt"
        )
    try:
        next(iterator)
    except StopIteration:
        return
    raise GenerationValidationError("runtime returned extra observations")


def _receipt_body(
    plan: GenerationPlan,
    run_payload: dict[str, object],
    runtime_payload: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generation_id": plan.generation_id,
        "run_sha256": cast(str, run_payload["run_sha256"]),
        "runtime_sha256": cast(str, runtime_payload["runtime_sha256"]),
        "sample_count": len(samples),
        "generated_tokens": sum(
            cast(int, item["generated_tokens"]) for item in samples
        ),
        "elapsed_seconds": sum(
            cast(float, item["elapsed_seconds"]) for item in samples
        ),
        "hit_token_limit_count": sum(
            item["hit_token_limit"] is True for item in samples
        ),
        "samples_sha256": canonical_sha256([item["sample_sha256"] for item in samples]),
        "scientific_use": plan.scientific_use,
        "starts_experiment_timer": plan.starts_experiment_timer,
    }


def _parse_receipt(payload: dict[str, object]) -> GenerationReceipt:
    return GenerationReceipt(
        schema_version=cast(int, payload["schema_version"]),
        generation_id=cast(str, payload["generation_id"]),
        run_sha256=cast(str, payload["run_sha256"]),
        runtime_sha256=cast(str, payload["runtime_sha256"]),
        sample_count=cast(int, payload["sample_count"]),
        generated_tokens=cast(int, payload["generated_tokens"]),
        elapsed_seconds=cast(float, payload["elapsed_seconds"]),
        hit_token_limit_count=cast(int, payload["hit_token_limit_count"]),
        samples_sha256=cast(str, payload["samples_sha256"]),
        scientific_use=cast(bool, payload["scientific_use"]),
        starts_experiment_timer=cast(bool, payload["starts_experiment_timer"]),
        receipt_sha256=cast(str, payload["receipt_sha256"]),
    )


def _finalize(
    plan: GenerationPlan,
    smoke: SmokePlan,
    output_dir: Path,
    run_payload: dict[str, object],
    selected: tuple[_SelectedPrompt, ...],
) -> GenerationReceipt:
    runtime = _load_json(output_dir / "runtime.json", "runtime receipt")
    _validate_bound_hash(runtime, "runtime_sha256", "runtime receipt")
    _validate_stored_runtime(runtime, smoke)
    samples = [
        _load_existing_sample(
            _sample_path(output_dir, item.index), item, plan.max_new_tokens
        )
        for item in selected
    ]
    body = _receipt_body(plan, run_payload, runtime, samples)
    payload = _bound_payload(body, "receipt_sha256")
    _write_or_validate(output_dir / "receipt.json", payload, "generation receipt")
    return _parse_receipt(payload)


def _validate_stored_runtime(payload: dict[str, Any], smoke: SmokePlan) -> None:
    if payload.get("adapter_sha256") != smoke.adapter.sha256:
        raise GenerationValidationError("stored runtime adapter hash mismatch")
    if payload.get("template_sha256") != smoke.template_sha256:
        raise GenerationValidationError("stored runtime template hash mismatch")
    versions = payload.get("runtime_versions")
    if not isinstance(versions, dict) or not versions:
        raise GenerationValidationError("stored runtime versions are invalid")


def execute_generation_bank(
    plan: GenerationPlan,
    smoke: SmokePlan,
    *,
    code_revision: str,
    banks: tuple[Bank, ...],
    limit_per_bank: int | None,
    output_dir: Path,
    run: GenerationRunner,
) -> GenerationReceipt:
    """Generate selected prompts, checkpoint each output, and finalize a receipt."""
    _validate_plan_binding(plan, smoke)
    revision = verify_executor_revision(code_revision)
    selected = _select_prompts(banks, limit_per_bank)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    run_body = _run_body(plan, smoke, revision, banks, selected, limit_per_bank)
    run_payload = _bound_payload(run_body, "run_sha256")
    _write_or_validate(output_dir / "run.json", run_payload, "run receipt")
    pending = _pending_prompts(output_dir, selected, plan.max_new_tokens)
    if pending:
        runtime = run(smoke, plan, tuple(item.prompt for item in pending))
        runtime_payload = _validate_runtime(runtime, smoke)
        _write_or_validate(
            output_dir / "runtime.json", runtime_payload, "runtime receipt"
        )
        _write_observations(
            output_dir, pending, runtime.observations, plan.max_new_tokens
        )
    return _finalize(plan, smoke, output_dir, run_payload, selected)
