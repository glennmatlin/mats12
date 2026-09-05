"""Compact, replayable evidence for a retained generation output directory."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

from concept_filtering.serialization import (
    canonical_sha256,
    file_sha256,
    json_equal_exact,
    json_loads_unique,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_RUN_FIELDS = {
    "schema_version",
    "generation_id",
    "generation_plan_sha256",
    "smoke_plan_sha256",
    "organism_sha256",
    "code_revision",
    "limit_per_bank",
    "max_new_tokens",
    "do_sample",
    "scientific_use",
    "starts_experiment_timer",
    "banks",
    "selected",
    "run_sha256",
}
_RUNTIME_FIELDS = {
    "adapter_sha256",
    "template_sha256",
    "runtime_versions",
    "runtime_sha256",
}
_AGGREGATE_FIELDS = {
    "schema_version",
    "generation_id",
    "run_sha256",
    "runtime_sha256",
    "sample_count",
    "generated_tokens",
    "elapsed_seconds",
    "hit_token_limit_count",
    "samples_sha256",
    "scientific_use",
    "starts_experiment_timer",
    "receipt_sha256",
}
_SELECTED_FIELDS = {
    "index",
    "behavior",
    "question_id",
    "prompt_sha256",
    "bank_output_sha256",
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
_ENTRY_FIELDS = {
    "index",
    "behavior",
    "question_id",
    "sample_sha256",
    "sample_file_sha256",
    "completion_sha256",
}
_EVIDENCE_FIELDS = {
    "schema_version",
    "evidence_id",
    "generation_id",
    "output_directory",
    "executor_revision",
    "run_sha256",
    "run_file_sha256",
    "runtime_sha256",
    "runtime_file_sha256",
    "aggregate_receipt_sha256",
    "aggregate_receipt_file_sha256",
    "sample_count",
    "samples_sha256",
    "completion_hashes_sha256",
    "behaviors",
    "sample_receipts",
    "scientific_use",
    "starts_experiment_timer",
    "evidence_sha256",
}


class GenerationEvidenceError(ValueError):
    """Raised when retained generation evidence is inconsistent."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationEvidenceError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise GenerationEvidenceError(f"{field} must be a JSON array")
    return cast(list[object], value)


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        return _mapping(json_loads_unique(path.read_text(encoding="utf-8")), field)
    except (OSError, UnicodeError, ValueError) as error:
        raise GenerationEvidenceError(f"cannot load {field}: {error}") from error


def _exact_fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        raise GenerationEvidenceError(f"{field} fields do not match schema")


def _validate_bound_hash(payload: dict[str, Any], field: str, label: str) -> None:
    body = dict(payload)
    stored = body.pop(field, None)
    if not isinstance(stored, str) or canonical_sha256(body) != stored:
        raise GenerationEvidenceError(f"{label} hash does not match")


def _false_flag(payload: dict[str, Any], field: str, label: str) -> None:
    if payload.get(field) is not False:
        raise GenerationEvidenceError(f"{label} {field} must remain false")


def _validate_selected(raw: object) -> list[dict[str, Any]]:
    selected = [_mapping(item, "selected item") for item in _sequence(raw, "selected")]
    if not selected:
        raise GenerationEvidenceError("selected cannot be empty")
    for item in selected:
        _exact_fields(item, _SELECTED_FIELDS, "selected item")
    indices = [item.get("index") for item in selected]
    identities = [(item.get("behavior"), item.get("question_id")) for item in selected]
    if indices != list(range(len(selected))) or len(set(identities)) != len(selected):
        raise GenerationEvidenceError("selected identities are not ordered and unique")
    return selected


def _bank_counts(raw: object) -> dict[str, int]:
    banks = [_mapping(item, "bank") for item in _sequence(raw, "banks")]
    counts: dict[str, int] = {}
    for bank in banks:
        behavior = bank.get("behavior")
        count = bank.get("question_count")
        if not isinstance(behavior, str) or type(count) is not int or count <= 0:
            raise GenerationEvidenceError("bank identity or count is invalid")
        counts[behavior] = count
    if len(counts) != len(banks):
        raise GenerationEvidenceError("bank behaviors are not unique")
    return counts


def _validate_run(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _exact_fields(payload, _RUN_FIELDS, "run receipt")
    _validate_bound_hash(payload, "run_sha256", "run receipt")
    _false_flag(payload, "scientific_use", "run receipt")
    _false_flag(payload, "starts_experiment_timer", "run receipt")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise GenerationEvidenceError("run receipt execution policy is invalid")
    if payload.get("do_sample") is not False:
        raise GenerationEvidenceError("run receipt execution policy is invalid")
    if payload.get("limit_per_bank") is not None:
        raise GenerationEvidenceError("full-bank run cannot have a prompt limit")
    selected = _validate_selected(payload.get("selected"))
    selected_counts = Counter(cast(str, item["behavior"]) for item in selected)
    if dict(selected_counts) != _bank_counts(payload.get("banks")):
        raise GenerationEvidenceError("run bank and selected counts do not match")
    return selected


def _validate_runtime(payload: dict[str, Any]) -> None:
    _exact_fields(payload, _RUNTIME_FIELDS, "runtime receipt")
    _validate_bound_hash(payload, "runtime_sha256", "runtime receipt")
    versions = payload.get("runtime_versions")
    if not isinstance(versions, dict) or not versions:
        raise GenerationEvidenceError("runtime versions are invalid")


def _validate_sample_values(payload: dict[str, Any], max_new_tokens: int) -> None:
    completion = payload.get("completion")
    generated = payload.get("generated_tokens")
    elapsed = payload.get("elapsed_seconds")
    if not isinstance(completion, str) or not completion.strip():
        raise GenerationEvidenceError("sample completion is invalid")
    if payload.get("completion_sha256") != canonical_sha256(completion):
        raise GenerationEvidenceError("sample completion hash does not match")
    if type(generated) is not int or not 1 <= generated <= max_new_tokens:
        raise GenerationEvidenceError("sample generated-token count is invalid")
    if (
        not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise GenerationEvidenceError("sample elapsed time is invalid")
    if type(payload.get("input_tokens")) is not int or payload["input_tokens"] <= 0:
        raise GenerationEvidenceError("sample input-token count is invalid")
    if type(payload.get("hit_token_limit")) is not bool:
        raise GenerationEvidenceError("sample token-limit flag is invalid")


def _load_sample(
    path: Path, selected: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    payload = _load_json(path, "sample receipt")
    _exact_fields(payload, _SAMPLE_FIELDS, "sample receipt")
    _validate_bound_hash(payload, "sample_sha256", "sample receipt")
    bound_fields = (
        "index",
        "behavior",
        "question_id",
        "prompt_sha256",
        "bank_output_sha256",
    )
    observed = [payload.get(field) for field in bound_fields]
    expected = [selected.get(field) for field in bound_fields]
    if not json_equal_exact(observed, expected):
        raise GenerationEvidenceError("sample receipt does not match run selection")
    _validate_sample_values(payload, max_new_tokens)
    return payload


def _sample_entry(path: Path, payload: dict[str, Any]) -> dict[str, object]:
    return {
        "index": payload["index"],
        "behavior": payload["behavior"],
        "question_id": payload["question_id"],
        "sample_sha256": payload["sample_sha256"],
        "sample_file_sha256": file_sha256(path),
        "completion_sha256": payload["completion_sha256"],
    }


def _load_samples(
    output_dir: Path, selected: list[dict[str, Any]], max_new_tokens: int
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    expected = [
        output_dir / "samples" / f"{index:04d}.json" for index in range(len(selected))
    ]
    observed = sorted((output_dir / "samples").glob("*.json"))
    if observed != expected:
        raise GenerationEvidenceError("sample receipt file set does not match run")
    samples = [
        _load_sample(path, item, max_new_tokens)
        for path, item in zip(expected, selected, strict=True)
    ]
    return samples, [
        _sample_entry(path, sample)
        for path, sample in zip(expected, samples, strict=True)
    ]


def _aggregate_body(
    run: dict[str, Any], runtime: dict[str, Any], samples: list[dict[str, Any]]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generation_id": run["generation_id"],
        "run_sha256": run["run_sha256"],
        "runtime_sha256": runtime["runtime_sha256"],
        "sample_count": len(samples),
        "generated_tokens": sum(
            cast(int, sample["generated_tokens"]) for sample in samples
        ),
        "elapsed_seconds": sum(
            cast(float, sample["elapsed_seconds"]) for sample in samples
        ),
        "hit_token_limit_count": sum(
            sample["hit_token_limit"] is True for sample in samples
        ),
        "samples_sha256": canonical_sha256(
            [sample["sample_sha256"] for sample in samples]
        ),
        "scientific_use": False,
        "starts_experiment_timer": False,
    }


def _validate_aggregate(
    payload: dict[str, Any],
    run: dict[str, Any],
    runtime: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    _exact_fields(payload, _AGGREGATE_FIELDS, "aggregate receipt")
    _validate_bound_hash(payload, "receipt_sha256", "aggregate receipt")
    body = {
        field: value for field, value in payload.items() if field != "receipt_sha256"
    }
    if not json_equal_exact(body, _aggregate_body(run, runtime, samples)):
        raise GenerationEvidenceError(
            "aggregate receipt does not match retained samples"
        )


def _behavior_summaries(entries: list[dict[str, object]]) -> dict[str, object]:
    behaviors = sorted({cast(str, entry["behavior"]) for entry in entries})
    return {
        behavior: {
            "sample_count": sum(entry["behavior"] == behavior for entry in entries),
            "completion_hashes_sha256": canonical_sha256(
                [
                    entry["completion_sha256"]
                    for entry in entries
                    if entry["behavior"] == behavior
                ]
            ),
        }
        for behavior in behaviors
    }


def _file_bindings(
    output_dir: Path,
    run: dict[str, Any],
    runtime: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, object]:
    return {
        "run_sha256": run["run_sha256"],
        "run_file_sha256": file_sha256(output_dir / "run.json"),
        "runtime_sha256": runtime["runtime_sha256"],
        "runtime_file_sha256": file_sha256(output_dir / "runtime.json"),
        "aggregate_receipt_sha256": aggregate["receipt_sha256"],
        "aggregate_receipt_file_sha256": file_sha256(output_dir / "receipt.json"),
    }


def _evidence_body(
    output_dir: Path,
    evidence_id: str,
    run: dict[str, Any],
    runtime: dict[str, Any],
    aggregate: dict[str, Any],
    entries: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "generation_id": run["generation_id"],
        "output_directory": str(output_dir),
        "executor_revision": run["code_revision"],
        **_file_bindings(output_dir, run, runtime, aggregate),
        "sample_count": len(entries),
        "samples_sha256": canonical_sha256(
            [entry["sample_sha256"] for entry in entries]
        ),
        "completion_hashes_sha256": canonical_sha256(
            [entry["completion_sha256"] for entry in entries]
        ),
        "behaviors": _behavior_summaries(entries),
        "sample_receipts": entries,
        "scientific_use": False,
        "starts_experiment_timer": False,
    }


def build_generation_evidence(
    output_dir: Path, *, evidence_id: str
) -> dict[str, object]:
    run = _load_json(output_dir / "run.json", "run receipt")
    selected = _validate_run(run)
    runtime = _load_json(output_dir / "runtime.json", "runtime receipt")
    _validate_runtime(runtime)
    max_new_tokens = run.get("max_new_tokens")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise GenerationEvidenceError("run max_new_tokens is invalid")
    samples, entries = _load_samples(output_dir, selected, max_new_tokens)
    aggregate = _load_json(output_dir / "receipt.json", "aggregate receipt")
    _validate_aggregate(aggregate, run, runtime, samples)
    body = _evidence_body(output_dir, evidence_id, run, runtime, aggregate, entries)
    return {**body, "evidence_sha256": canonical_sha256(body)}


def _validate_evidence_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [
        _mapping(item, "generation evidence sample")
        for item in _sequence(payload.get("sample_receipts"), "sample_receipts")
    ]
    for entry in entries:
        _exact_fields(entry, _ENTRY_FIELDS, "generation evidence sample")
        _validate_evidence_entry(entry)
    indices = [entry.get("index") for entry in entries]
    identities = [
        (entry.get("behavior"), entry.get("question_id")) for entry in entries
    ]
    if indices != list(range(len(entries))) or len(set(identities)) != len(entries):
        raise GenerationEvidenceError(
            "generation evidence samples are not ordered and unique"
        )
    return entries


def _validate_evidence_entry(entry: dict[str, Any]) -> None:
    if type(entry.get("index")) is not int or entry["index"] < 0:
        raise GenerationEvidenceError("generation evidence sample index is invalid")
    for field in ("behavior", "question_id"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise GenerationEvidenceError(
                f"generation evidence sample {field} is invalid"
            )
    for field in ("sample_sha256", "sample_file_sha256", "completion_sha256"):
        value = entry.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise GenerationEvidenceError(
                f"generation evidence sample {field} is invalid"
            )


def _validate_evidence_summaries(
    payload: dict[str, Any], entries: list[dict[str, Any]]
) -> None:
    expected = {
        "sample_count": len(entries),
        "samples_sha256": canonical_sha256(
            [entry.get("sample_sha256") for entry in entries]
        ),
        "completion_hashes_sha256": canonical_sha256(
            [entry.get("completion_sha256") for entry in entries]
        ),
        "behaviors": _behavior_summaries(cast(list[dict[str, object]], entries)),
    }
    if any(
        not json_equal_exact(payload.get(field), value)
        for field, value in expected.items()
    ):
        raise GenerationEvidenceError(
            "generation evidence summary does not match samples"
        )


def load_generation_evidence(path: Path) -> dict[str, Any]:
    payload = _load_json(path, "generation evidence")
    _exact_fields(payload, _EVIDENCE_FIELDS, "generation evidence")
    _validate_bound_hash(payload, "evidence_sha256", "generation evidence")
    _false_flag(payload, "scientific_use", "generation evidence")
    _false_flag(payload, "starts_experiment_timer", "generation evidence")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise GenerationEvidenceError("generation evidence schema_version must be 1")
    entries = _validate_evidence_entries(payload)
    _validate_evidence_summaries(payload, entries)
    return payload
