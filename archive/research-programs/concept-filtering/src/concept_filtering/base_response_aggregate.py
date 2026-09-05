"""Status, finalization, and verification over the base-response sample set."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.base_response_generation import (
    BaseResponseConfig,
    BaseResponseEvidence,
    generation_payload,
)
from concept_filtering.base_response_join import JoinedExample
from concept_filtering.base_response_runner import (
    BaseResponseRunnerError,
    read_sample_bytes,
    validate_sample_payload,
)
from concept_filtering.serialization import (
    canonical_json_bytes,
    canonical_sha256,
    json_loads_unique,
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")


class BaseResponseAggregateError(ValueError):
    """Raised when the sample set cannot be summarized or finalized."""


@dataclass(frozen=True)
class StatusReport:
    expected: int
    complete: int
    missing: int
    invalid: int
    eos_stopped: int
    cap_stopped: int
    stage_leftovers: int


def _known_names(examples: tuple[JoinedExample, ...]) -> set[str]:
    return {f"{example.example_id}.json" for example in examples}


def _validate_directory_members(
    output_dir: Path, examples: tuple[JoinedExample, ...]
) -> int:
    known = _known_names(examples)
    stage_leftovers = 0
    for path in output_dir.iterdir():
        if path.name in known:
            continue
        if ".stage-" in path.name and path.name.split(".stage-")[0] in known:
            stage_leftovers += 1
            continue
        raise BaseResponseAggregateError(
            f"output directory contains unkeyed file {path.name}"
        )
    return stage_leftovers


def base_response_status(
    config: BaseResponseConfig,
    examples: tuple[JoinedExample, ...],
    output_dir: Path,
) -> StatusReport:
    """Summarize sample coverage without generating or repairing anything."""
    if not output_dir.is_dir():
        return StatusReport(len(examples), 0, len(examples), 0, 0, 0, 0)
    stage_leftovers = _validate_directory_members(output_dir, examples)
    complete = missing = invalid = eos = cap = 0
    for example in examples:
        path = output_dir / f"{example.example_id}.json"
        if not path.is_symlink() and not path.exists():
            missing += 1
            continue
        try:
            source = read_sample_bytes(path, f"sample {example.example_id}")
            payload = validate_sample_payload(
                source, config, example, f"sample {example.example_id}"
            )
        except (OSError, BaseResponseRunnerError):
            invalid += 1
            continue
        complete += 1
        if payload.get("terminated_by") == "eos":
            eos += 1
        else:
            cap += 1
    return StatusReport(
        len(examples), complete, missing, invalid, eos, cap, stage_leftovers
    )


@dataclass(frozen=True)
class AggregateReceipt:
    payload: dict[str, object]
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "receipt_sha256": self.receipt_sha256}


def _collect_samples(
    config: BaseResponseConfig,
    examples: tuple[JoinedExample, ...],
    output_dir: Path,
) -> list[dict[str, Any]]:
    if not output_dir.is_dir():
        raise BaseResponseAggregateError("every sample is missing")
    _validate_directory_members(output_dir, examples)
    samples: list[dict[str, Any]] = []
    for example in sorted(examples, key=lambda item: item.row_ordinal):
        path = output_dir / f"{example.example_id}.json"
        label = f"sample {example.example_id}"
        if not path.is_symlink() and not path.exists():
            raise BaseResponseAggregateError(f"{label} is missing")
        try:
            source = read_sample_bytes(path, label)
            samples.append(validate_sample_payload(source, config, example, label))
        except BaseResponseRunnerError as error:
            raise BaseResponseAggregateError(str(error)) from error
    return samples


def _sample_totals(samples: list[dict[str, Any]]) -> dict[str, object]:
    ordered = [
        {
            "example_id": sample["example_id"],
            "sample_sha256": sample["sample_sha256"],
            "response_sha256": sample["response_sha256"],
        }
        for sample in samples
    ]
    return {
        "sample_count": len(samples),
        "eos_count": sum(sample["terminated_by"] == "eos" for sample in samples),
        "cap_count": sum(sample["terminated_by"] == "cap" for sample in samples),
        "prompt_token_total": sum(
            cast(int, sample["prompt_token_count"]) for sample in samples
        ),
        "completion_token_total": sum(
            cast(int, sample["completion_token_count"]) for sample in samples
        ),
        "ordered_sample_sha256": canonical_sha256(ordered),
    }


def _aggregate_body(
    config: BaseResponseConfig,
    evidence: BaseResponseEvidence,
    samples: list[dict[str, Any]],
    executor_revision: str,
) -> dict[str, object]:
    return {
        "schema_version": config.output.aggregate_schema_version,
        "generation_id": config.generation_id,
        "config_sha256": config.config_sha256,
        "package_receipt_sha256": evidence.package_receipt_sha256,
        "b2_receipt_sha256": evidence.b2_receipt_sha256,
        "base": {
            "repository": config.base.repository,
            "revision": config.base.revision,
        },
        "tokenizer": {
            "repository": config.tokenizer.repository,
            "revision": config.tokenizer.revision,
        },
        "generation": generation_payload(config.generation),
        "executor_revision": executor_revision,
        **_sample_totals(samples),
    }


def _validated_revision(executor_revision: str) -> str:
    if _REVISION.fullmatch(executor_revision) is None:
        raise BaseResponseAggregateError(
            "executor revision must be a 40-character revision"
        )
    return executor_revision


def finalize_base_responses(
    config: BaseResponseConfig,
    evidence: BaseResponseEvidence,
    examples: tuple[JoinedExample, ...],
    output_dir: Path,
    output_path: Path,
    *,
    executor_revision: str,
) -> AggregateReceipt:
    """Publish the absent-only aggregate receipt after exact coverage."""
    revision = _validated_revision(executor_revision)
    samples = _collect_samples(config, examples, output_dir)
    body = _aggregate_body(config, evidence, samples, revision)
    receipt = AggregateReceipt(body, canonical_sha256(body))
    try:
        with output_path.open("xb") as handle:
            handle.write(canonical_json_bytes(receipt.to_dict()) + b"\n")
    except FileExistsError as error:
        raise BaseResponseAggregateError(
            f"refusing to overwrite aggregate receipt: {output_path}"
        ) from error
    except OSError as error:
        raise BaseResponseAggregateError(
            f"cannot publish aggregate receipt: {error}"
        ) from error
    return receipt


def verify_base_responses(
    config: BaseResponseConfig,
    evidence: BaseResponseEvidence,
    examples: tuple[JoinedExample, ...],
    output_dir: Path,
    receipt_path: Path,
) -> AggregateReceipt:
    """Replay every sample and the aggregate receipt without generating."""
    try:
        source = read_sample_bytes(receipt_path, "aggregate receipt")
    except BaseResponseRunnerError as error:
        raise BaseResponseAggregateError(str(error)) from error
    try:
        payload = json_loads_unique(source.decode())
    except (UnicodeError, ValueError) as error:
        raise BaseResponseAggregateError(
            f"cannot parse aggregate receipt: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise BaseResponseAggregateError("aggregate receipt must be a JSON object")
    observed = cast(dict[str, object], payload)
    body = {
        field: value for field, value in observed.items() if field != "receipt_sha256"
    }
    if observed.get("receipt_sha256") != canonical_sha256(body):
        raise BaseResponseAggregateError("aggregate receipt SHA-256 is invalid")
    revision = _validated_revision(cast(str, observed.get("executor_revision", "")))
    samples = _collect_samples(config, examples, output_dir)
    expected = _aggregate_body(config, evidence, samples, revision)
    if set(body) != set(expected):
        raise BaseResponseAggregateError("aggregate receipt fields differ from schema")
    for field, value in expected.items():
        if body.get(field) != value:
            raise BaseResponseAggregateError(f"aggregate {field} differs")
    return AggregateReceipt(body, cast(str, observed["receipt_sha256"]))
