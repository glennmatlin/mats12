"""Resumable, exclusive-write execution of base-response shards."""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.base_response_generation import BaseResponseConfig
from concept_filtering.base_response_join import JoinedExample
from concept_filtering.serialization import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_sha256,
    json_loads_unique,
)

# Timing stays outside the sample so replays and races are byte-identical.
_SAMPLE_FIELDS = {
    "schema_version",
    "generation_id",
    "config_sha256",
    "example_id",
    "row_ordinal",
    "orig_idx",
    "prompt_messages_sha256",
    "supplied_response_sha256",
    "prompt_token_count",
    "max_new_tokens",
    "completion_token_count",
    "response_text",
    "response_sha256",
    "terminated_by",
    "sample_sha256",
}


class BaseResponseRunnerError(ValueError):
    """Raised when shard execution or an existing sample breaks the contract."""


@dataclass(frozen=True)
class GeneratedResponse:
    prompt_token_count: int
    completion_token_count: int
    response_text: str
    terminated_by_eos: bool
    duration_seconds: float


@dataclass(frozen=True)
class ShardReport:
    claimed: int
    generated: int
    replayed: int


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaseResponseRunnerError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _file_identity(item: os.stat_result, label: str) -> tuple[int, int, int, int]:
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        raise BaseResponseRunnerError(f"{label} must be a unique file")
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns


def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise BaseResponseRunnerError(f"cannot read {label}: {error}") from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()), label)
            source = handle.read()
            after = _file_identity(os.fstat(handle.fileno()), label)
        path_identity = _file_identity(path.lstat(), label)
    except OSError as error:
        raise BaseResponseRunnerError(f"cannot read {label}: {error}") from error
    if before != after or path_identity != after:
        raise BaseResponseRunnerError(f"{label} changed while reading")
    return source


def read_sample_bytes(path: Path, label: str) -> bytes:
    """Read one sample while failing closed on aliases and mid-read mutation."""
    return _regular_bytes(path, label)


def _staged_bytes(path: Path, value: bytes, label: str) -> Path:
    temporary = path.with_name(f"{path.name}.stage-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise BaseResponseRunnerError(f"cannot stage {label}: {error}") from error
    return temporary


def _publish_sample(path: Path, value: bytes, label: str) -> bool:
    """Publish atomically; return True when a matching sample already existed."""
    temporary = _staged_bytes(path, value, label)
    try:
        try:
            os.link(temporary, path)
            return False
        except FileExistsError:
            existing = _regular_bytes(path, label)
            if existing != value:
                raise BaseResponseRunnerError(
                    f"{label} contradicts a prior sample"
                ) from None
            return True
        except OSError as error:
            raise BaseResponseRunnerError(f"cannot publish {label}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _max_new_tokens(config: BaseResponseConfig, prompt_token_count: object) -> int:
    if type(prompt_token_count) is not int or prompt_token_count <= 0:
        raise BaseResponseRunnerError("prompt token count is invalid")
    capacity = config.generation.max_total_tokens - prompt_token_count
    if capacity <= 0:
        raise BaseResponseRunnerError("prompt leaves no completion capacity")
    return capacity


def _validated_generation(
    config: BaseResponseConfig, response: GeneratedResponse
) -> int:
    capacity = _max_new_tokens(config, response.prompt_token_count)
    completion = response.completion_token_count
    if type(completion) is not int or completion < 0 or completion > capacity:
        raise BaseResponseRunnerError("completion token count is invalid")
    if not response.terminated_by_eos and completion != capacity:
        raise BaseResponseRunnerError("cap termination must exhaust capacity")
    duration = response.duration_seconds
    if type(duration) is not float or not math.isfinite(duration) or duration < 0:
        raise BaseResponseRunnerError("duration must be a finite non-negative float")
    return capacity


def _binding_values(
    config: BaseResponseConfig, example: JoinedExample
) -> dict[str, object]:
    return {
        "schema_version": config.output.sample_schema_version,
        "generation_id": config.generation_id,
        "config_sha256": config.config_sha256,
        "example_id": example.example_id,
        "row_ordinal": example.row_ordinal,
        "orig_idx": example.orig_idx,
        "prompt_messages_sha256": example.prompt_messages_sha256,
        "supplied_response_sha256": example.supplied_response_sha256,
    }


def _sample_body(
    config: BaseResponseConfig,
    example: JoinedExample,
    response: GeneratedResponse,
    capacity: int,
) -> dict[str, object]:
    return {
        **_binding_values(config, example),
        "prompt_token_count": response.prompt_token_count,
        "max_new_tokens": capacity,
        "completion_token_count": response.completion_token_count,
        "response_text": response.response_text,
        "response_sha256": bytes_sha256(response.response_text.encode()),
        "terminated_by": "eos" if response.terminated_by_eos else "cap",
    }


def _sample_bytes(body: dict[str, object]) -> bytes:
    signed = {**body, "sample_sha256": canonical_sha256(body)}
    return canonical_json_bytes(signed) + b"\n"


def _validate_sample_binding(
    payload: Mapping[str, object],
    config: BaseResponseConfig,
    example: JoinedExample,
    label: str,
) -> None:
    for field, value in _binding_values(config, example).items():
        observed = payload.get(field)
        if observed != value or type(observed) is not type(value):
            raise BaseResponseRunnerError(f"{label}: {field} contradicts this run")


def _validate_sample_accounting(
    payload: Mapping[str, object], config: BaseResponseConfig, label: str
) -> None:
    capacity = _max_new_tokens(config, payload.get("prompt_token_count"))
    if payload.get("max_new_tokens") != capacity:
        raise BaseResponseRunnerError(f"{label}: max_new_tokens is inconsistent")
    completion = payload.get("completion_token_count")
    if type(completion) is not int or completion < 0 or completion > capacity:
        raise BaseResponseRunnerError(f"{label}: completion token count is invalid")
    terminated = payload.get("terminated_by")
    if terminated not in ("eos", "cap"):
        raise BaseResponseRunnerError(f"{label}: terminated_by is invalid")
    if terminated == "cap" and completion != capacity:
        raise BaseResponseRunnerError(f"{label}: cap termination must exhaust capacity")
    text = payload.get("response_text")
    if not isinstance(text, str):
        raise BaseResponseRunnerError(f"{label}: response text must be a string")
    if payload.get("response_sha256") != bytes_sha256(text.encode()):
        raise BaseResponseRunnerError(f"{label}: response SHA-256 is invalid")


def validate_sample_payload(
    source: bytes,
    config: BaseResponseConfig,
    example: JoinedExample,
    label: str,
) -> dict[str, Any]:
    """Validate one existing sample's schema, signature, binding, and accounting."""
    try:
        payload = _object(json_loads_unique(source.decode()), label)
    except (UnicodeError, ValueError) as error:
        raise BaseResponseRunnerError(f"cannot parse {label}: {error}") from error
    if set(payload) != _SAMPLE_FIELDS:
        raise BaseResponseRunnerError(f"{label} fields differ from schema")
    body = {
        field: value for field, value in payload.items() if field != "sample_sha256"
    }
    if payload.get("sample_sha256") != canonical_sha256(body):
        raise BaseResponseRunnerError(f"{label}: sample SHA-256 is invalid")
    _validate_sample_binding(payload, config, example, label)
    _validate_sample_accounting(payload, config, label)
    return payload


def _replay_existing(
    path: Path, config: BaseResponseConfig, example: JoinedExample
) -> bool:
    if not path.is_symlink() and not path.exists():
        return False
    label = f"sample {example.example_id}"
    source = _regular_bytes(path, label)
    validate_sample_payload(source, config, example, label)
    return True


def claim_shard(
    examples: tuple[JoinedExample, ...], shard_index: int, shard_count: int
) -> tuple[JoinedExample, ...]:
    """Assign examples to shards deterministically by retained row ordinal."""
    if type(shard_count) is not int or shard_count < 1:
        raise BaseResponseRunnerError("shard count must be a positive integer")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise BaseResponseRunnerError("shard index must be within the shard count")
    return tuple(
        example
        for example in examples
        if example.row_ordinal % shard_count == shard_index
    )


def run_base_response_shard(
    config: BaseResponseConfig,
    examples: tuple[JoinedExample, ...],
    *,
    shard_index: int,
    shard_count: int,
    output_dir: Path,
    backend: Callable[[JoinedExample], GeneratedResponse],
) -> ShardReport:
    """Generate or replay every sample this shard owns, one atomic file each."""
    claimed = claim_shard(examples, shard_index, shard_count)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise BaseResponseRunnerError(
            f"cannot create output directory: {error}"
        ) from error
    generated = replayed = 0
    for example in claimed:
        target = output_dir / f"{example.example_id}.json"
        if _replay_existing(target, config, example):
            replayed += 1
            continue
        response = backend(example)
        capacity = _validated_generation(config, response)
        body = _sample_body(config, example, response, capacity)
        if _publish_sample(target, _sample_bytes(body), f"sample {example.example_id}"):
            replayed += 1
        else:
            generated += 1
    return ShardReport(len(claimed), generated, replayed)
