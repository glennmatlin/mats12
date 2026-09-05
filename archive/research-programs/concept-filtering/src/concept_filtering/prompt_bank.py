"""Receipt-bound prompt-bank extraction from pinned Inspect archives."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.inspect_archive import (
    InspectArchiveValidationError,
    InspectHeaderMetadata,
    approved_task_behavior,
    read_header_metadata,
    read_json_member,
    require_mapping,
    require_string,
    validate_archive_members,
    verified_archive_sha,
)
from concept_filtering.serialization import bytes_sha256, canonical_sha256, file_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCORER = "rubric_scorer"


@dataclass(frozen=True)
class PromptRecord:
    question_id: str
    behavior: str
    messages: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "behavior": self.behavior,
            "messages": list(self.messages),
            "question_id": self.question_id,
        }


@dataclass(frozen=True)
class PromptBankExtraction:
    schema_version: int
    archive_sha256: str
    task: str
    behavior: str
    question_count: int
    source_origin: str
    source_commit: str
    source_dirty: bool
    output_sha256: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "archive_sha256": self.archive_sha256,
            "task": self.task,
            "behavior": self.behavior,
            "question_count": self.question_count,
            "source_origin": self.source_origin,
            "source_commit": self.source_commit,
            "source_dirty": self.source_dirty,
            "output_sha256": self.output_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


def _sample_names(
    members: dict[str, zipfile.ZipInfo],
) -> list[str]:
    return sorted(
        name
        for name, member in members.items()
        if name.startswith("samples/")
        and name.endswith(".json")
        and not member.is_dir()
    )


def _message(value: object, context: str) -> dict[str, Any]:
    message = require_mapping(value, context)
    require_string(message, "role", context)
    return message


def _raw_messages(sample: dict[str, Any], context: str) -> list[dict[str, Any]]:
    value = sample.get("messages")
    if not isinstance(value, list):
        raise InspectArchiveValidationError(f"{context}.messages must be an array")
    return [
        _message(item, f"{context}.messages[{index}]")
        for index, item in enumerate(cast(list[object], value))
    ]


def _validation_messages(
    messages: list[dict[str, Any]], context: str
) -> tuple[dict[str, str], ...]:
    roles = [message["role"] for message in messages]
    if roles != ["system", "user", "assistant", "assistant"]:
        raise InspectArchiveValidationError(
            f"{context}.messages has unexpected validation role sequence"
        )
    if messages[2].get("content") != "<think>":
        raise InspectArchiveValidationError(
            f"{context}.messages assistant prefill must be <think>"
        )
    return (
        _prompt_message(messages[0], context),
        _prompt_message(messages[1], context),
    )


def _refusal_messages(
    messages: list[dict[str, Any]], context: str
) -> tuple[dict[str, str], ...]:
    roles = [message["role"] for message in messages]
    if roles != ["user", "assistant"]:
        raise InspectArchiveValidationError(
            f"{context}.messages has unexpected refusal role sequence"
        )
    return (_prompt_message(messages[0], context),)


def _prompt_message(message: dict[str, Any], context: str) -> dict[str, str]:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InspectArchiveValidationError(
            f"{context}.messages prompt content must be a non-empty string"
        )
    return {"role": cast(str, message["role"]), "content": content}


def _prompt_record(
    sample: dict[str, Any], *, behavior: str, context: str
) -> PromptRecord:
    question_id = require_string(sample, "id", context)
    input_text = require_string(sample, "input", context)
    raw_messages = _raw_messages(sample, context)
    messages = (
        _validation_messages(raw_messages, context)
        if behavior == "validate_feelings"
        else _refusal_messages(raw_messages, context)
    )
    if messages[-1]["content"] != input_text:
        raise InspectArchiveValidationError(
            f"{context}.input does not match the user prompt"
        )
    return PromptRecord(question_id, behavior, messages)


def _read_prompts(
    archive_path: Path,
) -> tuple[InspectHeaderMetadata, dict[str, PromptRecord]]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = validate_archive_members(archive)
            metadata = read_header_metadata(
                read_json_member(archive, members, "header.json"), _SCORER
            )
            prompts = _sample_prompts(archive, members, behavior=metadata.behavior)
            return metadata, prompts
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise InspectArchiveValidationError(
            f"cannot read Inspect archive {archive_path}: {error}"
        ) from error


def _sample_prompts(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    *,
    behavior: str,
) -> dict[str, PromptRecord]:
    prompts: dict[str, PromptRecord] = {}
    for name in _sample_names(members):
        sample = require_mapping(read_json_member(archive, members, name), name)
        prompt = _prompt_record(sample, behavior=behavior, context=name)
        if prompt.question_id in prompts:
            raise InspectArchiveValidationError(
                f"duplicate question id in sample records: {prompt.question_id}"
            )
        prompts[prompt.question_id] = prompt
    return prompts


def _validate_prompt_set(
    metadata: InspectHeaderMetadata, prompts: dict[str, PromptRecord]
) -> None:
    if len(prompts) != metadata.sample_count:
        raise InspectArchiveValidationError(
            f"sample record count {len(prompts)} does not match header "
            f"{metadata.sample_count}"
        )
    if frozenset(prompts) != metadata.sample_ids:
        raise InspectArchiveValidationError(
            "sample record ids do not match header sample_ids"
        )


def _prompt_bytes(prompts: dict[str, PromptRecord]) -> bytes:
    text = "".join(
        json.dumps(
            prompts[question_id].to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for question_id in sorted(prompts)
    )
    return text.encode("utf-8")


def _extraction_body(
    metadata: InspectHeaderMetadata, archive_sha256: str, output_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "archive_sha256": archive_sha256,
        "task": metadata.task,
        "behavior": metadata.behavior,
        "question_count": metadata.sample_count,
        "source_origin": metadata.source_origin,
        "source_commit": metadata.source_commit,
        "source_dirty": metadata.source_dirty,
        "output_sha256": output_sha256,
    }


def _build_extraction(
    metadata: InspectHeaderMetadata, archive_sha256: str, output_sha256: str
) -> PromptBankExtraction:
    body = _extraction_body(metadata, archive_sha256, output_sha256)
    return PromptBankExtraction(
        schema_version=1,
        archive_sha256=archive_sha256,
        task=metadata.task,
        behavior=metadata.behavior,
        question_count=metadata.sample_count,
        source_origin=metadata.source_origin,
        source_commit=metadata.source_commit,
        source_dirty=metadata.source_dirty,
        output_sha256=output_sha256,
        receipt_sha256=canonical_sha256(body),
    )


def _validate_request(
    expected_sha256: str, output_path: Path, receipt_path: Path
) -> None:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise InspectArchiveValidationError(
            "expected_sha256 must be 64 lowercase hexadecimal digits"
        )
    if output_path.resolve() == receipt_path.resolve():
        raise InspectArchiveValidationError("output and receipt paths must differ")
    existing = [str(path) for path in (output_path, receipt_path) if path.exists()]
    if existing:
        raise InspectArchiveValidationError(f"output already exists: {existing[0]}")


def _write_extraction(
    output_bytes: bytes,
    extraction: PromptBankExtraction,
    output_path: Path,
    receipt_path: Path,
) -> None:
    try:
        with output_path.open("xb") as handle:
            handle.write(output_bytes)
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(
                extraction.to_dict(),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    except (OSError, UnicodeError) as error:
        raise InspectArchiveValidationError(
            f"cannot write prompt-bank extraction: {error}"
        ) from error


def extract_prompt_bank(
    archive_path: Path,
    *,
    expected_sha256: str,
    output_path: Path,
    receipt_path: Path,
) -> PromptBankExtraction:
    """Verify one archive and write the exact model-input prompts."""
    _validate_request(expected_sha256, output_path, receipt_path)
    archive_sha256 = verified_archive_sha(archive_path, expected_sha256)
    metadata, prompts = _read_prompts(archive_path)
    _validate_prompt_set(metadata, prompts)
    output_bytes = _prompt_bytes(prompts)
    extraction = _build_extraction(metadata, archive_sha256, bytes_sha256(output_bytes))
    _write_extraction(output_bytes, extraction, output_path, receipt_path)
    return extraction


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InspectArchiveValidationError(
            f"cannot load prompt-bank receipt {path}: {error}"
        ) from error
    payload = require_mapping(value, "prompt-bank receipt")
    if set(payload) != set(PromptBankExtraction.__dataclass_fields__):
        raise InspectArchiveValidationError(
            "prompt-bank receipt fields do not match schema"
        )
    return payload


def _sha(payload: dict[str, Any], field: str) -> str:
    value = require_string(payload, field, "prompt-bank receipt")
    if _SHA256.fullmatch(value) is None:
        raise InspectArchiveValidationError(f"prompt-bank receipt.{field} is invalid")
    return value


def _parse_receipt(payload: dict[str, Any]) -> PromptBankExtraction:
    task = require_string(payload, "task", "prompt-bank receipt")
    behavior = require_string(payload, "behavior", "prompt-bank receipt")
    schema_version = payload.get("schema_version")
    question_count = payload.get("question_count")
    if schema_version != 1 or type(schema_version) is not int:
        raise InspectArchiveValidationError("prompt-bank schema_version must be 1")
    if type(question_count) is not int or question_count <= 0:
        raise InspectArchiveValidationError("prompt-bank question_count is invalid")
    if approved_task_behavior(task) != behavior:
        raise InspectArchiveValidationError("prompt-bank task behavior is invalid")
    dirty = payload.get("source_dirty")
    if type(dirty) is not bool:
        raise InspectArchiveValidationError("prompt-bank source_dirty must be boolean")
    body = dict(payload)
    receipt_sha = cast(str, body.pop("receipt_sha256"))
    if canonical_sha256(body) != receipt_sha:
        raise InspectArchiveValidationError("prompt-bank receipt hash mismatch")
    return PromptBankExtraction(
        schema_version=schema_version,
        archive_sha256=_sha(payload, "archive_sha256"),
        task=task,
        behavior=behavior,
        question_count=question_count,
        source_origin=require_string(payload, "source_origin", "prompt-bank receipt"),
        source_commit=require_string(payload, "source_commit", "prompt-bank receipt"),
        source_dirty=dirty,
        output_sha256=_sha(payload, "output_sha256"),
        receipt_sha256=_sha(payload, "receipt_sha256"),
    )


def _load_prompt_rows(path: Path, *, behavior: str) -> tuple[PromptRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InspectArchiveValidationError(
            f"cannot load prompt bank: {error}"
        ) from error
    prompts = tuple(_parse_prompt_row(value, behavior=behavior) for value in values)
    if tuple(prompt.question_id for prompt in prompts) != tuple(
        sorted(prompt.question_id for prompt in prompts)
    ):
        raise InspectArchiveValidationError("prompt bank is not sorted by question id")
    return prompts


def _parse_prompt_row(value: object, *, behavior: str) -> PromptRecord:
    row = require_mapping(value, "prompt-bank row")
    if set(row) != {"behavior", "messages", "question_id"}:
        raise InspectArchiveValidationError(
            "prompt-bank row fields do not match schema"
        )
    if row.get("behavior") != behavior:
        raise InspectArchiveValidationError("prompt-bank row behavior mismatch")
    messages = _raw_messages(row, "prompt-bank row")
    prompt_messages = _loaded_prompt_messages(messages, behavior=behavior)
    return PromptRecord(
        require_string(row, "question_id", "prompt-bank row"),
        behavior,
        prompt_messages,
    )


def _loaded_prompt_messages(
    messages: list[dict[str, Any]], *, behavior: str
) -> tuple[dict[str, str], ...]:
    roles = [message["role"] for message in messages]
    expected = ["system", "user"] if behavior == "validate_feelings" else ["user"]
    if roles != expected:
        raise InspectArchiveValidationError("prompt-bank message roles are invalid")
    return tuple(_prompt_message(message, "prompt-bank row") for message in messages)


def load_prompt_bank(
    receipt_path: Path, *, output_path: Path
) -> tuple[PromptBankExtraction, tuple[PromptRecord, ...]]:
    """Load a prompt bank only when its receipt and output still match."""
    receipt = _parse_receipt(_load_receipt(receipt_path))
    if receipt.schema_version != 1 or receipt.question_count <= 0:
        raise InspectArchiveValidationError("prompt-bank receipt schema is invalid")
    if file_sha256(output_path) != receipt.output_sha256:
        raise InspectArchiveValidationError("prompt-bank output hash mismatch")
    prompts = _load_prompt_rows(output_path, behavior=receipt.behavior)
    unique_count = len({prompt.question_id for prompt in prompts})
    if len(prompts) != receipt.question_count or unique_count != len(prompts):
        raise InspectArchiveValidationError("prompt-bank question count is invalid")
    return receipt, prompts
