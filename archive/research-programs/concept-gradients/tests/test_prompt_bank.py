from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from concept_filtering.cli import main
from concept_filtering.inspect_archive import InspectArchiveValidationError
from concept_filtering.prompt_bank import extract_prompt_bank, load_prompt_bank


def _write_archive(
    path: Path,
    *,
    task: str = "c12-valid-feelings-sft",
    duplicate_id: bool = False,
    malformed_messages: bool = False,
) -> str:
    ids = ["question-2", "question-1"]
    header = {
        "status": "success",
        "eval": {
            "task": task,
            "dataset": {"samples": 2, "sample_ids": ids, "shuffled": False},
            "revision": {
                "origin": "git@example.test/research.git",
                "commit": "abc1234",
                "dirty": True,
            },
        },
        "results": {
            "total_samples": 2,
            "completed_samples": 2,
            "scores": [
                {
                    "name": "rubric_scorer",
                    "scorer": "rubric_scorer",
                    "scored_samples": 2,
                    "unscored_samples": 0,
                    "metrics": {"mean": {"value": 1.0}},
                }
            ],
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("header.json", json.dumps(header))
        for index, question_id in enumerate(ids):
            sample_id = ids[0] if duplicate_id and index else question_id
            user = f"User prompt {index}"
            if task == "c12-valid-feelings-sft":
                messages: list[dict[str, object]] = [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": user},
                    {
                        "role": "assistant",
                        "content": "wrong" if malformed_messages else "<think>",
                    },
                    {"role": "assistant", "content": "Archived answer"},
                ]
            else:
                messages = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "Archived answer"},
                ]
            archive.writestr(
                f"samples/{index:03d}.json",
                json.dumps({"id": sample_id, "input": user, "messages": messages}),
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extracts_deterministic_validation_prompt_bank(tmp_path: Path) -> None:
    archive = tmp_path / "source.eval"
    expected_sha = _write_archive(archive)
    output = tmp_path / "prompts.jsonl"
    receipt = tmp_path / "receipt.json"

    extraction = extract_prompt_bank(
        archive,
        expected_sha256=expected_sha,
        output_path=output,
        receipt_path=receipt,
    )

    assert extraction.behavior == "validate_feelings"
    assert extraction.question_count == 2
    loaded, prompts = load_prompt_bank(receipt, output_path=output)
    assert loaded == extraction
    assert [prompt.question_id for prompt in prompts] == ["question-1", "question-2"]
    assert prompts[0].messages == (
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt 1"},
    )


def test_extracts_user_only_refusal_prompts(tmp_path: Path) -> None:
    archive = tmp_path / "source.eval"
    expected_sha = _write_archive(archive, task="L05-refuse-then-redirect")

    extraction = extract_prompt_bank(
        archive,
        expected_sha256=expected_sha,
        output_path=tmp_path / "prompts.jsonl",
        receipt_path=tmp_path / "receipt.json",
    )
    _, prompts = load_prompt_bank(
        tmp_path / "receipt.json", output_path=tmp_path / "prompts.jsonl"
    )

    assert extraction.behavior == "refusal"
    assert prompts[0].messages == ({"role": "user", "content": "User prompt 1"},)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"duplicate_id": True}, "duplicate question"),
        ({"malformed_messages": True}, "assistant prefill"),
    ],
)
def test_rejects_invalid_prompt_records(
    tmp_path: Path, kwargs: dict[str, bool], error: str
) -> None:
    archive = tmp_path / "source.eval"
    expected_sha = _write_archive(
        archive,
        duplicate_id=kwargs.get("duplicate_id", False),
        malformed_messages=kwargs.get("malformed_messages", False),
    )

    with pytest.raises(InspectArchiveValidationError, match=error):
        extract_prompt_bank(
            archive,
            expected_sha256=expected_sha,
            output_path=tmp_path / "prompts.jsonl",
            receipt_path=tmp_path / "receipt.json",
        )


def test_rejects_existing_output_without_overwriting(tmp_path: Path) -> None:
    archive = tmp_path / "source.eval"
    expected_sha = _write_archive(archive)
    output = tmp_path / "prompts.jsonl"
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(InspectArchiveValidationError, match="already exists"):
        extract_prompt_bank(
            archive,
            expected_sha256=expected_sha,
            output_path=output,
            receipt_path=tmp_path / "receipt.json",
        )

    assert output.read_text(encoding="utf-8") == "keep me\n"


def test_loaded_bank_rejects_mutated_output(tmp_path: Path) -> None:
    archive = tmp_path / "source.eval"
    expected_sha = _write_archive(archive)
    output = tmp_path / "prompts.jsonl"
    receipt = tmp_path / "receipt.json"
    extract_prompt_bank(
        archive,
        expected_sha256=expected_sha,
        output_path=output,
        receipt_path=receipt,
    )
    output.write_text(output.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(InspectArchiveValidationError, match="output hash mismatch"):
        load_prompt_bank(receipt, output_path=output)


def test_extract_prompt_bank_cli_writes_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "source.eval"
    expected_sha = _write_archive(archive)
    output = tmp_path / "prompts.jsonl"
    receipt = tmp_path / "receipt.json"

    exit_code = main(
        [
            "extract-prompt-bank",
            str(archive),
            "--expected-sha256",
            expected_sha,
            "--output",
            str(output),
            "--receipt",
            str(receipt),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["question_count"] == 2
    assert summary["behavior"] == "validate_feelings"
