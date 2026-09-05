from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from concept_filtering.inspect_archive import (
    InspectArchiveValidationError,
    extract_scores,
    load_extraction_receipt,
)


def write_archive(
    path: Path,
    *,
    samples: list[tuple[str, float]],
    completed_samples: int | None = None,
    task: str = "c12-valid-feelings-sft",
) -> str:
    header = {
        "status": "success",
        "eval": {
            "task": task,
            "dataset": {
                "samples": len(samples),
                "sample_ids": [question_id for question_id, _ in samples],
                "shuffled": False,
            },
            "revision": {
                "type": "git",
                "origin": "git@example.test/research.git",
                "commit": "abc1234",
                "dirty": True,
            },
        },
        "results": {
            "total_samples": len(samples),
            "completed_samples": (
                len(samples) if completed_samples is None else completed_samples
            ),
            "scores": [
                {
                    "name": "rubric_scorer",
                    "scorer": "rubric_scorer",
                    "scored_samples": len(samples),
                    "unscored_samples": 0,
                    "metrics": {
                        "mean": {
                            "name": "mean",
                            "value": (
                                sum(score for _, score in samples) / len(samples)
                                if samples
                                else 0.0
                            ),
                        }
                    },
                }
            ],
        },
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("header.json", json.dumps(header))
        for index, (question_id, score) in enumerate(samples, start=1):
            archive.writestr(
                f"samples/{index:03d}_{question_id}_epoch_1.json",
                json.dumps(
                    {
                        "id": question_id,
                        "epoch": 1,
                        "scores": {
                            "rubric_scorer": {
                                "value": score,
                                "explanation": "fixture",
                            }
                        },
                    }
                ),
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extracts_scores_and_provenance_without_inspect_dependency(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "source.eval"
    output_path = tmp_path / "scores.jsonl"
    receipt_path = tmp_path / "extraction-receipt.json"
    expected_sha256 = write_archive(
        archive_path, samples=[("question-2", 3.0), ("question-1", -1.0)]
    )

    extraction = extract_scores(
        archive_path,
        scorer="rubric_scorer",
        expected_sha256=expected_sha256,
        output_path=output_path,
        receipt_path=receipt_path,
    )

    assert extraction.question_count == 2
    assert extraction.mean_score == 1.0
    assert extraction.source_dirty is True
    assert extraction.source_commit == "abc1234"
    assert extraction.archive_sha256 == expected_sha256
    assert len(extraction.output_sha256) == 64
    assert len(extraction.receipt_sha256) == 64
    assert json.loads(receipt_path.read_text()) == extraction.to_dict()
    assert load_extraction_receipt(receipt_path, output_path=output_path) == extraction
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {
            "behavior": "validate_feelings",
            "question_id": "question-1",
            "score": -1.0,
        },
        {
            "behavior": "validate_feelings",
            "question_id": "question-2",
            "score": 3.0,
        },
    ]


def test_rejects_archive_hash_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.eval"
    write_archive(archive_path, samples=[("question-1", 1.0)])

    with pytest.raises(InspectArchiveValidationError, match="SHA-256 mismatch"):
        extract_scores(
            archive_path,
            scorer="rubric_scorer",
            expected_sha256="0" * 64,
            output_path=tmp_path / "scores.jsonl",
            receipt_path=tmp_path / "receipt.json",
        )


def test_rejects_incomplete_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.eval"
    expected_sha256 = write_archive(
        archive_path,
        samples=[("question-1", 1.0), ("question-2", 2.0)],
        completed_samples=1,
    )

    with pytest.raises(InspectArchiveValidationError, match="completed_samples"):
        extract_scores(
            archive_path,
            scorer="rubric_scorer",
            expected_sha256=expected_sha256,
            output_path=tmp_path / "scores.jsonl",
            receipt_path=tmp_path / "receipt.json",
        )


def test_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.eval"
    expected_sha256 = write_archive(
        archive_path,
        samples=[("question-1", 1.0), ("question-1", 2.0)],
    )

    with pytest.raises(InspectArchiveValidationError, match="duplicate question"):
        extract_scores(
            archive_path,
            scorer="rubric_scorer",
            expected_sha256=expected_sha256,
            output_path=tmp_path / "scores.jsonl",
            receipt_path=tmp_path / "receipt.json",
        )


def test_rejects_empty_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.eval"
    expected_sha256 = write_archive(archive_path, samples=[])

    with pytest.raises(InspectArchiveValidationError, match="positive integer"):
        extract_scores(
            archive_path,
            scorer="rubric_scorer",
            expected_sha256=expected_sha256,
            output_path=tmp_path / "scores.jsonl",
            receipt_path=tmp_path / "receipt.json",
        )


def test_rejects_unapproved_task_mapping(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.eval"
    expected_sha256 = write_archive(
        archive_path,
        samples=[("question-1", 1.0)],
        task="unrelated-task",
    )

    with pytest.raises(InspectArchiveValidationError, match="approved behavior"):
        extract_scores(
            archive_path,
            scorer="rubric_scorer",
            expected_sha256=expected_sha256,
            output_path=tmp_path / "scores.jsonl",
            receipt_path=tmp_path / "receipt.json",
        )


def test_extraction_receipt_rejects_mutated_output(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.eval"
    output_path = tmp_path / "scores.jsonl"
    receipt_path = tmp_path / "receipt.json"
    expected_sha256 = write_archive(archive_path, samples=[("question-1", 1.0)])
    extract_scores(
        archive_path,
        scorer="rubric_scorer",
        expected_sha256=expected_sha256,
        output_path=output_path,
        receipt_path=receipt_path,
    )
    output_path.write_text(output_path.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(InspectArchiveValidationError, match="output hash mismatch"):
        load_extraction_receipt(receipt_path, output_path=output_path)
