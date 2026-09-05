from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest
from test_base_response_generation import VALID_CONFIG, b2_body, package_body
from test_base_response_join import (
    SOURCE_REVISION,
    build_entry,
    build_package_bytes,
    build_rows,
    read_rows,
    write_parquet,
)
from test_base_response_runner import RecordingBackend

import concept_filtering.base_response_generation as base_response_generation
import concept_filtering.cli as cli
from concept_filtering.cli import main
from concept_filtering.serialization import bytes_sha256, canonical_sha256

EXECUTOR_REVISION = "b" * 40


def _local_git_blob(root: Path, revision: str, path: str) -> bytes:
    del revision
    return (root / path).read_bytes()


def _snapshot_files(root: Path, names: list[str]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for name in names:
        value = f"snapshot {name}".encode()
        (root / name).write_bytes(value)
        manifest[name] = bytes_sha256(value)
    return manifest


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setattr(base_response_generation, "_git_blob", _local_git_blob)
    monkeypatch.setattr(cli, "_source_root_for_base_responses", lambda: tmp_path)

    parquet_path = tmp_path / "train_1.parquet"
    parquet_bytes = write_parquet(parquet_path, build_rows())
    entries = [
        build_entry(row, ordinal, overlength=(ordinal == 2))
        for ordinal, row in enumerate(read_rows(parquet_path))
    ]
    package_path = tmp_path / "package.jsonl"
    package_path.write_bytes(build_package_bytes(entries))

    receipt_body = package_body()
    receipt_body["entry_schema_version"] = 1
    receipt_body["sample"] = {
        "sha256": bytes_sha256(parquet_bytes),
        "source_revision": SOURCE_REVISION,
        "expected_row_count": 3,
        "expected_split": 1,
    }
    receipt_body["package_sha256"] = bytes_sha256(package_path.read_bytes())
    receipt_body["row_count"] = 3
    receipt_body["retained_row_count"] = 2
    receipt_body["retained_input_token_count"] = 14
    package_receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body),
    }
    gate_body = b2_body()
    b2_receipt = {**gate_body, "receipt_sha256": canonical_sha256(gate_body)}

    payload = copy.deepcopy(VALID_CONFIG)
    base_manifest = _snapshot_files(
        tmp_path / "base-snapshot",
        sorted(cast(dict[str, str], cast(dict[str, object], payload["base"])["files"])),
    )
    tokenizer_manifest = _snapshot_files(
        tmp_path / "tokenizer-snapshot" / "split-1",
        sorted(
            cast(dict[str, str], cast(dict[str, object], payload["tokenizer"])["files"])
        ),
    )
    cast(dict[str, object], payload["base"])["files"] = base_manifest
    cast(dict[str, object], payload["tokenizer"])["files"] = tokenizer_manifest
    package = cast(dict[str, object], payload["package"])
    gate = cast(dict[str, object], payload["b2_gate"])
    package["receipt_sha256"] = package_receipt["receipt_sha256"]
    gate["receipt_sha256"] = b2_receipt["receipt_sha256"]
    for relative, receipt in (
        ("reports/reconstruction/data/package.json", package_receipt),
        ("reports/reconstruction/smokes/b2.json", b2_receipt),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
    config_path = tmp_path / "base-responses.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "config": str(config_path),
        "package": str(package_path),
        "parquet": str(parquet_path),
        "base_snapshot": str(tmp_path / "base-snapshot"),
        "tokenizer_snapshot": str(tmp_path / "tokenizer-snapshot"),
        "output_dir": str(tmp_path / "responses"),
        "receipt": str(tmp_path / "aggregate.json"),
    }


def _shared_args(world: dict[str, str]) -> list[str]:
    return [
        "--config",
        world["config"],
        "--package",
        world["package"],
        "--parquet",
        world["parquet"],
    ]


def test_cli_plan_reports_joined_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = _world(tmp_path, monkeypatch)

    code = main(
        [
            "base-responses",
            "plan",
            *_shared_args(world),
            "--base-snapshot",
            world["base_snapshot"],
            "--tokenizer-snapshot",
            world["tokenizer_snapshot"],
        ]
    )

    assert code == 0
    result = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert result["retained_examples"] == 2
    assert result["generation_id"] == "fixture-base-responses-v1"
    assert result["model_loaded"] is False


def _run_shard(world: dict[str, str]) -> int:
    return main(
        [
            "base-responses",
            "run-shard",
            *_shared_args(world),
            "--base-snapshot",
            world["base_snapshot"],
            "--tokenizer-snapshot",
            world["tokenizer_snapshot"],
            "--shard-index",
            "0",
            "--shard-count",
            "1",
            "--output-dir",
            world["output_dir"],
        ]
    )


def _fake_backend(
    config: object,
    base_snapshot: Path,
    tokenizer_snapshot: Path,
    device: str | None,
    dtype: str | None,
) -> RecordingBackend:
    del config, base_snapshot, tokenizer_snapshot
    assert device is None
    assert dtype is None
    return RecordingBackend()


def test_cli_run_shard_threads_device_and_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = _world(tmp_path, monkeypatch)
    observed: list[tuple[str | None, str | None]] = []

    def _recording_backend(
        config: object,
        base_snapshot: Path,
        tokenizer_snapshot: Path,
        device: str | None,
        dtype: str | None,
    ) -> RecordingBackend:
        del config, base_snapshot, tokenizer_snapshot
        observed.append((device, dtype))
        return RecordingBackend()

    monkeypatch.setattr(cli, "_base_response_backend", _recording_backend)

    assert (
        main(
            [
                "base-responses",
                "run-shard",
                *_shared_args(world),
                "--base-snapshot",
                world["base_snapshot"],
                "--tokenizer-snapshot",
                world["tokenizer_snapshot"],
                "--shard-index",
                "0",
                "--shard-count",
                "1",
                "--output-dir",
                world["output_dir"],
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert observed == [("cpu", "float32")]


def test_cli_run_shard_status_finalize_verify_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = _world(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_base_response_backend", _fake_backend)

    def _pass_revision(value: str) -> str:
        return value

    monkeypatch.setattr(cli, "_base_response_executor_revision", _pass_revision)

    assert _run_shard(world) == 0
    shard = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert shard["claimed"] == 2
    assert shard["generated"] == 2

    assert (
        main(
            [
                "base-responses",
                "status",
                *_shared_args(world),
                "--output-dir",
                world["output_dir"],
            ]
        )
        == 0
    )
    status = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert status["complete"] == 2
    assert status["missing"] == 0

    assert (
        main(
            [
                "base-responses",
                "finalize",
                *_shared_args(world),
                "--output-dir",
                world["output_dir"],
                "--code-revision",
                EXECUTOR_REVISION,
                "--output",
                world["receipt"],
            ]
        )
        == 0
    )
    finalized = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert finalized["sample_count"] == 2

    assert (
        main(
            [
                "base-responses",
                "verify",
                *_shared_args(world),
                "--output-dir",
                world["output_dir"],
                "--receipt",
                world["receipt"],
            ]
        )
        == 0
    )
    verified = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert verified["verified"] is True
    assert verified["receipt_sha256"] == finalized["receipt_sha256"]


def test_cli_reports_contract_violation_as_user_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = _world(tmp_path, monkeypatch)
    Path(world["package"]).write_bytes(b"tampered\n")

    code = main(
        [
            "base-responses",
            "status",
            *_shared_args(world),
            "--output-dir",
            world["output_dir"],
        ]
    )

    assert code == 2
    assert "package" in capsys.readouterr().err
