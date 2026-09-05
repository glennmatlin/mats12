from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import concept_filtering.cli as cli
import concept_filtering.smoke as smoke
from concept_filtering.cli import main
from concept_filtering.smoke import RuntimeObservation, SmokePlan

ROOT = Path(__file__).parents[1]


def _clean_git_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "executor.txt").write_text("reviewed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "executor.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "Test executor",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_validate_receipt_prints_machine_readable_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "model",
                "kind": "model",
                "status": "verified",
                "repository": "example/model",
                "revision": "a" * 40,
                "path": None,
                "sha256": None,
                "source_url": "https://example.test/model",
                "evidence": "Fixture.",
                "divergence": None,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-receipt", str(receipt_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "artifact_id": "model",
        "kind": "model",
        "revision": "a" * 40,
        "status": "verified",
    }


def test_validate_receipt_returns_two_for_invalid_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt_path = tmp_path / "invalid.json"
    receipt_path.write_text("{}", encoding="utf-8")

    exit_code = main(["validate-receipt", str(receipt_path)])

    assert exit_code == 2
    error = capsys.readouterr().err
    assert "schema_version" in error


def test_validate_smoke_plan_prints_interim_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "validate-smoke-plan",
            str(ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "decision_status": "approved_interim",
        "rank": 32,
        "scientific_use": False,
        "smoke_id": "public-rank32-split-1-smoke-v1",
        "starts_experiment_timer": False,
    }


def test_run_smoke_uses_runtime_and_writes_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    expected_sha256 = ROOT / "configs/reconstruction/receipts/full-data-adapter.json"
    adapter_receipt = json.loads(expected_sha256.read_text(encoding="utf-8"))

    def fake_run(plan: SmokePlan, cache_dir: Path) -> RuntimeObservation:
        del cache_dir
        return RuntimeObservation(
            completion="Smoke response.",
            adapter_sha256=adapter_receipt["sha256"],
            template_sha256=plan.template_sha256,
            runtime_versions={"transformers": "4.57.6"},
        )

    monkeypatch.setattr(
        cli,
        "run_transformers_smoke",
        fake_run,
    )
    output = tmp_path / "receipt.json"
    repository_root = tmp_path / "repository"
    revision = _clean_git_repo(repository_root)
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository_root)

    exit_code = main(
        [
            "run-smoke",
            str(plan_path),
            "--code-revision",
            revision,
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["scientific_use"] is False
    assert summary["starts_experiment_timer"] is False
    assert len(summary["receipt_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8"))["rank"] == 32
