from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from concept_filtering.cli import main
from concept_filtering.training_recipe_reconstruction import (
    TrainingRecipeReconstructionError,
    load_training_recipe_reconstruction,
    verify_training_recipe_reconstruction,
)


def _write_artifacts(path: Path) -> dict[str, str]:
    path.mkdir()
    adapter = {
        "base_model_name_or_path": "owner/base",
        "revision": None,
        "peft_type": "LORA",
        "peft_version": "0.18.1",
        "task_type": "CAUSAL_LM",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj"],
    }
    logs = [
        {"step": 1, "learning_rate": 0.0, "num_tokens": 10.0},
        {"step": 2, "learning_rate": 0.0002, "num_tokens": 20.0},
    ]
    trainer = {
        "global_step": 2,
        "max_steps": 2,
        "epoch": 1.0,
        "num_train_epochs": 1,
        "train_batch_size": 2,
        "num_input_tokens_seen": 0,
        "log_history": logs,
    }
    files = {
        "adapter_config.json": json.dumps(adapter).encode(),
        "trainer_state.json": json.dumps(trainer).encode(),
        "rng_state_0.pth": b"final rng state",
    }
    for name, content in files.items():
        (path / name).write_bytes(content)
    return {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }


def _unresolved() -> dict[str, object]:
    fields = [
        "optimizer",
        "scheduler_configuration",
        "gradient_accumulation",
        "initial_seed",
        "precision",
        "packing_and_sequence_policy",
        "data_order",
    ]
    return {
        field: {
            "status": "unavailable",
            "reason": f"The public artifacts do not recover {field}.",
        }
        for field in fields
    }


def _write_config(path: Path, files: dict[str, str]) -> None:
    payload = {
        "schema_version": 1,
        "reconstruction_id": "fixture-recipe",
        "scientific_use": False,
        "starts_experiment_timer": False,
        "artifact": {
            "repository": "owner/adapter",
            "revision": "a" * 40,
            "path": "split-1",
        },
        "files": files,
        "expected": {
            "adapter": {
                "base_model_name_or_path": "owner/base",
                "revision": None,
                "peft_type": "LORA",
                "peft_version": "0.18.1",
                "task_type": "CAUSAL_LM",
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "target_modules": ["k_proj", "q_proj"],
            },
            "trainer_state": {
                "global_step": 2,
                "max_steps": 2,
                "epoch": 1.0,
                "num_train_epochs": 1,
                "train_batch_size": 2,
                "num_input_tokens_seen": 0,
                "logged_step_count": 2,
                "final_logged_num_tokens": 20.0,
                "peak_learning_rate": 0.0002,
                "peak_learning_rate_step": 2,
                "final_logged_learning_rate": 0.0002,
            },
        },
        "unresolved": _unresolved(),
        "divergences": ["This is observed artifact evidence, not a runnable recipe."],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _replace_trainer_state(
    artifact_dir: Path, config_path: Path, trainer: dict[str, object]
) -> None:
    content = json.dumps(trainer).encode()
    (artifact_dir / "trainer_state.json").write_bytes(content)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["files"]["trainer_state.json"] = hashlib.sha256(content).hexdigest()
    history = cast(list[dict[str, object]], trainer["log_history"])
    payload["expected"]["trainer_state"].update(
        {
            "global_step": trainer["global_step"],
            "max_steps": trainer["max_steps"],
            "final_logged_num_tokens": history[-1]["num_tokens"],
        }
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def test_builds_receipt_from_exact_recipe_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "split-1"
    config_path = tmp_path / "config.json"
    _write_config(config_path, _write_artifacts(artifact_dir))

    receipt = verify_training_recipe_reconstruction(
        load_training_recipe_reconstruction(config_path),
        artifact_dir=artifact_dir,
    )

    assert receipt.rank == 32
    assert receipt.logged_step_count == 2
    assert receipt.final_logged_num_tokens == 20.0
    assert receipt.scientific_use is False
    assert receipt.starts_experiment_timer is False
    assert len(receipt.learning_rate_trace_sha256) == 64
    assert len(receipt.receipt_sha256) == 64


def test_rejects_recipe_artifact_hash_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "split-1"
    config_path = tmp_path / "config.json"
    _write_config(config_path, _write_artifacts(artifact_dir))
    (artifact_dir / "trainer_state.json").write_text("{}", encoding="utf-8")

    with pytest.raises(TrainingRecipeReconstructionError, match="SHA-256"):
        verify_training_recipe_reconstruction(
            load_training_recipe_reconstruction(config_path),
            artifact_dir=artifact_dir,
        )


def test_rejects_changed_expected_recipe_field(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "split-1"
    config_path = tmp_path / "config.json"
    _write_config(config_path, _write_artifacts(artifact_dir))
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["expected"]["adapter"]["r"] = 64
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingRecipeReconstructionError, match="adapter observation"):
        verify_training_recipe_reconstruction(
            load_training_recipe_reconstruction(config_path),
            artifact_dir=artifact_dir,
        )


def test_rejects_nonmonotonic_cumulative_token_trace(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "split-1"
    config_path = tmp_path / "config.json"
    _write_config(config_path, _write_artifacts(artifact_dir))
    trainer = json.loads((artifact_dir / "trainer_state.json").read_text())
    trainer["log_history"][-1]["num_tokens"] = 5.0
    _replace_trainer_state(artifact_dir, config_path, trainer)

    with pytest.raises(TrainingRecipeReconstructionError, match="nondecreasing"):
        verify_training_recipe_reconstruction(
            load_training_recipe_reconstruction(config_path),
            artifact_dir=artifact_dir,
        )


def test_rejects_step_count_inconsistent_with_completed_state(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "split-1"
    config_path = tmp_path / "config.json"
    _write_config(config_path, _write_artifacts(artifact_dir))
    trainer = json.loads((artifact_dir / "trainer_state.json").read_text())
    trainer["global_step"] = 3
    trainer["max_steps"] = 3
    _replace_trainer_state(artifact_dir, config_path, trainer)

    with pytest.raises(TrainingRecipeReconstructionError, match="step count"):
        verify_training_recipe_reconstruction(
            load_training_recipe_reconstruction(config_path),
            artifact_dir=artifact_dir,
        )


@pytest.mark.parametrize("field", ["scientific_use", "starts_experiment_timer"])
def test_rejects_enabled_recipe_boundary(tmp_path: Path, field: str) -> None:
    artifact_dir = tmp_path / "split-1"
    config_path = tmp_path / "config.json"
    _write_config(config_path, _write_artifacts(artifact_dir))
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload[field] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingRecipeReconstructionError, match="must remain false"):
        load_training_recipe_reconstruction(config_path)


def test_cli_writes_recipe_evidence_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_dir = tmp_path / "split-1"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "receipt.json"
    _write_config(config_path, _write_artifacts(artifact_dir))

    exit_code = main(
        [
            "verify-training-recipe-reconstruction",
            str(config_path),
            "--artifact-dir",
            str(artifact_dir),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["adapter_observation"]["r"] == 32
    assert json.loads(output_path.read_text(encoding="utf-8"))["receipt_sha256"]


def test_cli_refuses_to_overwrite_recipe_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_dir = tmp_path / "split-1"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "receipt.json"
    _write_config(config_path, _write_artifacts(artifact_dir))
    output_path.write_text("keep", encoding="utf-8")

    exit_code = main(
        [
            "verify-training-recipe-reconstruction",
            str(config_path),
            "--artifact-dir",
            str(artifact_dir),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert "exists" in capsys.readouterr().err
    assert output_path.read_text(encoding="utf-8") == "keep"
