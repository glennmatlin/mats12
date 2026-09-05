from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import concept_filtering.cli as cli
import concept_filtering.training_smoke as training_smoke
from concept_filtering.cli import main
from concept_filtering.training_recipe import TrainingRecipe, load_training_recipe
from concept_filtering.training_smoke import (
    TrainingSmokeError,
    TrainingSmokeObservation,
    execute_training_smoke,
)

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/training/public-rank32-reconstruction-v1.json"
REVISION = "a" * 40


def _hash(character: str) -> str:
    return character * 64


def _observation() -> TrainingSmokeObservation:
    return TrainingSmokeObservation(
        runtime_versions={
            "huggingface_hub": "0.36.2",
            "peft": "0.18.1",
            "tokenizers": "0.22.2",
            "torch": "2.13.0",
            "transformers": "4.57.6",
        },
        example_count=5,
        retained_example_count=4,
        dropped_overlength_count=1,
        packed_sequence_count=2,
        sample_order_sha256=_hash("1"),
        input_ids_sha256=_hash("2"),
        labels_sha256=_hash("3"),
        batch_input_ids_sha256=(_hash("b"), _hash("c")),
        batch_labels_sha256=(_hash("d"), _hash("e")),
        smoke_fixture_sha256=(
            "ca63d3a0e29f03db1be98f9b935fc46ab67b634e59c352e92ee98ccc0678d44b"
        ),
        chat_template_sha256=(
            "fb2b8736121ab4001112a9ccfb33c0081230536c861c706bf24b1b390964c1cd"
        ),
        trainable_parameter_count=1024,
        optimizer_step_count=2,
        checkpoint_step=1,
        next_batch_index=1,
        uninterrupted_losses=(1.5, 1.4),
        resumed_losses=(1.4,),
        uninterrupted_state_sha256=_hash("4"),
        resumed_state_sha256=_hash("4"),
        max_abs_parameter_difference=0.0,
        checkpoint_files={
            "adapter_config.json": _hash("5"),
            "adapter_model.safetensors": _hash("6"),
            "optimizer.pt": _hash("7"),
            "rng.pt": _hash("8"),
            "scheduler.pt": _hash("9"),
            "state.json": _hash("a"),
        },
    )


def _accept_revision(expected: str) -> str:
    return expected


def _fake_runtime(_recipe: TrainingRecipe, _run_dir: Path) -> TrainingSmokeObservation:
    return _observation()


def test_execute_training_smoke_writes_release_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_training_recipe(RECIPE)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    output = tmp_path / "receipt.json"

    receipt = execute_training_smoke(
        recipe,
        code_revision=REVISION,
        output_path=output,
        run=lambda _: _observation(),
    )

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == receipt.to_dict()
    assert stored["scientific_use"] is False
    assert stored["scientific_training_allowed"] is False
    assert stored["bounded_smoke_execution"] is True
    assert stored["releases"]["executor_release"] == REVISION
    assert stored["releases"]["coordinator_release"] == REVISION
    assert stored["releases"]["input_producer_releases"] == (
        recipe.input_producer_releases
    )
    assert stored["releases"]["output_release"] not in {
        REVISION,
        *recipe.input_producer_releases.values(),
    }
    assert stored["execution"]["max_abs_parameter_difference"] == 0.0
    assert stored["execution"]["next_batch_index"] == 1
    assert stored["contract_sha256"] == recipe.contract_sha256
    assert stored["recipe_sha256"] == recipe.recipe_sha256
    assert stored["receipt_sha256"] == receipt.receipt_sha256


def test_relative_config_path_revalidates_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path("configs/training/public-rank32-reconstruction-v1.json")
    recipe = load_training_recipe(relative)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)

    receipt = execute_training_smoke(
        recipe,
        code_revision=REVISION,
        output_path=tmp_path / "receipt.json",
        run=lambda _: _observation(),
    )

    assert receipt.payload["recipe_sha256"] == recipe.recipe_sha256


def test_rejects_post_resume_difference_above_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_training_recipe(RECIPE)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    observation = _observation()
    changed = replace(
        observation,
        max_abs_parameter_difference=1e-4,
        resumed_state_sha256=_hash("b"),
    )
    output = tmp_path / "receipt.json"

    with pytest.raises(TrainingSmokeError, match="tolerance"):
        execute_training_smoke(
            recipe,
            code_revision=REVISION,
            output_path=output,
            run=lambda _: changed,
        )

    assert not output.exists()


def test_rejects_runtime_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_training_recipe(RECIPE)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    observation = _observation()
    changed = replace(
        observation,
        runtime_versions={**observation.runtime_versions, "torch": "latest"},
    )

    with pytest.raises(TrainingSmokeError, match="runtime versions"):
        execute_training_smoke(
            recipe,
            code_revision=REVISION,
            output_path=tmp_path / "receipt.json",
            run=lambda _: changed,
        )


def test_rejects_authority_laundering_before_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = replace(load_training_recipe(RECIPE), scientific_training_allowed=True)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    called = False

    def run(_: TrainingRecipe) -> TrainingSmokeObservation:
        nonlocal called
        called = True
        return _observation()

    with pytest.raises(TrainingSmokeError, match="authority"):
        execute_training_smoke(
            recipe,
            code_revision=REVISION,
            output_path=tmp_path / "receipt.json",
            run=run,
        )

    assert called is False


@pytest.mark.parametrize("mutation", ["rank", "runtime", "release", "override"])
def test_rejects_recipe_identity_laundering_before_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    recipe = load_training_recipe(RECIPE)
    if mutation == "rank":
        recipe = replace(recipe, rank=64)
    elif mutation == "runtime":
        recipe.runtime_versions["torch"] = "latest"
    elif mutation == "release":
        recipe.input_producer_releases["public_recipe_evidence"] = "f" * 40
    else:
        recipe.smoke_overrides["world_size"] = 2
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    called = False

    def run(_: TrainingRecipe) -> TrainingSmokeObservation:
        nonlocal called
        called = True
        return _observation()

    with pytest.raises(TrainingSmokeError, match="canonical config"):
        execute_training_smoke(
            recipe,
            code_revision=REVISION,
            output_path=tmp_path / "receipt.json",
            run=run,
        )

    assert called is False


def test_rejects_restarted_data_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_training_recipe(RECIPE)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    changed = replace(_observation(), next_batch_index=0)

    with pytest.raises(TrainingSmokeError, match="step 1"):
        execute_training_smoke(
            recipe,
            code_revision=REVISION,
            output_path=tmp_path / "receipt.json",
            run=lambda _: changed,
        )


def test_rejects_missing_checkpoint_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_training_recipe(RECIPE)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    observation = _observation()
    files = dict(observation.checkpoint_files)
    del files["rng.pt"]
    changed = replace(observation, checkpoint_files=files)

    with pytest.raises(TrainingSmokeError, match="checkpoint files"):
        execute_training_smoke(
            recipe,
            code_revision=REVISION,
            output_path=tmp_path / "receipt.json",
            run=lambda _: changed,
        )


def test_rejects_nonfinite_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_training_recipe(RECIPE)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    observation = _observation()
    changed = replace(observation, uninterrupted_losses=(1.5, float("nan")))

    with pytest.raises(TrainingSmokeError, match="loss"):
        execute_training_smoke(
            recipe,
            code_revision=REVISION,
            output_path=tmp_path / "receipt.json",
            run=lambda _: changed,
        )


def test_refuses_existing_output_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_training_recipe(RECIPE)
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    output = tmp_path / "receipt.json"
    output.write_text("keep\n", encoding="utf-8")
    called = False

    def run(_: object) -> TrainingSmokeObservation:
        nonlocal called
        called = True
        return _observation()

    with pytest.raises(TrainingSmokeError, match="overwrite"):
        execute_training_smoke(
            recipe,
            code_revision=REVISION,
            output_path=output,
            run=run,
        )

    assert called is False
    assert output.read_text(encoding="utf-8") == "keep\n"


def test_cli_runs_bounded_training_smoke(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_smoke, "verify_executor_revision", _accept_revision)
    monkeypatch.setattr(
        cli,
        "run_transformers_training_smoke",
        _fake_runtime,
    )
    output = tmp_path / "receipt.json"

    exit_code = main(
        [
            "run-training-smoke",
            str(RECIPE),
            "--code-revision",
            REVISION,
            "--run-dir",
            str(tmp_path / "run"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["bounded_smoke_execution"] is True
    assert summary["scientific_training_allowed"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["receipt_sha256"]
