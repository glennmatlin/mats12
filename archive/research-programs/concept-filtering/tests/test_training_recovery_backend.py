from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import concept_filtering.training_recovery_transformers as backend
from concept_filtering.named_training_recipe import (
    NamedTrainingRecipe,
    load_named_training_recipe,
)
from concept_filtering.serialization import canonical_sha256
from concept_filtering.training_recovery_smoke import (
    PackedBatch,
    PackedSmokeData,
    TrainingRecoverySmokeError,
    TrainingTokenizer,
)

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/reconstruction/training/public-rank32-named-recipe-v1.json"
TORCH = pytest.importorskip("torch")
SAFE = pytest.importorskip("safetensors.torch")
TRANSFORMERS = pytest.importorskip("transformers")
pytest.importorskip("peft")
COMPONENTS = vars(backend)["_Components"]


class _TinyModel:
    def __init__(self) -> None:
        self.weight = TORCH.nn.Parameter(TORCH.tensor([1.0]))

    def parameters(self) -> list[Any]:
        return [self.weight]

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        noise = TORCH.nn.functional.dropout(
            TORCH.ones_like(input_ids, dtype=TORCH.float32),
            p=0.5,
            training=True,
        ).mean()
        target = labels[labels.ne(-100)].float().mean() / 100.0
        return SimpleNamespace(loss=(self.weight * noise - target).square().mean())


class _Peft:
    @staticmethod
    def get_peft_model_state_dict(model: _TinyModel) -> dict[str, Any]:
        return {"weight": model.weight.detach().clone()}

    @staticmethod
    def set_peft_model_state_dict(
        model: _TinyModel, state: dict[str, Any]
    ) -> SimpleNamespace:
        with TORCH.no_grad():
            model.weight.copy_(state["weight"])
        return SimpleNamespace(unexpected_keys=[])


def _data() -> PackedSmokeData:
    recipe = load_named_training_recipe(RECIPE)
    values = tuple(range(1, 129))
    batches = (
        PackedBatch(values, values, (1,) * 128),
        PackedBatch(tuple(reversed(values)), tuple(reversed(values)), (1,) * 128),
    )
    hashes = _batch_hashes(batches)
    return PackedSmokeData(
        fixture_sha256=recipe.smoke_fixture.sha256,
        source_sample_receipt_sha256=(
            recipe.smoke_fixture.source_sample_receipt_sha256
        ),
        input_example_count=4,
        eligible_example_count=3,
        dropped_overlength_example_ids=("3" * 64,),
        selected_example_ids=("4" * 64, "5" * 64),
        token_stream_sha256=canonical_sha256(list(values) + list(reversed(values))),
        packed_batches_sha256=canonical_sha256(list(hashes)),
        packed_batch_sha256=hashes,
        rendered_token_count=256,
        assistant_label_token_count=256,
        assistant_turn_count=2,
        eos_label_count=2,
        batches=batches,
    )


def _batch_hashes(batches: tuple[PackedBatch, ...]) -> tuple[str, ...]:
    return tuple(
        canonical_sha256(
            {
                "input_ids": list(batch.input_ids),
                "labels": list(batch.labels),
                "attention_mask": list(batch.attention_mask),
            }
        )
        for batch in batches
    )


def _components(
    torch: Any,
    transformers: Any,
    peft: Any,
    recipe: NamedTrainingRecipe,
    base_snapshot: Path,
) -> Any:
    del transformers, peft, base_snapshot
    torch.manual_seed(recipe.values.training.seed)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _constant_rate)
    torch.manual_seed(recipe.values.training.seed)
    return COMPONENTS(model, optimizer, scheduler)


def _constant_rate(step: int) -> float:
    del step
    return 1.0


def _module(name: str) -> Any:
    modules: dict[str, Any] = {
        "torch": TORCH,
        "transformers": object(),
        "peft": _Peft(),
        "safetensors.torch": SAFE,
    }
    return modules[name]


def _verify_files(root: Path, expected: dict[str, str], label: str) -> None:
    del root, expected, label


def _ignore_base(recipe: NamedTrainingRecipe, root: Path) -> None:
    del recipe, root


def _tokenizer(
    transformers: Any, recipe: NamedTrainingRecipe, snapshot: Path
) -> TrainingTokenizer:
    del transformers, recipe, snapshot
    return cast(TrainingTokenizer, object())


def _prepare_data(
    recipe: NamedTrainingRecipe, tokenizer: TrainingTokenizer, fixture: Path
) -> PackedSmokeData:
    del recipe, tokenizer, fixture
    return _data()


def _runtime_versions() -> dict[str, str]:
    return dict(load_named_training_recipe(RECIPE).runtime_versions)


def _tiny_base(path: Path) -> None:
    config = TRANSFORMERS.Olmo3Config(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sliding_window=64,
        layer_types=["full_attention"],
    )
    TRANSFORMERS.Olmo3ForCausalLM(config).save_pretrained(path)


def _patch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "_module", _module)
    monkeypatch.setattr(backend, "_runtime_versions", _runtime_versions)
    monkeypatch.setattr(backend, "_verify_files", _verify_files)
    monkeypatch.setattr(backend, "_tokenizer", _tokenizer)
    monkeypatch.setattr(backend, "_prepare_data", _prepare_data)
    monkeypatch.setattr(backend, "_components", _components)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    recipe = load_named_training_recipe(RECIPE)
    return _run_recipe(recipe, tmp_path, monkeypatch)


def _run_recipe(
    recipe: NamedTrainingRecipe,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    _patch_backend(monkeypatch)
    return backend.run_transformers_training_recovery(
        recipe,
        base_snapshot=tmp_path / "base",
        tokenizer_snapshot=tmp_path / "tokenizer",
        fixture_path=tmp_path / "fixture.json",
        work_dir=tmp_path / "work",
    )


def _semantic_mutations(
    recipe: NamedTrainingRecipe,
) -> tuple[NamedTrainingRecipe, ...]:
    values = recipe.values
    return (
        replace(
            recipe,
            values=replace(values, order=replace(values.order, algorithm="other")),
        ),
        replace(
            recipe,
            values=replace(values, sequence=replace(values.sequence, packing="other")),
        ),
        replace(
            recipe,
            values=replace(
                values, sequence=replace(values.sequence, loss_mask="other")
            ),
        ),
        replace(
            recipe,
            values=replace(
                values, checkpoint=replace(values.checkpoint, boundary="other")
            ),
        ),
    )


def test_rejects_recipe_semantics_not_implemented_by_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_named_training_recipe(RECIPE)

    for mutation in _semantic_mutations(recipe):
        with pytest.raises(TrainingRecoverySmokeError, match="unsupported"):
            _run_recipe(mutation, tmp_path, monkeypatch)


def test_checkpoint_round_trip_restores_optimizer_scheduler_rng_and_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _run(tmp_path, monkeypatch)

    assert (
        observation.uninterrupted_adapter_sha256 == observation.resumed_adapter_sha256
    )
    assert observation.max_abs_difference == 0.0
    assert observation.uninterrupted_losses == observation.checkpoint_resume_losses
    assert set(observation.checkpoint.file_sha256) == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "cursor.json",
        "optimizer.pt",
        "rng.pt",
        "scheduler.pt",
        "trainer_state.json",
    }


def test_tiny_real_olmo_lora_checkpoint_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "tiny-base"
    _tiny_base(base)
    recipe = load_named_training_recipe(RECIPE)
    monkeypatch.setattr(backend, "_verify_base_snapshot", _ignore_base)
    monkeypatch.setattr(backend, "_runtime_versions", _runtime_versions)
    monkeypatch.setattr(backend, "_tokenizer", _tokenizer)
    monkeypatch.setattr(backend, "_prepare_data", _prepare_data)

    observation = backend.run_transformers_training_recovery(
        recipe,
        base_snapshot=base,
        tokenizer_snapshot=tmp_path / "tokenizer",
        fixture_path=tmp_path / "fixture.json",
        work_dir=tmp_path / "work",
    )

    assert observation.max_abs_difference == 0.0
    assert (
        observation.uninterrupted_adapter_sha256 == observation.resumed_adapter_sha256
    )


def test_rejects_tampered_checkpoint_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = cast(
        Callable[[NamedTrainingRecipe, PackedSmokeData, Path, Any], None],
        vars(backend)["_verify_checkpoint"],
    )

    def tamper(
        recipe: NamedTrainingRecipe,
        data: PackedSmokeData,
        path: Path,
        checkpoint: Any,
    ) -> None:
        (path / "cursor.json").write_text("{}\n", encoding="utf-8")
        original(recipe, data, path, checkpoint)

    _patch_backend(monkeypatch)
    monkeypatch.setattr(backend, "_verify_checkpoint", tamper)
    recipe = load_named_training_recipe(RECIPE)
    with pytest.raises(TrainingRecoverySmokeError, match="hashes differ"):
        backend.run_transformers_training_recovery(
            recipe,
            base_snapshot=tmp_path / "base",
            tokenizer_snapshot=tmp_path / "tokenizer",
            fixture_path=tmp_path / "fixture.json",
            work_dir=tmp_path / "work",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_terminal_adapter_tensors(value: float) -> None:
    left = {"weight": TORCH.tensor([value])}
    right = {"weight": TORCH.tensor([0.0])}

    with pytest.raises(TrainingRecoverySmokeError, match="finite"):
        maximum_difference = cast(
            Callable[[Any, Any, Any], float], vars(backend)["_max_abs_difference"]
        )
        maximum_difference(TORCH, left, right)


def test_rejects_existing_work_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "work").mkdir()

    with pytest.raises(TrainingRecoverySmokeError, match="work directory"):
        _run(tmp_path, monkeypatch)


def test_rejects_existing_checkpoint_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = vars(backend)["_recovery_path"]

    def collide(
        modules: tuple[Any, Any, Any, Any],
        recipe: NamedTrainingRecipe,
        base_snapshot: Path,
        data: PackedSmokeData,
        checkpoint_dir: Path,
    ) -> Any:
        checkpoint_dir.mkdir()
        return original(modules, recipe, base_snapshot, data, checkpoint_dir)

    _patch_backend(monkeypatch)
    monkeypatch.setattr(backend, "_recovery_path", collide)
    recipe = load_named_training_recipe(RECIPE)
    with pytest.raises(TrainingRecoverySmokeError, match="checkpoint directory"):
        backend.run_transformers_training_recovery(
            recipe,
            base_snapshot=tmp_path / "base",
            tokenizer_snapshot=tmp_path / "tokenizer",
            fixture_path=tmp_path / "fixture.json",
            work_dir=tmp_path / "work",
        )


def test_rejects_adapter_markers_in_base_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    recipe = load_named_training_recipe(RECIPE)
    _patch_backend(monkeypatch)

    with pytest.raises(TrainingRecoverySmokeError, match="adapter marker"):
        backend.run_transformers_training_recovery(
            recipe,
            base_snapshot=base,
            tokenizer_snapshot=tmp_path / "tokenizer",
            fixture_path=tmp_path / "fixture.json",
            work_dir=tmp_path / "work",
        )


@pytest.mark.parametrize("attribute", ["_hf_peft_config_loaded", "peft_config"])
def test_rejects_loaded_base_with_preexisting_adapter(attribute: str) -> None:
    model = SimpleNamespace(**{attribute: True})
    validator = vars(backend)["_validate_fresh_base"]

    with pytest.raises(TrainingRecoverySmokeError, match="preexisting adapter"):
        validator(model)
