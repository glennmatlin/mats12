from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import concept_filtering.cli as cli
import concept_filtering.training_recovery_smoke as recovery
from concept_filtering.named_training_recipe import (
    NamedTrainingRecipe,
    SmokePreprocessingExpectation,
    load_named_training_recipe,
)
from concept_filtering.serialization import canonical_sha256
from concept_filtering.training_recovery_smoke import (
    CheckpointObservation,
    ConversationMessage,
    PackedBatch,
    PackedSmokeData,
    TrainingRecoveryObservation,
    TrainingRow,
    build_packed_smoke_data,
    execute_training_recovery_smoke,
)

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/reconstruction/training/public-rank32-named-recipe-v1.json"
FIXTURE = (
    ROOT
    / "configs"
    / "reconstruction"
    / "training"
    / "fixtures"
    / "public-rank32-recovery-smoke-fixture-v1.json"
)


def _identity(value: str) -> str:
    return value


def _runtime_versions() -> dict[str, str]:
    return {
        "peft": "0.18.1",
        "pyarrow": "23.0.1",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
        "torch": "2.13.0",
        "transformers": "4.57.6",
    }


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 999
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        rendered = "|".join(f"{item['role']}:{item['content']}" for item in messages)
        return f"{rendered}{self.eos_token}"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        eos_start = len(text) - len(self.eos_token)
        input_ids = [ord(character) for character in text[:eos_start]]
        input_ids.append(self.eos_token_id)
        offsets = [(index, index + 1) for index in range(eos_start)]
        offsets.append((eos_start, len(text)))
        return {"input_ids": input_ids, "offset_mapping": offsets}


class _MultiTurnTokenizer(_Tokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        parts = [
            f"{item['role']}:{item['content']}"
            + (self.eos_token if item["role"] == "assistant" else "|")
            for item in messages
        ]
        return "".join(parts)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            if text.startswith(self.eos_token, index):
                input_ids.append(self.eos_token_id)
                offsets.append((index, index + len(self.eos_token)))
                index += len(self.eos_token)
            else:
                input_ids.append(ord(text[index]))
                offsets.append((index, index + 1))
                index += 1
        return {"input_ids": input_ids, "offset_mapping": offsets}


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


def _packed_data() -> PackedSmokeData:
    recipe = load_named_training_recipe(RECIPE)
    batches = (
        PackedBatch((1,) * 128, (-100,) * 108 + (1,) * 20, (1,) * 128),
        PackedBatch(
            (2,) * 37 + (0,) * 91,
            (-100,) * 16 + (2,) * 21 + (-100,) * 91,
            (1,) * 37 + (0,) * 91,
        ),
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
        token_stream_sha256=canonical_sha256([1] * 128 + [2] * 37),
        packed_batches_sha256=canonical_sha256(list(hashes)),
        packed_batch_sha256=hashes,
        rendered_token_count=165,
        assistant_label_token_count=41,
        assistant_turn_count=2,
        eos_label_count=2,
        batches=batches,
    )


def _observation() -> TrainingRecoveryObservation:
    return TrainingRecoveryObservation(
        data=_packed_data(),
        consumed_packed_sequence_count=2,
        checkpoint=CheckpointObservation(
            optimizer_step=1,
            data_cursor=1,
            file_sha256={
                "adapter_config.json": "e" * 64,
                "adapter_model.safetensors": "e" * 64,
                "cursor.json": "e" * 64,
                "optimizer.pt": "e" * 64,
                "rng.pt": "e" * 64,
                "scheduler.pt": "e" * 64,
                "trainer_state.json": "e" * 64,
            },
        ),
        uninterrupted_adapter_sha256="f" * 64,
        resumed_adapter_sha256="f" * 64,
        max_abs_difference=0.0,
        uninterrupted_losses=(1.0, 0.5),
        checkpoint_resume_losses=(1.0, 0.5),
        device="cpu",
        model_dtype="bfloat16",
        runtime_versions=_runtime_versions(),
    )


def _recipe_for_data(data: PackedSmokeData) -> NamedTrainingRecipe:
    recipe = load_named_training_recipe(RECIPE)
    expected = SmokePreprocessingExpectation(
        input_example_count=data.input_example_count,
        eligible_example_count=data.eligible_example_count,
        dropped_overlength_example_ids=data.dropped_overlength_example_ids,
        selected_example_ids=data.selected_example_ids,
        token_stream_sha256=data.token_stream_sha256,
        packed_batches_sha256=data.packed_batches_sha256,
        packed_batch_sha256=data.packed_batch_sha256,
        rendered_token_count=data.rendered_token_count,
        assistant_label_token_count=data.assistant_label_token_count,
        assistant_turn_count=data.assistant_turn_count,
        eos_label_count=data.eos_label_count,
    )
    return replace(recipe, smoke=replace(recipe.smoke, expected_preprocessing=expected))


def _rehash_data(
    data: PackedSmokeData, batches: tuple[PackedBatch, ...]
) -> PackedSmokeData:
    hashes = _batch_hashes(batches)
    tokens = [
        token
        for batch in batches
        for token, attention in zip(batch.input_ids, batch.attention_mask, strict=True)
        if attention == 1
    ]
    labels = [label for batch in batches for label in batch.labels]
    return replace(
        data,
        token_stream_sha256=canonical_sha256(tokens),
        packed_batches_sha256=canonical_sha256(list(hashes)),
        packed_batch_sha256=hashes,
        rendered_token_count=len(tokens),
        assistant_label_token_count=sum(label != -100 for label in labels),
        batches=batches,
    )


def _mutate_first_batch(
    data: PackedSmokeData, *, label: int | None = None, attention: int | None = None
) -> PackedSmokeData:
    first = data.batches[0]
    labels = list(first.labels)
    attention_mask = list(first.attention_mask)
    if label is not None:
        labels[0] = label
    if attention is not None:
        attention_mask[0] = attention
    batch = replace(first, labels=tuple(labels), attention_mask=tuple(attention_mask))
    return _rehash_data(data, (batch, *data.batches[1:]))


def _replace_terminal_supervised_token(data: PackedSmokeData) -> PackedSmokeData:
    last = data.batches[-1]
    index = max(index for index, label in enumerate(last.labels) if label != -100)
    input_ids = list(last.input_ids)
    labels = list(last.labels)
    input_ids[index] += 1
    labels[index] += 1
    batch = replace(last, input_ids=tuple(input_ids), labels=tuple(labels))
    return _rehash_data(data, (*data.batches[:-1], batch))


def _replace_dropped_id(data: PackedSmokeData) -> PackedSmokeData:
    return replace(data, dropped_overlength_example_ids=("6" * 64,))


def _unmask_prompt_token(data: PackedSmokeData) -> PackedSmokeData:
    return _mutate_first_batch(data, label=1)


def _hide_prompt_attention(data: PackedSmokeData) -> PackedSmokeData:
    return _mutate_first_batch(data, attention=0)


def _swap_batches(data: PackedSmokeData) -> PackedSmokeData:
    return _rehash_data(data, tuple(reversed(data.batches)))


def _cli_command(tmp_path: Path, output: Path) -> list[str]:
    return [
        "run-training-recovery-smoke",
        str(RECIPE),
        "--code-revision",
        "a" * 40,
        "--base-snapshot",
        str(tmp_path / "base"),
        "--tokenizer-snapshot",
        str(tmp_path / "tokenizer"),
        "--fixture",
        str(tmp_path / "fixture.json"),
        "--work-dir",
        str(tmp_path / "work"),
        "--output",
        str(output),
    ]


def _patch_cli_recipe(
    monkeypatch: pytest.MonkeyPatch, recipe: NamedTrainingRecipe
) -> None:
    def load_recipe(_path: Path) -> NamedTrainingRecipe:
        return recipe

    monkeypatch.setattr(cli, "load_named_training_recipe", load_recipe)


def _row(index: int, prompt: str, response: str) -> TrainingRow:
    return TrainingRow(
        index,
        (
            ConversationMessage("user", prompt),
            ConversationMessage("assistant", response),
        ),
    )


def _mask_rows() -> tuple[TrainingRow, ...]:
    return (
        _row(9, "prompt", "response one"),
        _row(3, "question", "response two"),
        _row(7, "oversized", "x" * 80),
    )


def _build_mask_data(rows: tuple[TrainingRow, ...]) -> PackedSmokeData:
    return build_packed_smoke_data(
        rows,
        _Tokenizer(),
        source_revision="a" * 40,
        seed=42,
        sample_limit=2,
        sequence_length=64,
        overlength_limit=50,
        fixture_sha256="1" * 64,
        source_sample_receipt_sha256="2" * 64,
    )


def _assert_mask_contract(data: PackedSmokeData) -> None:
    assert len(data.selected_example_ids) == 2
    assert len(data.dropped_overlength_example_ids) == 1
    assert data.eligible_example_count == 2
    assert data.assistant_label_token_count > data.eos_label_count
    assert data.assistant_turn_count == 2
    assert data.eos_label_count == 2
    assert all(len(batch.input_ids) == 64 for batch in data.batches)
    assert any(
        label == -100 and attention == 1
        for batch in data.batches
        for label, attention in zip(batch.labels, batch.attention_mask, strict=True)
    )
    assert any(999 in batch.labels for batch in data.batches)
    assert all(
        any(label != -100 for label in batch.labels) for batch in data.batches[:2]
    )


def test_builds_stable_order_assistant_mask_and_drops_overlength() -> None:
    rows = _mask_rows()
    first = _build_mask_data(rows)
    second = _build_mask_data(tuple(reversed(rows)))

    assert first == second
    _assert_mask_contract(first)


def test_frozen_fixture_exercises_observed_conversation_shapes() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assistant_counts = [
        sum(message["role"] == "assistant" for message in row["messages"])
        for row in rows
    ]
    empty_user = any(
        message == {"role": "user", "content": ""}
        for row in rows
        for message in row["messages"]
    )

    assert max(assistant_counts) >= 2
    assert empty_user


def _multiturn_row() -> TrainingRow:
    return TrainingRow(
        1,
        (
            ConversationMessage("user", "opening question"),
            ConversationMessage("assistant", "first answer"),
            ConversationMessage("user", "follow up"),
            ConversationMessage("assistant", "second answer"),
        ),
    )


def test_supervises_every_assistant_turn_in_multiturn_conversation() -> None:
    data = build_packed_smoke_data(
        (_multiturn_row(),),
        _MultiTurnTokenizer(),
        source_revision="a" * 40,
        seed=42,
        sample_limit=1,
        sequence_length=48,
        overlength_limit=128,
        fixture_sha256="1" * 64,
        source_sample_receipt_sha256="2" * 64,
    )
    labels = [label for batch in data.batches for label in batch.labels if label >= 0]
    supervised = "".join(chr(label) for label in labels if label != 999)

    assert "first answer" in supervised
    assert "second answer" in supervised
    assert "follow up" not in supervised
    assert data.assistant_turn_count == 2
    assert data.eos_label_count == 1


def test_executes_bounded_recovery_smoke_and_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation()
    recipe = _recipe_for_data(observation.data)
    monkeypatch.setattr(recovery, "verify_executor_revision", _identity)
    output = tmp_path / "receipt.json"

    receipt = execute_training_recovery_smoke(
        recipe,
        code_revision="a" * 40,
        output_path=output,
        run=lambda value: observation,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == receipt.to_dict()
    assert payload["recipe"]["core_sha256"] == recipe.recipe_core_sha256
    assert payload["authority"]["sha256"] == recipe.smoke_authority_sha256
    assert payload["data"]["fixture_sha256"] == recipe.smoke_fixture.sha256
    assert payload["data"]["eos_label_count"] == 2
    assert payload["checkpoint"]["optimizer_step"] == 1
    assert payload["comparison"]["passed"] is True
    assert not any(payload["authority"]["values"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        _replace_dropped_id,
        _unmask_prompt_token,
        _hide_prompt_attention,
        _replace_terminal_supervised_token,
        _swap_batches,
    ],
)
def test_rejects_coherently_rehashed_preprocessing_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[PackedSmokeData], PackedSmokeData],
) -> None:
    baseline = _observation()
    recipe = _recipe_for_data(baseline.data)
    changed_data = mutation(baseline.data)
    changed = replace(baseline, data=changed_data)
    monkeypatch.setattr(recovery, "verify_executor_revision", _identity)

    with pytest.raises(
        recovery.TrainingRecoverySmokeError, match=r"frozen|assistant turn"
    ):
        execute_training_recovery_smoke(
            recipe,
            code_revision="a" * 40,
            output_path=tmp_path / "receipt.json",
            run=lambda value: changed,
        )


def test_rejects_recovery_observation_outside_declared_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation()
    recipe = _recipe_for_data(observation.data)
    monkeypatch.setattr(recovery, "verify_executor_revision", _identity)
    failed = replace(
        observation,
        max_abs_difference=recipe.smoke.comparison.absolute_tolerance * 2,
    )

    with pytest.raises(recovery.TrainingRecoverySmokeError, match="tolerance"):
        execute_training_recovery_smoke(
            recipe,
            code_revision="a" * 40,
            output_path=tmp_path / "receipt.json",
            run=lambda value: failed,
        )


def test_cli_runs_transformers_recovery_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation()
    recipe = _recipe_for_data(observation.data)
    monkeypatch.setattr(recovery, "verify_executor_revision", _identity)
    _patch_cli_recipe(monkeypatch, recipe)

    def fake_run(
        recipe: NamedTrainingRecipe,
        *,
        base_snapshot: Path,
        tokenizer_snapshot: Path,
        fixture_path: Path,
        work_dir: Path,
    ) -> TrainingRecoveryObservation:
        assert all((base_snapshot, tokenizer_snapshot, fixture_path, work_dir))
        return observation

    monkeypatch.setattr(cli, "run_transformers_training_recovery", fake_run)
    output = tmp_path / "receipt.json"
    exit_code = cli.main(_cli_command(tmp_path, output))

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["receipt_sha256"]
    assert output.exists()
