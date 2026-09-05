from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import concept_filtering.training_smoke_transformers as training_backend
from concept_filtering.chat_rendering import render_assistant_only
from concept_filtering.training_recipe import load_training_recipe
from concept_filtering.training_smoke import TrainingSmokeError
from concept_filtering.training_smoke_transformers import (
    run_transformers_training_smoke,
)

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/training/public-rank32-reconstruction-v1.json"
HAS_RUNTIME = all(
    importlib.util.find_spec(name) is not None
    for name in ("peft", "tokenizers", "torch", "transformers")
)


def _subsequence_start(values: tuple[int, ...], target: list[int]) -> int:
    for index in range(len(values) - len(target) + 1):
        if list(values[index : index + len(target)]) == target:
            return index
    raise AssertionError("token subsequence was not found")


def _real_smoke_tokenizer(payload: Any) -> Any:
    tokenizers = importlib.import_module("tokenizers")
    transformers = importlib.import_module("transformers")
    words = {"<unk>", "<bos>", "<eos>", "<pad>", "<|im_start|>", "<|im_end|>"}
    for example in payload["examples"]:
        for message in example["messages"]:
            words.update(message["content"].split())
    words.add(payload["overlength_probe"]["repeat_token"])
    vocabulary = {token: index for index, token in enumerate(sorted(words))}
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocabulary, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )


@pytest.mark.skipif(not HAS_RUNTIME, reason="inference extra is not installed")
def test_real_tokenizer_masks_the_intervening_user_turn() -> None:
    fixture: Any = json.loads(
        (
            ROOT / "configs/training/fixtures/public-rank32-training-smoke-v1.json"
        ).read_text()
    )
    tokenizer = _real_smoke_tokenizer(fixture)
    template = fixture["chat_template"]
    tokenizer.chat_template = template
    messages = fixture["examples"][1]["messages"]

    rendered = render_assistant_only(tokenizer, messages, template)
    user_ids = tokenizer(
        "<|im_start|>user\nSay goodbye<|im_end|>\n", add_special_tokens=False
    )["input_ids"]
    start = _subsequence_start(rendered.input_ids, user_ids)

    assert rendered.assistant_turn_count == 2
    assert all(
        label == -100 for label in rendered.labels[start : start + len(user_ids)]
    )


@pytest.mark.skipif(not HAS_RUNTIME, reason="inference extra is not installed")
def test_real_tiny_olmo3_lora_checkpoint_resume(tmp_path: Path) -> None:
    recipe = load_training_recipe(RECIPE)

    observation = run_transformers_training_smoke(recipe, tmp_path / "run")
    replay = run_transformers_training_smoke(recipe, tmp_path / "replay")

    assert replay == observation
    assert observation.example_count == 5
    assert observation.retained_example_count == 4
    assert observation.dropped_overlength_count == 1
    assert observation.packed_sequence_count == 2
    assert len(set(observation.batch_input_ids_sha256)) == 2
    assert observation.optimizer_step_count == 2
    assert observation.checkpoint_step == 1
    assert observation.next_batch_index == 1
    assert observation.trainable_parameter_count > 0
    assert observation.max_abs_parameter_difference <= (
        recipe.post_resume_absolute_tolerance
    )
    assert observation.uninterrupted_state_sha256 == (observation.resumed_state_sha256)
    assert set(observation.checkpoint_files) == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng.pt",
        "scheduler.pt",
        "state.json",
    }
    state = json.loads(
        (tmp_path / "run/checkpoint-1/state.json").read_text(encoding="utf-8")
    )
    assert state["next_batch_index"] == 1
    adapter_config = json.loads(
        (tmp_path / "run/checkpoint-1/adapter_config.json").read_text(encoding="utf-8")
    )
    assert adapter_config["target_modules"] == sorted(recipe.target_modules)


@pytest.mark.skipif(not HAS_RUNTIME, reason="inference extra is not installed")
def test_restarted_cursor_changes_the_final_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_training_recipe(RECIPE)

    def restart_cursor(_recipe: object, _dataset: object, _path: Path) -> int:
        return 0

    monkeypatch.setattr(training_backend, "_validate_checkpoint_state", restart_cursor)

    observation = run_transformers_training_smoke(recipe, tmp_path / "run")

    assert observation.max_abs_parameter_difference > (
        recipe.post_resume_absolute_tolerance
    )


@pytest.mark.skipif(not HAS_RUNTIME, reason="inference extra is not installed")
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("invalid_message", "fixture message"),
        ("missing_multi_turn", "one multi-turn"),
        ("missing_empty_user", "one multi-turn"),
    ],
)
def test_malformed_fixture_fails_before_run_directory(
    tmp_path: Path, mutation: str, message: str
) -> None:
    payload = json.loads(RECIPE.read_text(encoding="utf-8"))
    source_fixture = (
        ROOT / "configs/training/fixtures/public-rank32-training-smoke-v1.json"
    )
    fixture = json.loads(source_fixture.read_text(encoding="utf-8"))
    if mutation == "invalid_message":
        fixture["examples"][0]["messages"][0] = 7
    elif mutation == "missing_multi_turn":
        fixture["examples"][1]["messages"] = fixture["examples"][0]["messages"]
    else:
        fixture["examples"][2]["messages"][0]["content"] = "not empty"
    fixture_path = tmp_path / "fixtures/public-rank32-training-smoke-v1.json"
    fixture_path.parent.mkdir()
    fixture_bytes = (json.dumps(fixture, indent=2) + "\n").encode()
    fixture_path.write_bytes(fixture_bytes)
    payload["execution"]["smoke_fixture"]["sha256"] = hashlib.sha256(
        fixture_bytes
    ).hexdigest()
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    recipe = load_training_recipe(config_path)
    run_dir = tmp_path / "run"

    with pytest.raises(TrainingSmokeError, match=message):
        run_transformers_training_smoke(recipe, run_dir)

    assert not run_dir.exists()
