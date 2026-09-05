from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from concept_filtering.generation_bank import load_generation_plan
from concept_filtering.prompt_bank import PromptRecord
from concept_filtering.smoke import SmokeValidationError, load_smoke_plan
from concept_filtering.smoke_transformers import (
    download_smoke_snapshots,
    generate_prompt,
    load_smoke_model,
)

ROOT = Path(__file__).parents[1]


class FakeHub:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, object]] = []

    def snapshot_download(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return str(self.root / f"snapshot-{len(self.calls)}")


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(path: str, *, local_files_only: bool) -> object:
        del path, local_files_only
        return SimpleNamespace(chat_template="different template")


class FakeTransformers:
    AutoTokenizer = FakeAutoTokenizer


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def to(self, device: str) -> FakeTensor:
        del device
        return self

    def __getitem__(self, key: object) -> FakeTensor:
        del key
        return FakeTensor((3,))


class FakeGenerationTokenizer:
    eos_token_id = 100257
    pad_token_id = 100277

    def apply_chat_template(self, *_: object, **__: object) -> FakeTensor:
        return FakeTensor((1, 4))

    def decode(self, *_: object, **__: object) -> str:
        return "completion"


class FakeGenerationModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def generate(self, _: FakeTensor, **kwargs: object) -> FakeTensor:
        self.kwargs = kwargs
        return FakeTensor((1, 7))


class FakeTorch:
    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


def test_download_snapshots_pins_revisions_and_adapter_directory(
    tmp_path: Path,
) -> None:
    plan = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    hub = FakeHub(tmp_path)

    download_smoke_snapshots(hub, plan, tmp_path / "cache")

    assert hub.calls[0]["repo_id"] == plan.base.repository
    assert hub.calls[0]["revision"] == plan.base.revision
    assert hub.calls[1]["repo_id"] == plan.adapter.repository
    assert hub.calls[1]["revision"] == plan.adapter.revision
    assert hub.calls[1]["allow_patterns"] == ["split-1/*"]


def test_model_load_rejects_tokenizer_selecting_another_template(
    tmp_path: Path,
) -> None:
    plan = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    adapter_snapshot = tmp_path / "adapter"
    template = adapter_snapshot / plan.template_path
    template.parent.mkdir(parents=True)
    template.write_text("pinned template", encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="pinned template"):
        load_smoke_model(
            FakeTransformers,
            object(),
            object(),
            plan,
            tmp_path / "base",
            adapter_snapshot,
        )


def test_prompt_generation_explicitly_stops_at_tokenizer_eos() -> None:
    smoke = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    plan = load_generation_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-generation.json"
    )
    prompt = PromptRecord(
        question_id="q1",
        behavior="refusal",
        messages=({"role": "user", "content": "Prompt"},),
    )
    tokenizer = FakeGenerationTokenizer()
    model = FakeGenerationModel()

    observation = generate_prompt(
        FakeTorch,
        tokenizer,
        model,
        smoke,
        plan,
        prompt,
    )

    assert model.kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert observation.completion == "completion"
