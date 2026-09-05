from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import concept_filtering.generation_bank as generation_bank
from concept_filtering.generation_bank import (
    GenerationObservation,
    GenerationPlan,
    GenerationRuntime,
    GenerationValidationError,
    execute_generation_bank,
    load_generation_plan,
)
from concept_filtering.prompt_bank import PromptBankExtraction, PromptRecord
from concept_filtering.serialization import canonical_sha256
from concept_filtering.smoke import SmokePlan, load_smoke_plan

ROOT = Path(__file__).parents[1]


def _write_plan(path: Path, smoke_sha256: str, **updates: object) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "generation_id": "rank32-pinned-banks-v1",
        "smoke_plan_sha256": smoke_sha256,
        "max_new_tokens": 1536,
        "do_sample": False,
        "scientific_use": False,
        "starts_experiment_timer": False,
        "protocol_basis": "Reconstructed cap covers ordinary archived outputs.",
    }
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bank(
    behavior: str, count: int = 2
) -> tuple[PromptBankExtraction, tuple[PromptRecord, ...]]:
    body = {
        "schema_version": 1,
        "archive_sha256": "a" * 64,
        "task": "fixture-task",
        "behavior": behavior,
        "question_count": count,
        "source_origin": "fixture",
        "source_commit": "abc1234",
        "source_dirty": True,
        "output_sha256": ("b" if behavior == "refusal" else "c") * 64,
    }
    receipt = PromptBankExtraction(
        schema_version=1,
        archive_sha256="a" * 64,
        task="fixture-task",
        behavior=behavior,
        question_count=count,
        source_origin="fixture",
        source_commit="abc1234",
        source_dirty=True,
        output_sha256=cast(str, body["output_sha256"]),
        receipt_sha256=canonical_sha256(body),
    )
    prompts = tuple(
        PromptRecord(
            question_id=f"{behavior}-{index}",
            behavior=behavior,
            messages=({"role": "user", "content": f"Prompt {index}"},),
        )
        for index in range(count)
    )
    return receipt, prompts


def _observation(prompt: PromptRecord) -> GenerationObservation:
    return GenerationObservation(
        question_id=prompt.question_id,
        behavior=prompt.behavior,
        completion=f"Answer for {prompt.question_id}",
        input_tokens=12,
        generated_tokens=8,
        elapsed_seconds=0.5,
        hit_token_limit=False,
    )


def _runtime(prompts: tuple[PromptRecord, ...]) -> GenerationRuntime:
    return GenerationRuntime(
        adapter_sha256=(
            "f75568de4d2ade8f3f5aaebd5309d9fa94dc6c3ff9a11f62667f3c267a4c251e"
        ),
        template_sha256=(
            "fb2b8736121ab4001112a9ccfb33c0081230536c861c706bf24b1b390964c1cd"
        ),
        runtime_versions={"torch": "2.13.0"},
        observations=(_observation(prompt) for prompt in prompts),
    )


def _plans(tmp_path: Path) -> tuple[SmokePlan, GenerationPlan]:
    smoke = load_smoke_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-smoke.json"
    )
    plan_path = tmp_path / "generation.json"
    _write_plan(plan_path, smoke.plan_sha256)
    return smoke, load_generation_plan(plan_path)


def _verified_revision(revision: str) -> str:
    return revision


def _successful_run(
    smoke: SmokePlan,
    plan: GenerationPlan,
    prompts: tuple[PromptRecord, ...],
) -> GenerationRuntime:
    del smoke, plan
    return _runtime(prompts)


def _unexpected_run(
    smoke: SmokePlan,
    plan: GenerationPlan,
    prompts: tuple[PromptRecord, ...],
) -> GenerationRuntime:
    del smoke, plan, prompts
    pytest.fail("tampered resume must fail first")


def test_public_generation_plan_freezes_reconstructed_cap() -> None:
    plan = load_generation_plan(
        ROOT / "configs/reconstruction/public-rank32-interim-generation.json"
    )

    assert plan.generation_id == "public-rank32-pinned-prompt-banks-v1"
    assert plan.max_new_tokens == 1536
    assert plan.do_sample is False
    assert plan.scientific_use is False
    assert plan.starts_experiment_timer is False


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"scientific_use": True}, "scientific_use"),
        ({"do_sample": True}, "do_sample"),
        ({"max_new_tokens": 9000}, "max_new_tokens"),
    ],
)
def test_generation_plan_rejects_claim_or_protocol_expansion(
    tmp_path: Path, update: dict[str, object], error: str
) -> None:
    path = tmp_path / "generation.json"
    _write_plan(path, "a" * 64, **update)

    with pytest.raises(GenerationValidationError, match=error):
        load_generation_plan(path)


def test_executes_two_banks_and_writes_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, plan = _plans(tmp_path)
    monkeypatch.setattr(generation_bank, "verify_executor_revision", _verified_revision)
    seen: list[tuple[str, ...]] = []

    def run(
        smoke_plan: SmokePlan,
        generation_plan: GenerationPlan,
        prompts: tuple[PromptRecord, ...],
    ) -> GenerationRuntime:
        del smoke_plan, generation_plan
        seen.append(tuple(prompt.question_id for prompt in prompts))
        return _runtime(prompts)

    receipt = execute_generation_bank(
        plan,
        smoke,
        code_revision="d" * 40,
        banks=(_bank("validate_feelings"), _bank("refusal")),
        limit_per_bank=1,
        output_dir=tmp_path / "run",
        run=run,
    )

    assert seen == [("validate_feelings-0", "refusal-0")]
    assert receipt.sample_count == 2
    assert receipt.generated_tokens == 16
    assert receipt.scientific_use is False
    assert receipt.starts_experiment_timer is False
    assert (tmp_path / "run/receipt.json").exists()
    assert len(list((tmp_path / "run/samples").glob("*.json"))) == 2


def test_resumes_after_interruption_without_regenerating_completed_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, plan = _plans(tmp_path)
    monkeypatch.setattr(generation_bank, "verify_executor_revision", _verified_revision)
    calls: list[tuple[str, ...]] = []

    def interrupted(
        smoke_plan: SmokePlan,
        generation_plan: GenerationPlan,
        prompts: tuple[PromptRecord, ...],
    ) -> GenerationRuntime:
        del smoke_plan, generation_plan
        calls.append(tuple(prompt.question_id for prompt in prompts))

        def observations():
            yield _observation(prompts[0])
            raise RuntimeError("simulated interruption")

        runtime = _runtime(())
        return GenerationRuntime(
            runtime.adapter_sha256,
            runtime.template_sha256,
            runtime.runtime_versions,
            observations(),
        )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute_generation_bank(
            plan,
            smoke,
            code_revision="d" * 40,
            banks=(_bank("refusal"),),
            limit_per_bank=None,
            output_dir=tmp_path / "run",
            run=interrupted,
        )

    execute_generation_bank(
        plan,
        smoke,
        code_revision="d" * 40,
        banks=(_bank("refusal"),),
        limit_per_bank=None,
        output_dir=tmp_path / "run",
        run=_successful_run,
    )

    assert calls == [("refusal-0", "refusal-1")]
    samples = list((tmp_path / "run/samples").glob("*.json"))
    assert len(samples) == 2


def test_resume_rejects_tampered_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, plan = _plans(tmp_path)
    monkeypatch.setattr(generation_bank, "verify_executor_revision", _verified_revision)
    output_dir = tmp_path / "run"
    execute_generation_bank(
        plan,
        smoke,
        code_revision="d" * 40,
        banks=(_bank("refusal", count=1),),
        limit_per_bank=None,
        output_dir=output_dir,
        run=_successful_run,
    )
    sample = next((output_dir / "samples").glob("*.json"))
    payload = json.loads(sample.read_text())
    payload["completion"] = "tampered"
    sample.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GenerationValidationError, match="sample receipt hash"):
        execute_generation_bank(
            plan,
            smoke,
            code_revision="d" * 40,
            banks=(_bank("refusal", count=1),),
            limit_per_bank=None,
            output_dir=output_dir,
            run=_unexpected_run,
        )
