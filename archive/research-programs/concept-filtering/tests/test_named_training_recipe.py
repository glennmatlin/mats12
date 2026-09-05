from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from concept_filtering.named_training_recipe import (
    NamedTrainingRecipeError,
    load_named_training_recipe,
)

ROOT = Path(__file__).parents[1]
NAMED_RECIPE = (
    ROOT / "configs/reconstruction/training/public-rank32-named-recipe-v1.json"
)
LOAD_SMOKE_EVIDENCE = (
    ROOT / "configs/reconstruction/receipts/public-rank32-load-smoke-v1.json"
)
RETAINED_LOAD_SMOKE = (
    ROOT / "reports/reconstruction/smokes/2026-08-14-public-rank32-split-1.json"
)


def test_loaded_recipe_exposes_immutable_validated_mappings() -> None:
    recipe = load_named_training_recipe(NAMED_RECIPE)
    mappings: tuple[tuple[object, str, object], ...] = (
        (recipe.artifacts.base.files, "config.json", "0" * 64),
        (recipe.evidence_receipt_sha256, "public_load_smoke", "0" * 64),
        (recipe.runtime_versions, "torch", "0.0.0"),
        (recipe.provenance, "lora.rank", object()),
    )

    for mapping, key, value in mappings:
        with pytest.raises(TypeError):
            cast(dict[str, object], mapping)[key] = value


def test_exposes_training_sample_upstream_source_identity() -> None:
    recipe = load_named_training_recipe(NAMED_RECIPE)

    assert recipe.artifacts.training_data.source_repository == (
        "allenai/Dolci-Think-SFT-7B"
    )
    assert recipe.artifacts.training_data.source_revision == (
        "72ec0fe32428bada1ec686a9168a0681eecd2094"
    )


def copy_recipe_bundle(tmp_path: Path) -> Path:
    training = tmp_path / "training"
    receipts = tmp_path / "receipts"
    evidence = tmp_path / "evidence/public-rank32-split1"
    fixtures = training / "fixtures"
    training.mkdir()
    fixtures.mkdir()
    receipts.mkdir()
    evidence.mkdir(parents=True)
    config_path = training / NAMED_RECIPE.name
    shutil.copyfile(NAMED_RECIPE, config_path)
    shutil.copyfile(
        ROOT / "configs/reconstruction/public-organism-v1.json",
        tmp_path / "public-organism-v1.json",
    )
    for name in (
        "full-data-adapter.json",
        "public-rank32-load-smoke-v1.json",
        "public-rank32-split1-recipe-evidence-v1.json",
        "public-rank32-train1-reconstruction-v1.json",
        "public-rank32-training-derivation-evidence-v1.json",
    ):
        source = ROOT / "configs/reconstruction/receipts" / name
        shutil.copyfile(source, receipts / name)
    for name in ("adapter_config.json", "trainer_state.json"):
        source = ROOT / "configs/reconstruction/evidence/public-rank32-split1" / name
        shutil.copyfile(source, evidence / name)
    _copy_smoke_fixture(fixtures)
    return config_path


def _copy_smoke_fixture(fixtures: Path) -> None:
    name = "public-rank32-recovery-smoke-fixture-v1.json"
    source = ROOT / "configs/reconstruction/training/fixtures" / name
    shutil.copyfile(source, fixtures / name)


def mutate_recipe(config_path: Path, mutation: Any) -> None:
    payload = cast(dict[str, Any], json.loads(config_path.read_text(encoding="utf-8")))
    mutation(payload)
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def rebind_receipt(
    config_path: Path, role: str, receipt_path: Path, payload: dict[str, Any]
) -> None:
    body = dict(payload)
    body.pop("receipt_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    source = json.dumps(payload).encode("utf-8")
    receipt_path.write_bytes(source)

    def update_outer_hash(config: dict[str, Any]) -> None:
        receipts = cast(dict[str, Any], config["evidence_receipts"])
        item = cast(dict[str, Any], receipts[role])
        item["sha256"] = hashlib.sha256(source).hexdigest()

    mutate_recipe(config_path, update_outer_hash)


def copied_receipt(tmp_path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    path = tmp_path / "receipts" / name
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return path, payload


def rebind_recipe_pointer(config_path: Path, tmp_path: Path, recipe_path: Path) -> None:
    path, receipt = copied_receipt(
        tmp_path, "public-rank32-training-derivation-evidence-v1.json"
    )
    sources = cast(dict[str, Any], receipt["sources"])
    sources["public_recipe_observation_receipt_sha256"] = hashlib.sha256(
        recipe_path.read_bytes()
    ).hexdigest()
    rebind_receipt(config_path, "training_derivation", path, receipt)


def rebind_smoke_adapter_replacement(config_path: Path, tmp_path: Path) -> None:
    path, receipt = copied_receipt(tmp_path, "public-rank32-load-smoke-v1.json")
    adapter = cast(dict[str, Any], receipt["adapter"])
    template = cast(dict[str, Any], receipt["template"])
    adapter.update(
        repository="replacement/adapter",
        revision="b" * 40,
        path="other/adapter_model.safetensors",
    )
    template.update(path="other/chat_template.jinja", sha256="a" * 64)
    rebind_receipt(config_path, "public_load_smoke", path, receipt)


def replace_tokenizer_with_replacement(payload: dict[str, Any]) -> None:
    artifacts = cast(dict[str, Any], payload["artifacts"])
    tokenizer = cast(dict[str, Any], artifacts["tokenizer"])
    tokenizer.update(repository="replacement/adapter", revision="b" * 40, path="other")
    files = cast(dict[str, Any], tokenizer["files"])
    files["chat_template.jinja"] = "a" * 64
    recipe = cast(dict[str, Any], payload["recipe"])
    sequence = cast(dict[str, Any], recipe["sequence"])
    sequence["chat_template_sha256"] = "a" * 64


def rebind_rank_sources(config_path: Path, tmp_path: Path, rank: int) -> None:
    recipe_path, recipe_receipt = copied_receipt(
        tmp_path, "public-rank32-split1-recipe-evidence-v1.json"
    )
    adapter = cast(dict[str, Any], recipe_receipt["adapter_observation"])
    adapter["r"] = rank
    rebind_receipt(
        config_path, "public_recipe_observation", recipe_path, recipe_receipt
    )
    smoke_path, smoke_receipt = copied_receipt(
        tmp_path, "public-rank32-load-smoke-v1.json"
    )
    smoke_receipt["rank"] = rank
    rebind_receipt(config_path, "public_load_smoke", smoke_path, smoke_receipt)
    rebind_recipe_pointer(config_path, tmp_path, recipe_path)


def replacement_trace_values() -> tuple[dict[str, object], dict[str, object]]:
    observations: dict[str, object] = {
        "optimizer_steps": 100,
        "logged_tokens": 52_428_800,
        "train_runtime_seconds": 1600.0,
        "train_samples_per_second": 4.0,
    }
    derivation: dict[str, object] = {
        "derived_packed_sequence_count": 6400.0,
        "derived_sequences_per_step": 64.0,
        "selected_effective_batch_size": 64,
        "selected_token_capacity": 52_428_800,
        "observed_capacity_fraction": 1.0,
    }
    return observations, derivation


def rebind_trace_sources(
    config_path: Path, tmp_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    recipe_path, recipe_receipt = copied_receipt(
        tmp_path, "public-rank32-split1-recipe-evidence-v1.json"
    )
    trainer = cast(dict[str, Any], recipe_receipt["trainer_state_observation"])
    trainer.update(global_step=100, max_steps=100, logged_step_count=100)
    trainer["final_logged_num_tokens"] = 52_428_800.0
    rebind_receipt(
        config_path, "public_recipe_observation", recipe_path, recipe_receipt
    )
    path, receipt = copied_receipt(
        tmp_path, "public-rank32-training-derivation-evidence-v1.json"
    )
    observations, derivation = replacement_trace_values()
    cast(dict[str, Any], receipt["observations"]).update(observations)
    cast(dict[str, Any], receipt["derivation"]).update(derivation)
    sources = cast(dict[str, Any], receipt["sources"])
    sources["public_recipe_observation_receipt_sha256"] = hashlib.sha256(
        recipe_path.read_bytes()
    ).hexdigest()
    rebind_receipt(config_path, "training_derivation", path, receipt)
    return observations, derivation


def replace_trace(
    payload: dict[str, Any],
    observations: dict[str, object],
    derivation: dict[str, object],
) -> None:
    trace = cast(dict[str, Any], payload["public_trace_derivation"])
    trace.update(observations)
    trace.update(derivation)


def test_loads_complete_named_recipe_and_public_evidence() -> None:
    recipe = load_named_training_recipe(NAMED_RECIPE)

    assert recipe.recipe_id == "public-rank32-named-recipe-v1"
    assert recipe.conditions == (
        "full_data",
        "dense_probe_removal",
        "matched_random_removal",
        "behavior_query_ekfac_removal",
    )
    assert recipe.values.lora.rank == 32
    assert recipe.values.batching.effective_batch_size == 64
    assert recipe.values.sequence.max_sequence_length == 8192
    assert recipe.values.checkpoint.interval_optimizer_steps == 10
    assert recipe.smoke.total_optimizer_steps == 2
    assert len(recipe.config_sha256) == 64
    assert len(recipe.recipe_core_sha256) == 64
    assert len(recipe.smoke_authority_sha256) == 64
    assert recipe.recipe_core_sha256 != recipe.smoke_authority_sha256
    assert set(recipe.smoke_authority) == {
        "scientific_use",
        "starts_experiment_timer",
        "provider_execution",
        "scheduler_submission",
        "full_condition_training",
        "feature_selection",
        "concept_scoring",
    }
    assert not any(recipe.smoke_authority.values())
    assert len(recipe.smoke_fixture.sha256) == 64


def test_recipe_core_identity_excludes_smoke_only_settings(tmp_path: Path) -> None:
    original = load_named_training_recipe(NAMED_RECIPE)
    config_path = copy_recipe_bundle(tmp_path)

    def change_smoke(payload: dict[str, Any]) -> None:
        smoke = cast(dict[str, Any], payload["smoke"])
        comparison = cast(dict[str, Any], smoke["comparison"])
        comparison["absolute_tolerance"] = 2e-06

    mutate_recipe(config_path, change_smoke)
    changed = load_named_training_recipe(config_path)

    assert changed.config_sha256 != original.config_sha256
    assert changed.recipe_core_sha256 == original.recipe_core_sha256
    assert changed.smoke_authority_sha256 == original.smoke_authority_sha256


def test_rejects_contradictory_frozen_preprocessing_counts(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_expected(payload: dict[str, Any]) -> None:
        smoke = cast(dict[str, Any], payload["smoke"])
        expected = cast(dict[str, Any], smoke["expected_preprocessing"])
        expected["selected_example_ids"] = ["0" * 64]

    mutate_recipe(config_path, change_expected)

    with pytest.raises(NamedTrainingRecipeError, match="preprocessing"):
        load_named_training_recipe(config_path)


def test_accepts_multiple_assistant_turns_with_terminal_example_eos(
    tmp_path: Path,
) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def add_multiturn_count(payload: dict[str, Any]) -> None:
        smoke = cast(dict[str, Any], payload["smoke"])
        expected = cast(dict[str, Any], smoke["expected_preprocessing"])
        expected["assistant_turn_count"] = 3

    mutate_recipe(config_path, add_multiturn_count)

    recipe = load_named_training_recipe(config_path)

    expected = recipe.smoke.expected_preprocessing
    assert expected.assistant_turn_count == 3
    assert expected.eos_label_count == 2


def test_rejects_tampered_smoke_fixture(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    fixture = (
        config_path.parent / "fixtures/public-rank32-recovery-smoke-fixture-v1.json"
    )
    fixture.write_text("{}\n", encoding="utf-8")

    with pytest.raises(NamedTrainingRecipeError, match="fixture SHA-256"):
        load_named_training_recipe(config_path)


def test_rejects_smoke_fixture_receipt_contradiction(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_receipt(payload: dict[str, Any]) -> None:
        fixture = cast(dict[str, Any], payload["smoke_fixture"])
        fixture["source_sample_receipt_sha256"] = "0" * 64

    mutate_recipe(config_path, change_receipt)

    with pytest.raises(NamedTrainingRecipeError, match="source sample receipt"):
        load_named_training_recipe(config_path)


def test_load_smoke_evidence_matches_retained_receipt() -> None:
    assert LOAD_SMOKE_EVIDENCE.read_bytes() == RETAINED_LOAD_SMOKE.read_bytes()


def test_rejects_recipe_without_complete_provenance(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def remove_entry(payload: dict[str, Any]) -> None:
        cast(dict[str, Any], payload["provenance"]).pop("sequence.loss_mask")

    mutate_recipe(config_path, remove_entry)

    with pytest.raises(NamedTrainingRecipeError, match="provenance coverage"):
        load_named_training_recipe(config_path)


def test_rejects_effective_batch_contradiction(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_batch(payload: dict[str, Any]) -> None:
        recipe = cast(dict[str, Any], payload["recipe"])
        batching = cast(dict[str, Any], recipe["batching"])
        batching["effective_batch_size"] = 63

    mutate_recipe(config_path, change_batch)

    with pytest.raises(NamedTrainingRecipeError, match="effective batch"):
        load_named_training_recipe(config_path)


def test_rejects_public_trace_derivation_contradiction(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_derivation(payload: dict[str, Any]) -> None:
        derivation = cast(dict[str, Any], payload["public_trace_derivation"])
        derivation["derived_packed_sequence_count"] = 1.0

    mutate_recipe(config_path, change_derivation)

    with pytest.raises(NamedTrainingRecipeError, match="packed sequence derivation"):
        load_named_training_recipe(config_path)


def test_rejects_changed_evidence_receipt_bytes(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    receipt = tmp_path / "receipts/public-rank32-train1-reconstruction-v1.json"
    receipt.write_text("{}", encoding="utf-8")

    with pytest.raises(NamedTrainingRecipeError, match="evidence receipt SHA-256"):
        load_named_training_recipe(config_path)


def test_rejects_receipt_with_rebound_file_hash_but_stale_body_hash(
    tmp_path: Path,
) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    receipt_path = (
        tmp_path / "receipts/public-rank32-training-derivation-evidence-v1.json"
    )
    receipt = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
    observations = cast(dict[str, Any], receipt["observations"])
    observations["optimizer_steps"] = 100
    source = json.dumps(receipt).encode("utf-8")
    receipt_path.write_bytes(source)

    def rebind_outer_hash(payload: dict[str, Any]) -> None:
        receipts = cast(dict[str, Any], payload["evidence_receipts"])
        derivation = cast(dict[str, Any], receipts["training_derivation"])
        derivation["sha256"] = hashlib.sha256(source).hexdigest()

    mutate_recipe(config_path, rebind_outer_hash)

    with pytest.raises(NamedTrainingRecipeError, match="receipt body SHA-256"):
        load_named_training_recipe(config_path)


def test_rejects_swapped_evidence_receipt_roles(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def swap_roles(payload: dict[str, Any]) -> None:
        receipts = cast(dict[str, Any], payload["evidence_receipts"])
        left = receipts["public_recipe_observation"]
        receipts["public_recipe_observation"] = receipts["training_sample"]
        receipts["training_sample"] = left

    mutate_recipe(config_path, swap_roles)

    with pytest.raises(NamedTrainingRecipeError, match="receipt role"):
        load_named_training_recipe(config_path)


def test_rejects_recipe_value_that_contradicts_public_receipt(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_rank(payload: dict[str, Any]) -> None:
        recipe = cast(dict[str, Any], payload["recipe"])
        lora = cast(dict[str, Any], recipe["lora"])
        lora["rank"] = 8

    mutate_recipe(config_path, change_rank)

    with pytest.raises(NamedTrainingRecipeError, match="public adapter evidence"):
        load_named_training_recipe(config_path)


def test_rejects_coordinated_public_trace_replacement(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def replace_trace(payload: dict[str, Any]) -> None:
        trace = cast(dict[str, Any], payload["public_trace_derivation"])
        trace["optimizer_steps"] = 100
        trace["logged_tokens"] = 52_428_800
        trace["train_samples_per_second"] = 6400 / 1545.2445
        trace["derived_packed_sequence_count"] = 6400.0
        trace["derived_sequences_per_step"] = 64.0
        trace["selected_token_capacity"] = 52_428_800
        trace["observed_capacity_fraction"] = 1.0

    mutate_recipe(config_path, replace_trace)

    with pytest.raises(NamedTrainingRecipeError, match="public trace evidence"):
        load_named_training_recipe(config_path)


def test_rejects_cross_field_identity_contradictions(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_template(payload: dict[str, Any]) -> None:
        recipe = cast(dict[str, Any], payload["recipe"])
        sequence = cast(dict[str, Any], recipe["sequence"])
        sequence["chat_template_sha256"] = "a" * 64

    mutate_recipe(config_path, change_template)

    with pytest.raises(NamedTrainingRecipeError, match="public adapter evidence"):
        load_named_training_recipe(config_path)


def test_rejects_coordinated_chat_template_replacement(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def replace_template_hash(payload: dict[str, Any]) -> None:
        artifacts = cast(dict[str, Any], payload["artifacts"])
        tokenizer = cast(dict[str, Any], artifacts["tokenizer"])
        files = cast(dict[str, Any], tokenizer["files"])
        files["chat_template.jinja"] = "a" * 64
        recipe = cast(dict[str, Any], payload["recipe"])
        sequence = cast(dict[str, Any], recipe["sequence"])
        sequence["chat_template_sha256"] = "a" * 64

    mutate_recipe(config_path, replace_template_hash)

    with pytest.raises(NamedTrainingRecipeError, match="public adapter evidence"):
        load_named_training_recipe(config_path)


def test_rejects_rebound_load_smoke_adapter_identity(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    receipt_path = tmp_path / "receipts/public-rank32-load-smoke-v1.json"
    receipt = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
    adapter = cast(dict[str, Any], receipt["adapter"])
    adapter["path"] = "split-1/replacement.safetensors"
    adapter["sha256"] = "a" * 64
    receipt["rank"] = 8
    rebind_receipt(config_path, "public_load_smoke", receipt_path, receipt)

    with pytest.raises(NamedTrainingRecipeError, match="public adapter evidence"):
        load_named_training_recipe(config_path)


def test_rejects_coordinated_adapter_and_tokenizer_rebinding(
    tmp_path: Path,
) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    rebind_smoke_adapter_replacement(config_path, tmp_path)
    mutate_recipe(config_path, replace_tokenizer_with_replacement)

    with pytest.raises(NamedTrainingRecipeError, match="public adapter evidence"):
        load_named_training_recipe(config_path)


def test_rejects_fully_rebound_rank_replacement(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    rebind_rank_sources(config_path, tmp_path, 16)

    def replace_rank(payload: dict[str, Any]) -> None:
        recipe = cast(dict[str, Any], payload["recipe"])
        cast(dict[str, Any], recipe["lora"])["rank"] = 16

    mutate_recipe(config_path, replace_rank)
    with pytest.raises(NamedTrainingRecipeError, match="raw adapter evidence"):
        load_named_training_recipe(config_path)


def test_rejects_fully_rebound_trace_replacement(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    observations, derivation = rebind_trace_sources(config_path, tmp_path)

    def apply_trace(payload: dict[str, Any]) -> None:
        replace_trace(payload, observations, derivation)

    mutate_recipe(config_path, apply_trace)
    with pytest.raises(NamedTrainingRecipeError, match="raw trainer evidence"):
        load_named_training_recipe(config_path)


def test_rejects_max_length_that_contradicts_public_organism(
    tmp_path: Path,
) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    receipt_path = (
        tmp_path / "receipts/public-rank32-training-derivation-evidence-v1.json"
    )
    receipt = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
    observations = cast(dict[str, Any], receipt["observations"])
    derivation = cast(dict[str, Any], receipt["derivation"])
    capacity = 123 * 64 * 16_384
    observations["max_sequence_length"] = 16_384
    derivation["selected_token_capacity"] = capacity
    derivation["observed_capacity_fraction"] = 63_696_407 / capacity
    rebind_receipt(config_path, "training_derivation", receipt_path, receipt)

    def replace_max_length(payload: dict[str, Any]) -> None:
        recipe = cast(dict[str, Any], payload["recipe"])
        sequence = cast(dict[str, Any], recipe["sequence"])
        sequence["max_sequence_length"] = 16_384
        trace = cast(dict[str, Any], payload["public_trace_derivation"])
        trace["selected_token_capacity"] = capacity
        trace["observed_capacity_fraction"] = 63_696_407 / capacity

    mutate_recipe(config_path, replace_max_length)

    with pytest.raises(NamedTrainingRecipeError, match="public organism evidence"):
        load_named_training_recipe(config_path)


def test_rejects_order_identity_that_differs_from_training_data(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_identity(payload: dict[str, Any]) -> None:
        recipe = cast(dict[str, Any], payload["recipe"])
        order = cast(dict[str, Any], recipe["order"])
        order["identity_scheme"] = "another-scheme"

    mutate_recipe(config_path, change_identity)

    with pytest.raises(NamedTrainingRecipeError, match="order identity scheme"):
        load_named_training_recipe(config_path)


@pytest.mark.parametrize("field", ["model_dtype", "checkpoint_interval"])
def test_rejects_smoke_execution_contradiction(tmp_path: Path, field: str) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_smoke(payload: dict[str, Any]) -> None:
        smoke = cast(dict[str, Any], payload["smoke"])
        if field == "model_dtype":
            smoke["model_dtype"] = "float32"
        else:
            overrides = cast(dict[str, Any], smoke["overrides"])
            overrides["checkpoint_interval_optimizer_steps"] = 2

    mutate_recipe(config_path, change_smoke)

    with pytest.raises(NamedTrainingRecipeError, match="smoke execution contract"):
        load_named_training_recipe(config_path)


def test_rejects_provenance_basis_flip(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def flip_basis(payload: dict[str, Any]) -> None:
        provenance = cast(dict[str, Any], payload["provenance"])
        rank = cast(dict[str, Any], provenance["lora.rank"])
        rank["basis"] = "reconstruction_choice"

    mutate_recipe(config_path, flip_basis)

    with pytest.raises(NamedTrainingRecipeError, match=r"provenance.*basis"):
        load_named_training_recipe(config_path)


@pytest.mark.parametrize(
    "field",
    [
        "scientific_use",
        "starts_experiment_timer",
        "provider_execution",
        "scheduler_submission",
        "full_condition_training",
        "feature_selection",
        "concept_scoring",
    ],
)
def test_rejects_enabled_execution_boundary(tmp_path: Path, field: str) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def enable_boundary(payload: dict[str, Any]) -> None:
        authority = cast(dict[str, Any], payload["smoke_authority"])
        authority[field] = True

    mutate_recipe(config_path, enable_boundary)

    with pytest.raises(NamedTrainingRecipeError, match=f"{field} must remain false"):
        load_named_training_recipe(config_path)


def test_rejects_smoke_scale_that_changes_recipe_identity(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)

    def change_rank(payload: dict[str, Any]) -> None:
        smoke = cast(dict[str, Any], payload["smoke"])
        overrides = cast(dict[str, Any], smoke["overrides"])
        overrides["rank"] = 8

    mutate_recipe(config_path, change_rank)

    with pytest.raises(NamedTrainingRecipeError, match=r"smoke\.overrides"):
        load_named_training_recipe(config_path)


def test_rejects_duplicate_recipe_keys(tmp_path: Path) -> None:
    config_path = copy_recipe_bundle(tmp_path)
    source = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        source.replace('"rank": 32', '"rank": 32, "rank": 8'), encoding="utf-8"
    )

    with pytest.raises(NamedTrainingRecipeError, match="duplicate JSON key"):
        load_named_training_recipe(config_path)
