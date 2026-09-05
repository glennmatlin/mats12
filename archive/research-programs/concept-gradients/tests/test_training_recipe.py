from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from concept_filtering.cli import main
from concept_filtering.serialization import canonical_sha256
from concept_filtering.training_recipe import load_training_recipe

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/training/public-rank32-reconstruction-v1.json"


def _recipe() -> dict[str, object]:
    return cast(dict[str, object], json.loads(RECIPE.read_text(encoding="utf-8")))


def _write_recipe(tmp_path: Path, payload: dict[str, object]) -> Path:
    fixture = ROOT / "configs/training/fixtures/public-rank32-training-smoke-v1.json"
    target_fixture = tmp_path / "fixtures/public-rank32-training-smoke-v1.json"
    target_fixture.parent.mkdir(exist_ok=True)
    shutil.copyfile(fixture, target_fixture)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _object(payload: dict[str, object], field: str) -> dict[str, object]:
    value = payload[field]
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _recipe_object(payload: dict[str, object], field: str) -> dict[str, object]:
    return _object(_object(payload, "recipe"), field)


def _execution_object(payload: dict[str, object], field: str) -> dict[str, object]:
    return _object(_object(payload, "execution"), field)


def test_cli_validates_named_recipe_and_reports_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 0
    core = _object(payload, "recipe")
    assert json.loads(capsys.readouterr().out) == {
        "bounded_smoke_execution_allowed": True,
        "contract_id": "public-rank32-reconstruction-smoke-v1",
        "contract_sha256": canonical_sha256(payload),
        "effective_batch_size": 64,
        "phase": "bounded_smoke",
        "rank": 32,
        "recipe_id": "public-rank32-reconstruction-v1",
        "recipe_sha256": canonical_sha256(core),
        "scientific_training_allowed": False,
        "scientific_use": False,
    }


def test_checked_in_named_recipe_validates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["validate-training-recipe", str(RECIPE)])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["recipe_id"] == "public-rank32-reconstruction-v1"
    assert summary["scientific_training_allowed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "scientific_use",
        "scientific_training_allowed",
        "provider_execution_allowed",
        "scheduler_execution_allowed",
        "sae_feature_selection_allowed",
        "concept_scoring_allowed",
        "concept_selected_training_allowed",
    ],
)
def test_rejects_authority_expansion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    payload = _recipe()
    _execution_object(payload, "boundaries")[field] = True

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("value", [False, 0, 1, None, "true", []])
def test_bounded_smoke_permission_requires_true_boolean(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    value: object,
) -> None:
    payload = _recipe()
    _execution_object(payload, "boundaries")["bounded_smoke_execution_allowed"] = value

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "bounded_smoke_execution_allowed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("adapter", "rank", 64),
        ("adapter", "method", "full_finetune"),
        ("data", "max_sequence_length", 4096),
        ("data", "packing", False),
        ("data", "loss_mask", "all_tokens"),
        ("optimization", "scheduler", "cosine"),
        ("optimization", "warmup_steps", 0),
        ("training", "epochs", 2),
        ("training", "effective_batch_size", 63),
        ("training", "precision", "float16"),
        ("checkpointing", "resume_mode", "weights_only"),
    ],
)
def test_rejects_recipe_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    section: str,
    field: str,
    value: object,
) -> None:
    payload = _recipe()
    _recipe_object(payload, section)[field] = value

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert field in capsys.readouterr().err


def test_rejects_inconsistent_effective_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    _recipe_object(payload, "training")["gradient_accumulation_steps"] = 2

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "effective_batch_size" in capsys.readouterr().err


def test_rejects_duplicate_json_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps(_recipe()).replace(
            '"scientific_use": false,',
            '"scientific_use": false, "scientific_use": true,',
            1,
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-training-recipe", str(path)])

    assert exit_code == 2
    assert "duplicate JSON key" in capsys.readouterr().err


def test_rejects_unknown_nested_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    _recipe_object(payload, "optimization")["unreviewed"] = True

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "unexpected fields" in capsys.readouterr().err


def test_rejects_non_sha_evidence_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    _recipe_object(payload, "evidence")["public_recipe_receipt_sha256"] = "latest"

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "SHA-256" in capsys.readouterr().err


def test_rejects_smoke_fixture_symlink_escape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    path = _write_recipe(tmp_path, payload)
    fixture = ROOT / "configs/training/fixtures/public-rank32-training-smoke-v1.json"
    (tmp_path / "escape.json").symlink_to(fixture)
    _execution_object(payload, "smoke_fixture")["path"] = "escape.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["validate-training-recipe", str(path)])

    assert exit_code == 2
    assert "must remain inside" in capsys.readouterr().err


def test_rejects_smoke_fixture_symlink_loop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    path = _write_recipe(tmp_path, payload)
    (tmp_path / "loop.json").symlink_to("loop.json")
    _execution_object(payload, "smoke_fixture")["path"] = "loop.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["validate-training-recipe", str(path)])

    assert exit_code == 2
    assert "cannot load smoke fixture" in capsys.readouterr().err


def test_rejects_unhashable_target_module_through_public_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    _recipe_object(payload, "adapter")["target_modules"] = [[]] * 7

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "non-empty strings" in capsys.readouterr().err


def test_provenance_must_cover_every_frozen_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    del _recipe_object(payload, "field_provenance")["data.append_eos"]

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "field_provenance" in capsys.readouterr().err


def test_source_provenance_reference_must_resolve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    entry = _object(_recipe_object(payload, "field_provenance"), "adapter.rank")
    entry["evidence_ref"] = "configs/reconstruction/receipts/base-model.json#/absent"

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "does not resolve" in capsys.readouterr().err


def test_source_provenance_value_must_support_the_frozen_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _recipe()
    entry = _object(_recipe_object(payload, "field_provenance"), "adapter.rank")
    entry["evidence_ref"] = (
        "configs/reconstruction/receipts/"
        "public-rank32-split1-recipe-evidence-v1.json"
        "#/adapter_observation/lora_alpha"
    )

    exit_code = main(
        ["validate-training-recipe", str(_write_recipe(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "does not support" in capsys.readouterr().err


def test_recipe_hash_excludes_smoke_authority_envelope(tmp_path: Path) -> None:
    payload = _recipe()
    original_core = canonical_sha256(_object(payload, "recipe"))
    execution = _object(payload, "execution")
    limitations = cast(list[str], execution["limitations"])
    limitations.append("A second valid limitation changes only the smoke contract.")

    path = _write_recipe(tmp_path, payload)
    loaded = load_training_recipe(path)

    assert loaded.recipe_sha256 == original_core
    assert loaded.contract_sha256 == canonical_sha256(payload)
    assert loaded.contract_sha256 != loaded.recipe_sha256
