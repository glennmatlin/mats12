from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import concept_filtering.training_data_package as data_package
from concept_filtering.cli import main
from concept_filtering.serialization import canonical_json_bytes, canonical_sha256
from concept_filtering.training_data_package import (
    TrainingDataPackageConfig,
    TrainingDataPackageError,
    load_bound_training_example,
    load_training_data_package_config,
)
from concept_filtering.training_recipe import load_training_recipe

ROOT = Path(__file__).parents[1]
PACKAGE_CONFIG = ROOT / "configs/training/public-rank32-training-data-package-v1.json"
RECIPE_CONFIG = ROOT / "configs/training/public-rank32-reconstruction-v1.json"
SAMPLE_RECEIPT = (
    ROOT / "configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json"
)

_TABLE = cast(
    Callable[[dict[str, object]], pa.Table],
    pa.table,  # pyright: ignore[reportUnknownMemberType]
)
_WRITE_TABLE = cast(
    Callable[[pa.Table, str], None],
    pq.write_table,  # pyright: ignore[reportUnknownMemberType]
)
_PUBLISH_EVIDENCE = cast(
    Callable[[Path, bytes, Path, bytes], None],
    getattr(data_package, "_publish_evidence"),  # noqa: B009
)
_PUBLISH_OR_MATCH = cast(
    Callable[[Path, bytes, str], object],
    getattr(data_package, "_publish_or_match"),  # noqa: B009
)
_VALIDATE_RECEIPT_BUILD = cast(
    Callable[..., None],
    getattr(data_package, "_validate_receipt_build"),  # noqa: B009
)


def _identity(value: str) -> str:
    return value


def _ignore_provenance(
    config: TrainingDataPackageConfig,
    *,
    recipe_path: Path,
    sample_receipt_path: Path,
) -> None:
    del config, recipe_path, sample_receipt_path


def _allow_fixture_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_package, "validate_input_provenance", _ignore_provenance)


def _template() -> str:
    return (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}{{ 'user ' + message['content'] + ' ' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ 'assistant ' + message['content'] + ' ' }}"
        "{% if loop.last %}{{ eos_token }}{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ 'environment ' + message['content'] + ' ' }}"
        "{% endif %}{% endfor %}"
    )


def _replace_with_alias(
    output: Path, external: Path, value: bytes, alias_kind: str
) -> None:
    output.unlink()
    external.write_bytes(value)
    if alias_kind == "symlink":
        output.symlink_to(external)
    else:
        os.link(external, output)


def _config() -> dict[str, object]:
    template = _template()
    return {
        "schema_version": 1,
        "package_id": "fixture-training-data-package-v1",
        "phase": "pre_feature_training_data_materialization",
        "boundaries": {
            "training_data_materialization_allowed": True,
            "base_response_generation_allowed": False,
            "activation_extraction_allowed": False,
            "training_execution_allowed": False,
            "scientific_use": False,
            "starts_experiment_timer": False,
            "provider_execution_allowed": False,
            "scheduler_execution_allowed": False,
            "sae_feature_selection_allowed": False,
            "concept_scoring_allowed": False,
            "concept_selected_training_allowed": False,
        },
        "sample": {
            "repository": "owner/sample",
            "revision": "1" * 40,
            "path": "data/train.parquet",
            "sha256": "2" * 64,
            "source_revision": "3" * 40,
            "training_sample_receipt_sha256": "4" * 64,
            "expected_row_count": 2,
            "expected_split": 1,
            "identity_scheme": "source-revision-orig-idx-v1",
        },
        "tokenizer": {
            "repository": "owner/tokenizer",
            "revision": "5" * 40,
            "files": {
                "tokenizer.json": "6" * 64,
                "tokenizer_config.json": "7" * 64,
            },
        },
        "runtime": {
            "huggingface_hub": "0.36.2",
            "pyarrow": "23.0.1",
            "tokenizers": "0.22.2",
            "transformers": "4.57.6",
        },
        "training": {
            "recipe_id": "fixture-recipe-v1",
            "recipe_sha256": "8" * 64,
            "chat_template": template,
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "max_sequence_length": 512,
            "overlength_policy": "drop",
            "truncation": False,
            "append_eos": True,
            "loss_mask": "assistant_only",
        },
        "output": {
            "format": "canonical-jsonl-v1",
            "entry_schema_version": 1,
        },
        "input_producer_releases": {
            "training_sample_evidence": "9" * 40,
            "training_recipe_evidence": "a" * 40,
        },
        "limitations": ["Fixture configuration is not scientific evidence."],
    }


def test_loads_fail_closed_training_data_package_config(tmp_path: Path) -> None:
    config_path = tmp_path / "package.json"
    payload = _config()
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_training_data_package_config(config_path)

    assert config.package_id == "fixture-training-data-package-v1"
    assert config.expected_row_count == 2
    assert config.training_data_materialization_allowed is True
    assert config.base_response_generation_allowed is False
    assert config.runtime_versions["transformers"] == "4.57.6"
    assert config.runtime_versions["pyarrow"] == "23.0.1"
    assert config.config_sha256 == canonical_sha256(payload)


def test_loaded_config_mappings_are_immutable(tmp_path: Path) -> None:
    config_path = tmp_path / "package.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    config = load_training_data_package_config(config_path)

    with pytest.raises(TypeError):
        cast(dict[str, bool], config.boundaries)["scientific_use"] = True
    with pytest.raises(TypeError):
        cast(dict[str, str], config.input_producer_releases)[
            "training_sample_evidence"
        ] = "f" * 40


def test_checked_in_package_config_matches_frozen_inputs() -> None:
    config = load_training_data_package_config(PACKAGE_CONFIG)
    recipe = load_training_recipe(RECIPE_CONFIG)
    sample_receipt = json.loads(SAMPLE_RECEIPT.read_text())
    fixture = json.loads(
        (
            ROOT / "configs/training/fixtures/public-rank32-training-smoke-v1.json"
        ).read_text()
    )

    assert config.training.recipe_sha256 == recipe.recipe_sha256
    assert config.training.chat_template == fixture["chat_template"]
    assert (
        config.sample.training_sample_receipt_sha256
        == (sample_receipt["receipt_sha256"])
    )
    assert config.sample.expected_row_count == sample_receipt["row_count"]
    assert config.sample.identity_scheme == sample_receipt["identity_scheme"]
    data_package.validate_input_provenance(
        config,
        recipe_path=RECIPE_CONFIG,
        sample_receipt_path=SAMPLE_RECEIPT,
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("sample", "repository", "other/sample"),
        ("sample", "revision", "f" * 40),
        ("sample", "path", "data/other.parquet"),
        ("sample", "sha256", "f" * 64),
        ("sample", "source_revision", "f" * 40),
        ("sample", "training_sample_receipt_sha256", "f" * 64),
        ("sample", "expected_row_count", 25008),
        ("tokenizer", "repository", "other/tokenizer"),
        ("tokenizer", "revision", "f" * 40),
        ("training", "recipe_id", "other-recipe"),
        ("training", "recipe_sha256", "f" * 64),
        ("training", "max_sequence_length", 4096),
        ("runtime", "transformers", "4.57.7"),
        ("releases", "training_recipe_evidence", "f" * 40),
    ],
)
def test_rejects_cross_artifact_provenance_drift(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    payload = json.loads(PACKAGE_CONFIG.read_text())
    key = "input_producer_releases" if section == "releases" else section
    target = cast(dict[str, object], payload[key])
    target[field] = value
    path = tmp_path / "package.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_training_data_package_config(path)

    with pytest.raises(TrainingDataPackageError, match="provenance"):
        data_package.validate_input_provenance(
            config,
            recipe_path=RECIPE_CONFIG,
            sample_receipt_path=SAMPLE_RECEIPT,
        )


@pytest.mark.parametrize(
    "field",
    [
        "base_response_generation_allowed",
        "activation_extraction_allowed",
        "training_execution_allowed",
        "scientific_use",
        "starts_experiment_timer",
        "provider_execution_allowed",
        "scheduler_execution_allowed",
        "sae_feature_selection_allowed",
        "concept_scoring_allowed",
        "concept_selected_training_allowed",
    ],
)
def test_rejects_authority_expansion(tmp_path: Path, field: str) -> None:
    payload = _config()
    boundaries = payload["boundaries"]
    assert isinstance(boundaries, dict)
    boundaries[field] = True
    config_path = tmp_path / "package.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingDataPackageError, match=field):
        load_training_data_package_config(config_path)


@pytest.mark.parametrize(
    "section",
    ["root", "sample", "tokenizer", "runtime", "training", "output", "releases"],
)
def test_rejects_unknown_config_fields(tmp_path: Path, section: str) -> None:
    payload = _config()
    target = (
        payload
        if section == "root"
        else payload["input_producer_releases" if section == "releases" else section]
    )
    assert isinstance(target, dict)
    target["unexpected"] = True
    config_path = tmp_path / "package.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingDataPackageError, match="fields differ"):
        load_training_data_package_config(config_path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("root", "phase", "scientific_run", "phase"),
        ("sample", "revision", "latest", "revision"),
        ("sample", "sha256", "x" * 64, "sha256"),
        ("sample", "source_revision", "main", "source_revision"),
        ("sample", "path", "../../secret", "path"),
        ("sample", "identity_scheme", "row-number-v1", "identity_scheme"),
        ("tokenizer", "revision", "main", "revision"),
        ("runtime", "transformers", "latest", "runtime"),
        ("runtime", "pyarrow", "main", "runtime"),
        ("training", "recipe_sha256", "x" * 64, "recipe_sha256"),
        ("training", "chat_template_sha256", "0" * 64, "chat_template"),
        ("training", "overlength_policy", "truncate", "overlength_policy"),
        ("training", "truncation", True, "truncation"),
        ("training", "append_eos", False, "append_eos"),
        ("training", "loss_mask", "all_tokens", "loss_mask"),
        ("output", "format", "parquet", "output.format"),
        ("output", "entry_schema_version", 2, "entry_schema_version"),
        ("releases", "training_recipe_evidence", "latest", "release"),
    ],
)
def test_rejects_identity_or_semantic_drift(
    tmp_path: Path, section: str, field: str, value: object, message: str
) -> None:
    payload = _config()
    key = "input_producer_releases" if section == "releases" else section
    target = payload if section == "root" else payload[key]
    assert isinstance(target, dict)
    target[field] = value
    config_path = tmp_path / "package.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingDataPackageError, match=message):
        load_training_data_package_config(config_path)


def test_rejects_tokenizer_file_set_drift(tmp_path: Path) -> None:
    payload = _config()
    tokenizer = payload["tokenizer"]
    assert isinstance(tokenizer, dict)
    files = cast(dict[str, object], tokenizer["files"])
    files["special_tokens_map.json"] = "b" * 64
    config_path = tmp_path / "package.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingDataPackageError, match=r"tokenizer\.files"):
        load_training_data_package_config(config_path)


def _write_source_shaped_sample(
    path: Path, *, orig_indices: tuple[int, int] = (9, 4)
) -> str:
    message = pa.struct(
        [pa.field("content", pa.string()), pa.field("role", pa.string())]
    )
    messages = pa.array(
        [
            [
                {"content": "question", "role": "user"},
                {"content": "answer", "role": "assistant"},
            ],
            [
                {"content": "", "role": "user"},
                {"content": "context", "role": "assistant"},
                {"content": "prompt", "role": "user"},
                {"content": "reply", "role": "assistant"},
            ],
        ],
        type=pa.list_(pa.field("element", message)),
    )
    table = _TABLE(
        {
            "messages": messages,
            "dataset_source": ["source-b", "source-a"],
            "orig_idx": pa.array(orig_indices, type=pa.int64()),
            "split": pa.array([1, 1], type=pa.int64()),
        }
    )
    _WRITE_TABLE(table, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rejects_negative_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_fixture_provenance(monkeypatch)
    sample_path = tmp_path / "train.parquet"
    payload = _config()
    sample = cast(dict[str, object], payload["sample"])
    sample["sha256"] = _write_source_shaped_sample(sample_path, orig_indices=(-1, 4))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(data_package, "verify_executor_revision", _identity)

    with pytest.raises(TrainingDataPackageError, match="orig_idx is invalid"):
        data_package.materialize_training_data_package(
            load_training_data_package_config(config_path),
            training_recipe_path=tmp_path / "recipe.json",
            training_sample_receipt_path=tmp_path / "sample-receipt.json",
            sample_path=sample_path,
            tokenizer_snapshot=tmp_path / "unused-tokenizer",
            code_revision="b" * 40,
            package_path=tmp_path / "package.jsonl",
            receipt_path=tmp_path / "receipt.json",
        )


def test_pair_publication_retains_exact_orphan_and_retries_after_receipt_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"

    def fail_receipt(path: Path, value: bytes, label: str) -> object:
        if label == "receipt":
            raise TrainingDataPackageError("injected receipt failure")
        return _PUBLISH_OR_MATCH(path, value, label)

    monkeypatch.setattr(data_package, "_publish_or_match", fail_receipt)
    with pytest.raises(TrainingDataPackageError, match="injected receipt failure"):
        _PUBLISH_EVIDENCE(package_path, b"package\n", receipt_path, b"receipt\n")
    assert package_path.read_bytes() == b"package\n"
    assert not receipt_path.exists()

    monkeypatch.setattr(data_package, "_publish_or_match", _PUBLISH_OR_MATCH)
    _PUBLISH_EVIDENCE(package_path, b"package\n", receipt_path, b"receipt\n")
    assert package_path.read_bytes() == b"package\n"
    assert receipt_path.read_bytes() == b"receipt\n"


def test_pair_publication_does_not_overwrite_racing_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"
    original_link = os.link

    def race_link(source: Path, destination: Path) -> None:
        package_path.write_bytes(b"racer\n")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", race_link)
    with pytest.raises(TrainingDataPackageError, match="existing package bytes differ"):
        _PUBLISH_EVIDENCE(package_path, b"package\n", receipt_path, b"receipt\n")
    assert package_path.read_bytes() == b"racer\n"
    assert not receipt_path.exists()


def test_pair_publication_keeps_package_for_competing_regular_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"

    def competing_receipt(path: Path, value: bytes, label: str) -> object:
        if label == "receipt":
            receipt_path.write_bytes(b"competing receipt\n")
        return _PUBLISH_OR_MATCH(path, value, label)

    monkeypatch.setattr(data_package, "_publish_or_match", competing_receipt)
    with pytest.raises(TrainingDataPackageError, match="existing receipt bytes differ"):
        _PUBLISH_EVIDENCE(package_path, b"package\n", receipt_path, b"receipt\n")
    assert package_path.read_bytes() == b"package\n"
    assert receipt_path.read_bytes() == b"competing receipt\n"


def test_pair_publication_retains_receipt_after_package_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"

    def replace_package(path: Path, value: bytes, label: str) -> object:
        if label == "receipt":
            package_path.unlink()
            package_path.write_bytes(b"racer\n")
        return _PUBLISH_OR_MATCH(path, value, label)

    monkeypatch.setattr(data_package, "_publish_or_match", replace_package)
    with pytest.raises(TrainingDataPackageError, match="package path identity"):
        _PUBLISH_EVIDENCE(package_path, b"package\n", receipt_path, b"receipt\n")
    assert package_path.read_bytes() == b"racer\n"
    assert receipt_path.read_bytes() == b"receipt\n"


@pytest.mark.parametrize("mutated_role", ["package", "receipt"])
def test_pair_publication_rejects_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutated_role: str
) -> None:
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"

    def mutate_after_receipt(path: Path, value: bytes, label: str) -> object:
        published = _PUBLISH_OR_MATCH(path, value, label)
        if label == "receipt":
            target = package_path if mutated_role == "package" else receipt_path
            target.write_bytes(b"racer\n")
        return published

    monkeypatch.setattr(data_package, "_publish_or_match", mutate_after_receipt)
    with pytest.raises(TrainingDataPackageError, match="bytes changed during"):
        _PUBLISH_EVIDENCE(package_path, b"package\n", receipt_path, b"receipt\n")
    mutated = package_path if mutated_role == "package" else receipt_path
    assert mutated.read_bytes() == b"racer\n"


@pytest.mark.parametrize(
    ("role", "alias_kind"),
    [
        ("package", "symlink"),
        ("package", "hardlink"),
        ("receipt", "symlink"),
        ("receipt", "hardlink"),
    ],
)
def test_pair_publication_rejects_aliased_output(
    tmp_path: Path, role: str, alias_kind: str
) -> None:
    value = b"package\n" if role == "package" else b"receipt\n"
    external = tmp_path / "external"
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"
    output = package_path if role == "package" else receipt_path
    external.write_bytes(value)
    if alias_kind == "symlink":
        output.symlink_to(external)
    else:
        os.link(external, output)

    with pytest.raises(TrainingDataPackageError, match="isolated regular file"):
        _PUBLISH_EVIDENCE(package_path, b"package\n", receipt_path, b"receipt\n")
    assert external.read_bytes() == value
    if role == "receipt":
        assert package_path.read_bytes() == b"package\n"


def test_materializes_content_free_join_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_fixture_provenance(monkeypatch)
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    _write_tokenizer_snapshot(snapshot)
    sample_path = tmp_path / "train.parquet"
    payload = _config()
    sample = payload["sample"]
    assert isinstance(sample, dict)
    sample["sha256"] = _write_source_shaped_sample(sample_path)
    training = payload["training"]
    assert isinstance(training, dict)
    training["max_sequence_length"] = 6
    _bind_tokenizer(payload, snapshot)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"
    revision = "b" * 40
    monkeypatch.setattr(data_package, "verify_executor_revision", _identity)

    receipt = data_package.materialize_training_data_package(
        load_training_data_package_config(config_path),
        training_recipe_path=tmp_path / "recipe.json",
        training_sample_receipt_path=tmp_path / "sample-receipt.json",
        sample_path=sample_path,
        tokenizer_snapshot=snapshot,
        code_revision=revision,
        package_path=package_path,
        receipt_path=receipt_path,
    )

    entries = [json.loads(line) for line in package_path.read_text().splitlines()]
    package_bytes = package_path.read_bytes()
    assert [entry["orig_idx"] for entry in entries] == [9, 4]
    assert [entry["assistant_turn_count"] for entry in entries] == [1, 2]
    assert [entry["full_data_member"] for entry in entries] == [True, False]
    assert entries[1]["prompt_messages_sha256"] == canonical_sha256(
        [
            {"content": "", "role": "user"},
            {"content": "context", "role": "assistant"},
            {"content": "prompt", "role": "user"},
        ]
    )
    assert entries[1]["supplied_response_sha256"] == canonical_sha256(
        {"content": "reply", "role": "assistant"}
    )
    assert b"question" not in package_bytes and b"answer" not in package_bytes
    assert receipt.row_count == 2
    assert receipt.retained_row_count == 1
    assert receipt.multi_turn_row_count == 1
    assert receipt.empty_initial_user_count == 1
    assert (
        receipt.payload["retained_input_token_count"] == entries[0]["input_token_count"]
    )
    assert (
        receipt.payload["retained_supervised_token_count"]
        == entries[0]["supervised_token_count"]
    )
    assert receipt.package_sha256 == hashlib.sha256(package_bytes).hexdigest()
    assert receipt.releases["executor_release"] == revision
    assert receipt.boundaries["scientific_use"] is False
    assert json.loads(receipt_path.read_text())["receipt_sha256"] == (
        receipt.receipt_sha256
    )
    example = load_bound_training_example(
        load_training_data_package_config(config_path),
        sample_path=sample_path,
        tokenizer_snapshot=snapshot,
        package_path=package_path,
        receipt_path=receipt_path,
        row_ordinal=0,
    )
    assert example.entry == entries[0]
    assert len(example.input_ids) == entries[0]["input_token_count"]
    assert (
        sum(value != -100 for value in example.labels)
        == entries[0]["supervised_token_count"]
    )
    assert (
        data_package.verify_training_data_package(
            load_training_data_package_config(config_path),
            training_recipe_path=tmp_path / "recipe.json",
            training_sample_receipt_path=tmp_path / "sample-receipt.json",
            code_revision=revision,
            sample_path=sample_path,
            tokenizer_snapshot=snapshot,
            package_path=package_path,
            receipt_path=receipt_path,
        )
        == receipt
    )
    receipt_bytes = receipt_path.read_bytes()
    for role, alias_kind in (
        ("package", "symlink"),
        ("package", "hardlink"),
        ("receipt", "symlink"),
        ("receipt", "hardlink"),
    ):
        output = package_path if role == "package" else receipt_path
        value = package_bytes if role == "package" else receipt_bytes
        external = tmp_path / f"verify-{role}-{alias_kind}"
        _replace_with_alias(output, external, value, alias_kind)
        with pytest.raises(TrainingDataPackageError, match="isolated regular file"):
            data_package.verify_training_data_package(
                load_training_data_package_config(config_path),
                training_recipe_path=tmp_path / "recipe.json",
                training_sample_receipt_path=tmp_path / "sample-receipt.json",
                code_revision=revision,
                sample_path=sample_path,
                tokenizer_snapshot=snapshot,
                package_path=package_path,
                receipt_path=receipt_path,
            )
        output.unlink()
        external.unlink()
        output.write_bytes(value)

    for mutated_path in (package_path, receipt_path):

        def mutate_after_validation(
            checked_config: TrainingDataPackageConfig,
            payload: dict[str, object],
            build: object,
            expected_revision: str,
            mutated_output: Path = mutated_path,
        ) -> None:
            _VALIDATE_RECEIPT_BUILD(checked_config, payload, build, expected_revision)
            mutated_output.write_bytes(b"racer\n")

        monkeypatch.setattr(
            data_package, "_validate_receipt_build", mutate_after_validation
        )
        with pytest.raises(
            TrainingDataPackageError, match="changed during verification"
        ):
            data_package.verify_training_data_package(
                load_training_data_package_config(config_path),
                training_recipe_path=tmp_path / "recipe.json",
                training_sample_receipt_path=tmp_path / "sample-receipt.json",
                code_revision=revision,
                sample_path=sample_path,
                tokenizer_snapshot=snapshot,
                package_path=package_path,
                receipt_path=receipt_path,
            )
        mutated_path.write_bytes(
            package_bytes if mutated_path == package_path else receipt_bytes
        )
    monkeypatch.setattr(
        data_package, "_validate_receipt_build", _VALIDATE_RECEIPT_BUILD
    )
    receipt_path.unlink()
    recovered = data_package.materialize_training_data_package(
        load_training_data_package_config(config_path),
        training_recipe_path=tmp_path / "recipe.json",
        training_sample_receipt_path=tmp_path / "sample-receipt.json",
        sample_path=sample_path,
        tokenizer_snapshot=snapshot,
        code_revision=revision,
        package_path=package_path,
        receipt_path=receipt_path,
    )
    assert recovered == receipt
    stored = json.loads(receipt_path.read_text())
    stored.pop("receipt_sha256")
    releases = stored["releases"]
    assert isinstance(releases, dict)
    substituted_revision = "e" * 40
    releases["executor_release"] = substituted_revision
    releases["coordinator_release"] = substituted_revision
    releases["output_release"] = canonical_sha256(
        {
            "executor_release": substituted_revision,
            "config_sha256": stored["config_sha256"],
            "package_sha256": stored["package_sha256"],
            "ordered_entry_sha256": stored["ordered_entry_sha256"],
        }
    )
    stored["receipt_sha256"] = canonical_sha256(stored)
    receipt_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(TrainingDataPackageError, match="executor release differs"):
        data_package.verify_training_data_package(
            load_training_data_package_config(config_path),
            training_recipe_path=tmp_path / "recipe.json",
            training_sample_receipt_path=tmp_path / "sample-receipt.json",
            code_revision=revision,
            sample_path=sample_path,
            tokenizer_snapshot=snapshot,
            package_path=package_path,
            receipt_path=receipt_path,
        )


def test_rejects_self_consistent_package_source_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_fixture_provenance(monkeypatch)
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    _write_tokenizer_snapshot(snapshot)
    sample_path = tmp_path / "train.parquet"
    payload = _config()
    sample = cast(dict[str, object], payload["sample"])
    sample["sha256"] = _write_source_shaped_sample(sample_path)
    _bind_tokenizer(payload, snapshot)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(data_package, "verify_executor_revision", _identity)
    config = load_training_data_package_config(config_path)
    data_package.materialize_training_data_package(
        config,
        training_recipe_path=tmp_path / "recipe.json",
        training_sample_receipt_path=tmp_path / "sample-receipt.json",
        sample_path=sample_path,
        tokenizer_snapshot=snapshot,
        code_revision="d" * 40,
        package_path=package_path,
        receipt_path=receipt_path,
    )
    entries = [json.loads(line) for line in package_path.read_text().splitlines()]
    entries[0]["dataset_source"] = "tampered-source"
    package_bytes = b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)
    package_path.write_bytes(package_bytes)
    stored = json.loads(receipt_path.read_text())
    stored.pop("receipt_sha256")
    stored["package_sha256"] = hashlib.sha256(package_bytes).hexdigest()
    stored["ordered_entry_sha256"] = canonical_sha256(entries)
    releases = cast(dict[str, object], stored["releases"])
    releases["output_release"] = canonical_sha256(
        {
            "executor_release": releases["executor_release"],
            "config_sha256": stored["config_sha256"],
            "package_sha256": stored["package_sha256"],
            "ordered_entry_sha256": stored["ordered_entry_sha256"],
        }
    )
    stored["receipt_sha256"] = canonical_sha256(stored)
    receipt_path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(TrainingDataPackageError, match="bound source replay"):
        data_package.verify_training_data_package(
            config,
            training_recipe_path=tmp_path / "recipe.json",
            training_sample_receipt_path=tmp_path / "sample-receipt.json",
            code_revision="d" * 40,
            sample_path=sample_path,
            tokenizer_snapshot=snapshot,
            package_path=package_path,
            receipt_path=receipt_path,
        )


def _write_tokenizer_snapshot(path: Path) -> None:
    tokenizers = cast(Any, importlib.import_module("tokenizers"))
    transformers = cast(Any, importlib.import_module("transformers"))
    vocabulary = {
        token: index
        for index, token in enumerate(
            ("<unk>", "<bos>", "<eos>", "<pad>", "user", "assistant")
        )
    }
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocabulary, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(path)


def _bind_tokenizer(payload: dict[str, object], snapshot: Path) -> None:
    tokenizer = payload["tokenizer"]
    assert isinstance(tokenizer, dict)
    files = cast(dict[str, object], tokenizer["files"])
    for name in files:
        files[name] = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    for field, distribution in (
        ("huggingface_hub", "huggingface-hub"),
        ("tokenizers", "tokenizers"),
        ("transformers", "transformers"),
    ):
        runtime[field] = importlib.metadata.version(distribution)


def test_loads_tokenizer_only_from_bound_snapshot_files(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    _write_tokenizer_snapshot(snapshot)
    payload = _config()
    _bind_tokenizer(payload, snapshot)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = data_package.load_bound_tokenizer(
        load_training_data_package_config(config_path), snapshot
    )

    assert loaded.eos_token_id is not None
    assert len(loaded) == 6


def test_cli_reports_hash_valid_malformed_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _allow_fixture_provenance(monkeypatch)
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    sample_path = tmp_path / "train.parquet"
    payload = _config()
    sample = cast(dict[str, object], payload["sample"])
    sample["sha256"] = _write_source_shaped_sample(sample_path)
    _bind_tokenizer(payload, snapshot)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(data_package, "verify_executor_revision", _identity)

    exit_code = main(
        [
            "materialize-training-data-package",
            str(config_path),
            "--training-recipe",
            str(tmp_path / "recipe.json"),
            "--training-sample-receipt",
            str(tmp_path / "sample-receipt.json"),
            "--parquet",
            str(sample_path),
            "--tokenizer-snapshot",
            str(snapshot),
            "--code-revision",
            "c" * 40,
            "--package",
            str(tmp_path / "package.jsonl"),
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert exit_code == 2
    assert "cannot load bound tokenizer" in capsys.readouterr().err


def test_cli_rejects_fifo_output_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _allow_fixture_provenance(monkeypatch)
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    _write_tokenizer_snapshot(snapshot)
    sample_path = tmp_path / "train.parquet"
    payload = _config()
    sample = cast(dict[str, object], payload["sample"])
    sample["sha256"] = _write_source_shaped_sample(sample_path)
    _bind_tokenizer(payload, snapshot)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"
    os.mkfifo(package_path)
    monkeypatch.setattr(data_package, "verify_executor_revision", _identity)

    exit_code = main(
        [
            "materialize-training-data-package",
            str(config_path),
            "--training-recipe",
            str(tmp_path / "recipe.json"),
            "--training-sample-receipt",
            str(tmp_path / "sample-receipt.json"),
            "--parquet",
            str(sample_path),
            "--tokenizer-snapshot",
            str(snapshot),
            "--code-revision",
            "c" * 40,
            "--package",
            str(package_path),
            "--output",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert "isolated regular file" in capsys.readouterr().err
    assert not receipt_path.exists()


def test_cli_materializes_training_data_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _allow_fixture_provenance(monkeypatch)
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    _write_tokenizer_snapshot(snapshot)
    sample_path = tmp_path / "train.parquet"
    payload = _config()
    sample = payload["sample"]
    assert isinstance(sample, dict)
    sample["sha256"] = _write_source_shaped_sample(sample_path)
    _bind_tokenizer(payload, snapshot)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    package_path = tmp_path / "package.jsonl"
    receipt_path = tmp_path / "receipt.json"
    revision = "c" * 40
    monkeypatch.setattr(data_package, "verify_executor_revision", _identity)

    exit_code = main(
        [
            "materialize-training-data-package",
            str(config_path),
            "--training-recipe",
            str(tmp_path / "recipe.json"),
            "--training-sample-receipt",
            str(tmp_path / "sample-receipt.json"),
            "--parquet",
            str(sample_path),
            "--tokenizer-snapshot",
            str(snapshot),
            "--code-revision",
            revision,
            "--package",
            str(package_path),
            "--output",
            str(receipt_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["row_count"] == 2
    assert summary["retained_row_count"] == 2
    assert summary["scientific_use"] is False
    assert package_path.exists() and receipt_path.exists()

    verify_exit = main(
        [
            "verify-training-data-package",
            str(config_path),
            "--training-recipe",
            str(tmp_path / "recipe.json"),
            "--training-sample-receipt",
            str(tmp_path / "sample-receipt.json"),
            "--parquet",
            str(sample_path),
            "--tokenizer-snapshot",
            str(snapshot),
            "--package",
            str(package_path),
            "--receipt",
            str(receipt_path),
            "--code-revision",
            revision,
        ]
    )

    verified = json.loads(capsys.readouterr().out)
    assert verify_exit == 0
    assert verified["receipt_sha256"] == summary["receipt_sha256"]
    assert verified["row_count"] == 2
