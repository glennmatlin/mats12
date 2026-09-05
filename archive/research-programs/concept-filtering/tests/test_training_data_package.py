from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import concept_filtering.training_data_package as data_package
from concept_filtering.chat_rendering import ChatRenderingError
from concept_filtering.cli import main
from concept_filtering.named_training_recipe import load_named_training_recipe
from concept_filtering.serialization import canonical_json_bytes, canonical_sha256
from concept_filtering.training_data_package import (
    TrainingDataPackageConfig,
    TrainingDataPackageError,
    TrainingDataPackageReceipt,
    load_training_data_package_config,
)

ROOT = Path(__file__).parents[1]
PACKAGE_CONFIG = ROOT / "configs/training/public-rank32-training-data-package-v1.json"
RECIPE_CONFIG = (
    ROOT / "configs/reconstruction/training/public-rank32-named-recipe-v1.json"
)
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
_ARRAY = cast(
    Callable[..., object],
    pa.array,  # pyright: ignore[reportUnknownMemberType]
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
_RENDER_TOKENS = cast(
    Callable[
        [TrainingDataPackageConfig, object, list[dict[str, str]], int],
        tuple[list[int], list[int], int],
    ],
    getattr(data_package, "_render_tokens"),  # noqa: B009
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


def _fixture_boundaries() -> dict[str, bool]:
    closed = (
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
    )
    return {
        "training_data_materialization_allowed": True,
        **dict.fromkeys(closed, False),
    }


def _fixture_sample() -> dict[str, object]:
    return {
        "repository": "owner/sample",
        "revision": "1" * 40,
        "path": "data/train.parquet",
        "sha256": "2" * 64,
        "source_revision": "3" * 40,
        "training_sample_receipt_sha256": "4" * 64,
        "expected_row_count": 2,
        "expected_split": 1,
        "identity_scheme": "source-revision-orig-idx-v1",
    }


def _fixture_training() -> dict[str, object]:
    template = _template()
    return {
        "recipe_id": "fixture-recipe-v1",
        "recipe_core_sha256": "8" * 64,
        "chat_template": template,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "max_sequence_length": 512,
        "overlength_policy": "drop",
        "truncation": False,
        "append_eos": True,
        "loss_mask": "assistant_tokens_including_terminal_eos",
    }


def _fixture_tokenizer() -> dict[str, object]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
    )
    return {
        "repository": "owner/tokenizer",
        "revision": "5" * 40,
        "path": "split-1",
        "files": {name: str(index + 6)[-1] * 64 for index, name in enumerate(names)},
    }


def _fixture_runtime() -> dict[str, str]:
    return {
        "huggingface_hub": "0.36.2",
        "pyarrow": "23.0.1",
        "tokenizers": "0.22.2",
        "transformers": "4.57.6",
    }


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "package_id": "fixture-training-data-package-v1",
        "phase": "pre_feature_training_data_materialization",
        "boundaries": _fixture_boundaries(),
        "sample": _fixture_sample(),
        "tokenizer": _fixture_tokenizer(),
        "runtime": _fixture_runtime(),
        "training": _fixture_training(),
        "output": {"format": "canonical-jsonl-v1", "entry_schema_version": 1},
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


def test_receipt_payload_and_nested_mappings_are_immutable() -> None:
    receipt = TrainingDataPackageReceipt(
        {
            "boundaries": {"base_response_generation_allowed": False},
            "releases": {"executor_release": "a" * 40},
        },
        "b" * 64,
    )
    with pytest.raises(TypeError):
        cast(dict[str, object], receipt.payload)["boundaries"] = {}
    with pytest.raises(TypeError):
        cast(dict[str, bool], receipt.boundaries)[
            "base_response_generation_allowed"
        ] = True
    copied = receipt.to_dict()
    cast(dict[str, bool], copied["boundaries"])["base_response_generation_allowed"] = (
        True
    )
    assert receipt.boundaries["base_response_generation_allowed"] is False


def test_checked_in_package_config_matches_frozen_inputs() -> None:
    config = load_training_data_package_config(PACKAGE_CONFIG)
    recipe = load_named_training_recipe(RECIPE_CONFIG)
    sample_receipt = json.loads(SAMPLE_RECEIPT.read_text())

    assert config.training.recipe_core_sha256 == recipe.recipe_core_sha256
    assert (
        config.training.chat_template_sha256
        == recipe.values.sequence.chat_template_sha256
    )
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
        ("tokenizer", "path", "other"),
        ("training", "recipe_id", "other-recipe"),
        ("training", "recipe_core_sha256", "f" * 64),
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
        ("training", "recipe_core_sha256", "x" * 64, "recipe_core_sha256"),
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


def test_render_failure_identifies_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    def fail(*_args: object, **_options: object) -> object:
        raise ChatRenderingError("fixture failure")

    monkeypatch.setattr(data_package, "render_assistant_only", fail)
    with pytest.raises(TrainingDataPackageError, match=r"row 17:.*fixture failure"):
        _RENDER_TOKENS(
            load_training_data_package_config(path), object(), _source_messages()[0], 17
        )


def test_rejects_tokenizer_file_set_drift(tmp_path: Path) -> None:
    payload = _config()
    tokenizer = payload["tokenizer"]
    assert isinstance(tokenizer, dict)
    files = cast(dict[str, object], tokenizer["files"])
    files["unexpected.json"] = "f" * 64
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
    messages = _ARRAY(_source_messages(), type=pa.list_(pa.field("element", message)))
    table = _TABLE(
        {
            "messages": messages,
            "dataset_source": ["source-b", "source-a"],
            "orig_idx": _ARRAY(orig_indices, type=pa.int64()),
            "split": _ARRAY([1, 1], type=pa.int64()),
        }
    )
    _WRITE_TABLE(table, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_messages() -> list[list[dict[str, str]]]:
    return [
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
    ]


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


@dataclass(frozen=True)
class PackageFixture:
    config: TrainingDataPackageConfig
    root: Path
    sample: Path
    tokenizer: Path
    package: Path
    receipt: Path
    revision: str


def _package_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, max_length: int = 512
) -> PackageFixture:
    _allow_fixture_provenance(monkeypatch)
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    snapshot = tmp_path / "tokenizer"
    snapshot.mkdir()
    _write_tokenizer_snapshot(snapshot)
    sample_path = tmp_path / "train.parquet"
    payload = _config()
    cast(dict[str, object], payload["sample"])["sha256"] = _write_source_shaped_sample(
        sample_path
    )
    cast(dict[str, object], payload["training"])["max_sequence_length"] = max_length
    _bind_tokenizer(payload, snapshot)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(data_package, "verify_executor_revision", _identity)
    return PackageFixture(
        load_training_data_package_config(config_path),
        tmp_path,
        sample_path,
        snapshot,
        tmp_path / "package.jsonl",
        tmp_path / "receipt.json",
        "b" * 40,
    )


def _materialize(fixture: PackageFixture) -> TrainingDataPackageReceipt:
    return data_package.materialize_training_data_package(
        fixture.config,
        training_recipe_path=fixture.root / "recipe.json",
        training_sample_receipt_path=fixture.root / "sample-receipt.json",
        sample_path=fixture.sample,
        tokenizer_snapshot=fixture.tokenizer,
        code_revision=fixture.revision,
        package_path=fixture.package,
        receipt_path=fixture.receipt,
    )


def _verify(fixture: PackageFixture) -> TrainingDataPackageReceipt:
    return data_package.verify_training_data_package(
        fixture.config,
        training_recipe_path=fixture.root / "recipe.json",
        training_sample_receipt_path=fixture.root / "sample-receipt.json",
        code_revision=fixture.revision,
        sample_path=fixture.sample,
        tokenizer_snapshot=fixture.tokenizer,
        package_path=fixture.package,
        receipt_path=fixture.receipt,
    )


def _entries(fixture: PackageFixture) -> list[dict[str, object]]:
    return [json.loads(line) for line in fixture.package.read_text().splitlines()]


def _assert_package_counts(
    receipt: TrainingDataPackageReceipt, entries: list[dict[str, object]]
) -> None:
    assert receipt.row_count == 2
    assert receipt.retained_row_count == 1
    assert receipt.multi_turn_row_count == 1
    assert receipt.empty_initial_user_count == 1
    assert (
        receipt.payload["retained_input_token_count"] == entries[0]["input_token_count"]
    )
    assert (
        receipt.payload["retained_supervised_token_count"]
        == (entries[0]["supervised_token_count"])
    )


def test_materializes_content_free_join_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch, max_length=6)
    receipt = _materialize(fixture)
    entries = _entries(fixture)
    package_bytes = fixture.package.read_bytes()
    assert [entry["orig_idx"] for entry in entries] == [9, 4]
    assert [entry["assistant_turn_count"] for entry in entries] == [1, 2]
    assert [entry["full_data_member"] for entry in entries] == [True, False]
    assert b"question" not in package_bytes and b"answer" not in package_bytes
    _assert_package_counts(receipt, entries)
    assert receipt.package_sha256 == hashlib.sha256(package_bytes).hexdigest()
    assert receipt.releases["executor_release"] == fixture.revision
    assert receipt.boundaries["scientific_use"] is False
    assert (
        json.loads(fixture.receipt.read_text())["receipt_sha256"]
        == receipt.receipt_sha256
    )


def test_hashes_prompt_and_supplied_response_without_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch, max_length=6)
    _materialize(fixture)
    entry = _entries(fixture)[1]
    prompt = _source_messages()[1][:-1]
    assert entry["prompt_messages_sha256"] == canonical_sha256(prompt)
    assert entry["supplied_response_sha256"] == canonical_sha256(
        _source_messages()[1][-1]
    )


def test_verifies_materialized_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch, max_length=6)
    receipt = _materialize(fixture)
    assert _verify(fixture) == receipt


@pytest.mark.parametrize(
    ("role", "alias_kind"),
    [
        (role, kind)
        for role in ("package", "receipt")
        for kind in ("symlink", "hardlink")
    ],
)
def test_verification_rejects_aliased_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, alias_kind: str
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch)
    _materialize(fixture)
    output = fixture.package if role == "package" else fixture.receipt
    external = tmp_path / f"verify-{role}-{alias_kind}"
    _replace_with_alias(output, external, output.read_bytes(), alias_kind)
    with pytest.raises(TrainingDataPackageError, match="isolated regular file"):
        _verify(fixture)


@pytest.mark.parametrize("role", ["package", "receipt"])
def test_verification_rejects_evidence_changed_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch)
    _materialize(fixture)
    mutated = fixture.package if role == "package" else fixture.receipt

    def mutate(
        config: TrainingDataPackageConfig,
        payload: dict[str, object],
        build: object,
        revision: str,
    ) -> None:
        _VALIDATE_RECEIPT_BUILD(config, payload, build, revision)
        mutated.write_bytes(b"racer\n")

    monkeypatch.setattr(data_package, "_validate_receipt_build", mutate)
    with pytest.raises(TrainingDataPackageError, match="changed during verification"):
        _verify(fixture)


def test_materialization_recovers_missing_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch)
    receipt = _materialize(fixture)
    fixture.receipt.unlink()
    assert _materialize(fixture) == receipt


def test_verification_rejects_rebound_executor_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch)
    _materialize(fixture)
    stored = json.loads(fixture.receipt.read_text())
    stored.pop("receipt_sha256")
    releases = cast(dict[str, object], stored["releases"])
    releases["executor_release"] = releases["coordinator_release"] = "e" * 40
    releases["output_release"] = _output_release(stored, "e" * 40)
    stored["receipt_sha256"] = canonical_sha256(stored)
    fixture.receipt.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(TrainingDataPackageError, match="executor release differs"):
        _verify(fixture)


def _output_release(stored: dict[str, object], revision: str) -> str:
    return canonical_sha256(
        {
            "executor_release": revision,
            "config_sha256": stored["config_sha256"],
            "package_sha256": stored["package_sha256"],
            "ordered_entry_sha256": stored["ordered_entry_sha256"],
        }
    )


def test_rejects_self_consistent_package_source_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch)
    _materialize(fixture)
    entries = _entries(fixture)
    entries[0]["dataset_source"] = "tampered-source"
    package_bytes = b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)
    fixture.package.write_bytes(package_bytes)
    _rewrite_receipt_for_package(fixture, entries, package_bytes)
    with pytest.raises(TrainingDataPackageError, match="bound source replay"):
        _verify(fixture)


def _rewrite_receipt_for_package(
    fixture: PackageFixture, entries: list[dict[str, object]], package_bytes: bytes
) -> None:
    stored = json.loads(fixture.receipt.read_text())
    stored.pop("receipt_sha256")
    stored["package_sha256"] = hashlib.sha256(package_bytes).hexdigest()
    stored["ordered_entry_sha256"] = canonical_sha256(entries)
    releases = cast(dict[str, object], stored["releases"])
    releases["output_release"] = _output_release(
        stored, cast(str, releases["executor_release"])
    )
    stored["receipt_sha256"] = canonical_sha256(stored)
    fixture.receipt.write_text(json.dumps(stored), encoding="utf-8")


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
    _write_tokenizer_extras(path)


def _write_tokenizer_extras(path: Path) -> None:
    extras = {
        "special_tokens_map.json": "{}\n",
        "chat_template.jinja": _template(),
        "merges.txt": "#version: 0.2\n",
        "vocab.json": "{}\n",
    }
    for name, value in extras.items():
        target = path / name
        if not target.exists():
            target.write_text(value, encoding="utf-8")


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


def _materialize_cli_args(fixture: PackageFixture) -> list[str]:
    return [
        "materialize-training-data-package",
        str(fixture.root / "config.json"),
        "--training-recipe",
        str(fixture.root / "recipe.json"),
        "--training-sample-receipt",
        str(fixture.root / "sample-receipt.json"),
        "--parquet",
        str(fixture.sample),
        "--tokenizer-snapshot",
        str(fixture.tokenizer),
        "--code-revision",
        fixture.revision,
        "--package",
        str(fixture.package),
        "--output",
        str(fixture.receipt),
    ]


def _verify_cli_args(fixture: PackageFixture) -> list[str]:
    return [
        "verify-training-data-package",
        str(fixture.root / "config.json"),
        "--training-recipe",
        str(fixture.root / "recipe.json"),
        "--training-sample-receipt",
        str(fixture.root / "sample-receipt.json"),
        "--parquet",
        str(fixture.sample),
        "--tokenizer-snapshot",
        str(fixture.tokenizer),
        "--package",
        str(fixture.package),
        "--receipt",
        str(fixture.receipt),
        "--code-revision",
        fixture.revision,
    ]


def _malform_bound_tokenizer(fixture: PackageFixture) -> PackageFixture:
    target = fixture.tokenizer / "tokenizer.json"
    target.write_text("{}", encoding="utf-8")
    path = fixture.root / "config.json"
    payload = json.loads(path.read_text())
    tokenizer = cast(dict[str, object], payload["tokenizer"])
    files = cast(dict[str, object], tokenizer["files"])
    files["tokenizer.json"] = hashlib.sha256(target.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return PackageFixture(
        load_training_data_package_config(path),
        fixture.root,
        fixture.sample,
        fixture.tokenizer,
        fixture.package,
        fixture.receipt,
        fixture.revision,
    )


def test_cli_reports_hash_valid_malformed_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _malform_bound_tokenizer(_package_fixture(tmp_path, monkeypatch))
    assert main(_materialize_cli_args(fixture)) == 2
    assert "cannot load bound tokenizer" in capsys.readouterr().err


def test_cli_rejects_fifo_output_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch)
    os.mkfifo(fixture.package)
    assert main(_materialize_cli_args(fixture)) == 2
    assert "isolated regular file" in capsys.readouterr().err
    assert not fixture.receipt.exists()


def test_cli_materializes_training_data_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _package_fixture(tmp_path, monkeypatch)
    exit_code = main(_materialize_cli_args(fixture))
    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["row_count"] == summary["retained_row_count"] == 2
    assert summary["scientific_use"] is False
    assert fixture.package.exists() and fixture.receipt.exists()
    assert main(_verify_cli_args(fixture)) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["receipt_sha256"] == summary["receipt_sha256"]
    assert verified["row_count"] == 2
