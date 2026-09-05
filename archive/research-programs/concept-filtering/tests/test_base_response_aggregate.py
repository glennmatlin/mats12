from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from test_base_response_generation import VALID_CONFIG
from test_base_response_runner import EXAMPLES, RecordingBackend

from concept_filtering.base_response_aggregate import (
    BaseResponseAggregateError,
    base_response_status,
    finalize_base_responses,
    verify_base_responses,
)
from concept_filtering.base_response_generation import (
    BaseResponseConfig,
    BaseResponseEvidence,
    load_base_response_config,
)
from concept_filtering.base_response_runner import run_base_response_shard
from concept_filtering.serialization import canonical_sha256

EXECUTOR_REVISION = "b" * 40


def _evidence() -> BaseResponseEvidence:
    return BaseResponseEvidence(
        package_receipt_sha256="c" * 64,
        b2_receipt_sha256="1" * 64,
        package_payload={},
        b2_payload={},
    )


def _config(tmp_path: Path) -> BaseResponseConfig:
    path = tmp_path / "base-responses.json"
    path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")
    return load_base_response_config(path)


def _generate_all(config: BaseResponseConfig, tmp_path: Path) -> Path:
    output_dir = tmp_path / "responses"
    run_base_response_shard(
        config,
        EXAMPLES,
        shard_index=0,
        shard_count=1,
        output_dir=output_dir,
        backend=RecordingBackend(),
    )
    return output_dir


def test_status_reports_complete_missing_and_invalid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    missing_path = output_dir / f"{EXAMPLES[1].example_id}.json"
    missing_path.unlink()
    invalid_path = output_dir / f"{EXAMPLES[2].example_id}.json"
    invalid_path.write_text(invalid_path.read_text()[:40])

    report = base_response_status(config, EXAMPLES, output_dir)

    assert report.expected == 4
    assert report.complete == 2
    assert report.missing == 1
    assert report.invalid == 1
    assert report.eos_stopped == 2
    assert report.cap_stopped == 0


def test_status_does_not_write(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    before = sorted(path.name for path in output_dir.iterdir())

    base_response_status(config, EXAMPLES, output_dir)

    assert sorted(path.name for path in output_dir.iterdir()) == before


def test_status_counts_absent_directory_as_all_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)

    report = base_response_status(config, EXAMPLES, tmp_path / "never-created")

    assert report.expected == 4
    assert report.missing == 4
    assert report.complete == 0


def test_status_rejects_unkeyed_extra_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    (output_dir / "stray.json").write_text("{}")

    with pytest.raises(BaseResponseAggregateError, match="stray"):
        base_response_status(config, EXAMPLES, output_dir)


def test_status_counts_aliased_sample_as_invalid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    sample_path = output_dir / f"{EXAMPLES[0].example_id}.json"
    external = tmp_path / "external.json"
    sample_path.rename(external)
    sample_path.symlink_to(external)

    report = base_response_status(config, EXAMPLES, output_dir)

    assert report.complete == 3
    assert report.invalid == 1


def test_status_ignores_stage_leftovers(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    sample_name = f"{EXAMPLES[0].example_id}.json"
    (output_dir / f"{sample_name}.stage-999").write_text("partial")

    report = base_response_status(config, EXAMPLES, output_dir)

    assert report.complete == 4
    assert report.stage_leftovers == 1


def _finalize(config: BaseResponseConfig, tmp_path: Path, output_dir: Path):
    return finalize_base_responses(
        config,
        _evidence(),
        EXAMPLES,
        output_dir,
        tmp_path / "aggregate.json",
        executor_revision=EXECUTOR_REVISION,
    )


def test_finalize_publishes_signed_aggregate_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)

    receipt = _finalize(config, tmp_path, output_dir)

    written = cast(
        dict[str, object], json.loads((tmp_path / "aggregate.json").read_text())
    )
    assert written == receipt.to_dict()
    body = {
        field: value for field, value in written.items() if field != "receipt_sha256"
    }
    assert written["receipt_sha256"] == canonical_sha256(body)
    assert written["schema_version"] == 1
    assert written["generation_id"] == config.generation_id
    assert written["config_sha256"] == config.config_sha256
    assert written["package_receipt_sha256"] == "c" * 64
    assert written["b2_receipt_sha256"] == "1" * 64
    assert written["executor_revision"] == EXECUTOR_REVISION
    assert written["sample_count"] == 4
    assert written["eos_count"] == 4
    assert written["cap_count"] == 0
    assert written["prompt_token_total"] == 20
    assert written["completion_token_total"] == 12
    ordered = cast(str, written["ordered_sample_sha256"])
    assert len(ordered) == 64


def test_finalize_is_absent_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    _finalize(config, tmp_path, output_dir)

    with pytest.raises(BaseResponseAggregateError, match="refus"):
        _finalize(config, tmp_path, output_dir)


def test_finalize_requires_exact_identity_coverage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    (output_dir / f"{EXAMPLES[1].example_id}.json").unlink()

    with pytest.raises(BaseResponseAggregateError, match="missing"):
        _finalize(config, tmp_path, output_dir)
    assert not (tmp_path / "aggregate.json").exists()


def test_finalize_rejects_invalid_sample(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    invalid = output_dir / f"{EXAMPLES[2].example_id}.json"
    invalid.write_text(invalid.read_text()[:40])

    with pytest.raises(BaseResponseAggregateError, match="cannot parse"):
        _finalize(config, tmp_path, output_dir)


def test_finalize_rejects_unkeyed_extra_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    (output_dir / "stray.json").write_text("{}")

    with pytest.raises(BaseResponseAggregateError, match="stray"):
        _finalize(config, tmp_path, output_dir)


def test_finalize_rejects_invalid_executor_revision(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)

    with pytest.raises(BaseResponseAggregateError, match="executor"):
        finalize_base_responses(
            config,
            _evidence(),
            EXAMPLES,
            output_dir,
            tmp_path / "aggregate.json",
            executor_revision="not-a-revision",
        )


def test_verify_replays_aggregate_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    receipt = _finalize(config, tmp_path, output_dir)

    verified = verify_base_responses(
        config, _evidence(), EXAMPLES, output_dir, tmp_path / "aggregate.json"
    )

    assert verified.receipt_sha256 == receipt.receipt_sha256


def test_verify_rejects_sample_tampered_after_finalize(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    _finalize(config, tmp_path, output_dir)
    sample_path = output_dir / f"{EXAMPLES[3].example_id}.json"
    payload = cast(dict[str, object], json.loads(sample_path.read_text()))
    body = {
        field: value for field, value in payload.items() if field != "sample_sha256"
    }
    body["response_text"] = "revised"
    body["response_sha256"] = canonical_sha256("revised")
    sample_path.write_text(
        json.dumps({**body, "sample_sha256": canonical_sha256(body)})
    )

    with pytest.raises(BaseResponseAggregateError, match=r"response SHA-256|ordered"):
        verify_base_responses(
            config, _evidence(), EXAMPLES, output_dir, tmp_path / "aggregate.json"
        )


def test_verify_rejects_tampered_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    _finalize(config, tmp_path, output_dir)
    receipt_path = tmp_path / "aggregate.json"
    payload = cast(dict[str, object], json.loads(receipt_path.read_text()))
    body = {
        field: value for field, value in payload.items() if field != "receipt_sha256"
    }
    body["sample_count"] = 9
    receipt_path.write_text(
        json.dumps({**body, "receipt_sha256": canonical_sha256(body)})
    )

    with pytest.raises(BaseResponseAggregateError, match="sample_count"):
        verify_base_responses(config, _evidence(), EXAMPLES, output_dir, receipt_path)


def test_verify_rejects_receipt_for_other_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = _generate_all(config, tmp_path)
    _finalize(config, tmp_path, output_dir)
    other = BaseResponseEvidence(
        package_receipt_sha256="d" * 64,
        b2_receipt_sha256="1" * 64,
        package_payload={},
        b2_payload={},
    )

    with pytest.raises(BaseResponseAggregateError, match="package_receipt"):
        verify_base_responses(
            config, other, EXAMPLES, output_dir, tmp_path / "aggregate.json"
        )
