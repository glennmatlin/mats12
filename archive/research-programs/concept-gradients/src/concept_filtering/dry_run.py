"""Fixture-only end-to-end execution of reconstruction contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.divergence import (
    DivergenceReport,
    DivergenceValidationError,
    ReferenceSpec,
    compare_expectations,
    load_reference_spec,
    write_divergence_report,
)
from concept_filtering.evaluation import (
    EvaluationReceipt,
    EvaluationValidationError,
    EvaluatorConfig,
    build_evaluation_receipt,
    load_evaluator_config,
    write_evaluation_receipt,
)
from concept_filtering.manifest import (
    DatasetManifest,
    ManifestValidationError,
    build_manifest,
    write_manifest,
)
from concept_filtering.organism import (
    OrganismConfig,
    OrganismValidationError,
    load_organism_config,
)
from concept_filtering.serialization import bytes_sha256, canonical_sha256


class DryRunValidationError(ValueError):
    """Raised when a fixture bundle cannot complete without ambiguity."""


@dataclass(frozen=True)
class DryRunReceipt:
    schema_version: int
    mode: str
    scientific_use: bool
    code_revision: str
    bundle_sha256: str
    organism_sha256: str
    reference_spec_sha256: str
    manifest_sha256: str
    divergence_report_sha256: str
    evaluation_result_sha256: str
    artifact_receipt_count: int
    gate_r2_blocked: bool
    run_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "scientific_use": self.scientific_use,
            "code_revision": self.code_revision,
            "bundle_sha256": self.bundle_sha256,
            "organism_sha256": self.organism_sha256,
            "reference_spec_sha256": self.reference_spec_sha256,
            "manifest_sha256": self.manifest_sha256,
            "divergence_report_sha256": self.divergence_report_sha256,
            "evaluation_result_sha256": self.evaluation_result_sha256,
            "artifact_receipt_count": self.artifact_receipt_count,
            "gate_r2_blocked": self.gate_r2_blocked,
            "run_sha256": self.run_sha256,
        }


@dataclass(frozen=True)
class DryRunComponents:
    organism: OrganismConfig
    manifest: DatasetManifest
    divergence: DivergenceReport
    evaluation: EvaluationReceipt
    reference_spec_sha256: str


@dataclass(frozen=True)
class DryRunInputs:
    organism: OrganismConfig
    reference: ReferenceSpec
    manifest: DatasetManifest
    divergence: DivergenceReport
    evaluator: EvaluatorConfig


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DryRunValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DryRunValidationError(f"{field} must be a non-empty string")
    return value


def _bundle_path(bundle_path: Path, raw_path: str, field: str) -> Path:
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise DryRunValidationError(f"{field} must be a relative path")
    bundle_dir = bundle_path.parent.resolve()
    resolved = (bundle_path.parent / relative_path).resolve()
    if not resolved.is_relative_to(bundle_dir):
        raise DryRunValidationError(f"{field} escapes bundle directory")
    return resolved


def _load_bundle_payload(bundle_path: Path) -> tuple[dict[str, Any], str]:
    try:
        source_bytes = bundle_path.read_bytes()
        value: object = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DryRunValidationError(
            f"cannot load dry-run bundle {bundle_path}: {error}"
        ) from error
    return _mapping(value, "dry-run bundle"), bytes_sha256(source_bytes)


def _validate_bundle_envelope(payload: dict[str, Any]) -> None:
    allowed = {
        "schema_version",
        "mode",
        "organism_config",
        "reference_spec",
        "sample_jsonl",
        "manifest_id",
        "evaluator_config",
        "scores_jsonl",
        "expected_organism_sha256",
        "expected_manifest_sha256",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise DryRunValidationError(
            f"dry-run bundle has unexpected fields: {', '.join(unexpected)}"
        )
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise DryRunValidationError("schema_version must be the integer 1")
    mode = _required_string(payload, "mode")
    if mode != "fixture-non-scientific":
        raise DryRunValidationError("mode must be fixture-non-scientific")
    _required_string(payload, "manifest_id")
    _required_string(payload, "expected_organism_sha256")
    _required_string(payload, "expected_manifest_sha256")


def _bundle_paths(bundle_path: Path, payload: dict[str, Any]) -> dict[str, Path]:
    return {
        field: _bundle_path(bundle_path, _required_string(payload, field), field)
        for field in (
            "organism_config",
            "reference_spec",
            "sample_jsonl",
            "evaluator_config",
            "scores_jsonl",
        )
    }


def _load_bundle(bundle_path: Path) -> tuple[dict[str, Any], dict[str, Path], str]:
    payload, bundle_sha256 = _load_bundle_payload(bundle_path)
    _validate_bundle_envelope(payload)
    return payload, _bundle_paths(bundle_path, payload), bundle_sha256


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise DryRunValidationError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise DryRunValidationError(f"output directory must be empty: {path}")


def _build_components(
    payload: dict[str, Any], paths: dict[str, Path], *, code_revision: str
) -> DryRunComponents:
    try:
        inputs = _build_inputs(payload, paths)
        if inputs.evaluator.scientific_use:
            raise DryRunValidationError(
                "fixture dry run requires an evaluator with scientific_use=false"
            )
        evaluation = _build_fixture_evaluation(
            inputs, payload, paths, code_revision=code_revision
        )
    except (
        OrganismValidationError,
        DivergenceValidationError,
        ManifestValidationError,
        EvaluationValidationError,
    ) as error:
        raise DryRunValidationError(str(error)) from error
    return DryRunComponents(
        organism=inputs.organism,
        manifest=inputs.manifest,
        divergence=inputs.divergence,
        evaluation=evaluation,
        reference_spec_sha256=inputs.reference.spec_sha256,
    )


def _build_fixture_evaluation(
    inputs: DryRunInputs,
    payload: dict[str, Any],
    paths: dict[str, Path],
    *,
    code_revision: str,
) -> EvaluationReceipt:
    return build_evaluation_receipt(
        paths["scores_jsonl"],
        evaluator=inputs.evaluator,
        code_revision=code_revision,
        organism=inputs.organism,
        manifest=inputs.manifest,
        expected_organism_sha256=_required_string(payload, "expected_organism_sha256"),
        expected_manifest_sha256=_required_string(payload, "expected_manifest_sha256"),
    )


def _build_inputs(payload: dict[str, Any], paths: dict[str, Path]) -> DryRunInputs:
    organism = load_organism_config(paths["organism_config"])
    reference = load_reference_spec(paths["reference_spec"])
    manifest = build_manifest(
        paths["sample_jsonl"],
        manifest_id=_required_string(payload, "manifest_id"),
    )
    divergence = compare_expectations(
        reference.reference_id,
        organism.organism_id,
        list(reference.expectations),
        organism.parameters,
    )
    return DryRunInputs(
        organism=organism,
        reference=reference,
        manifest=manifest,
        divergence=divergence,
        evaluator=load_evaluator_config(paths["evaluator_config"]),
    )


def _build_run_receipt(
    components: DryRunComponents, *, code_revision: str, bundle_sha256: str
) -> DryRunReceipt:
    body = _run_body(
        components, code_revision=code_revision, bundle_sha256=bundle_sha256
    )
    return DryRunReceipt(
        schema_version=1,
        mode="fixture-non-scientific",
        scientific_use=False,
        code_revision=code_revision,
        bundle_sha256=bundle_sha256,
        organism_sha256=components.organism.organism_sha256,
        reference_spec_sha256=components.reference_spec_sha256,
        manifest_sha256=components.manifest.manifest_sha256,
        divergence_report_sha256=components.divergence.report_sha256,
        evaluation_result_sha256=components.evaluation.result_sha256,
        artifact_receipt_count=len(components.organism.artifacts),
        gate_r2_blocked=components.divergence.is_blocked,
        run_sha256=canonical_sha256(body),
    )


def _run_body(
    components: DryRunComponents, *, code_revision: str, bundle_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "fixture-non-scientific",
        "scientific_use": False,
        "code_revision": code_revision,
        "bundle_sha256": bundle_sha256,
        "organism_sha256": components.organism.organism_sha256,
        "reference_spec_sha256": components.reference_spec_sha256,
        "manifest_sha256": components.manifest.manifest_sha256,
        "divergence_report_sha256": components.divergence.report_sha256,
        "evaluation_result_sha256": components.evaluation.result_sha256,
        "artifact_receipt_count": len(components.organism.artifacts),
        "gate_r2_blocked": components.divergence.is_blocked,
    }


def _write_outputs(
    output_dir: Path, components: DryRunComponents, run: DryRunReceipt
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(components.manifest, output_dir / "dataset-manifest.json")
    write_divergence_report(
        components.divergence, output_dir / "divergence-report.json"
    )
    write_evaluation_receipt(
        components.evaluation, output_dir / "evaluation-receipt.json"
    )
    (output_dir / "run-receipt.json").write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute_dry_run(
    bundle_path: Path, *, code_revision: str, output_dir: Path
) -> DryRunReceipt:
    """Execute every offline contract without loading a model or running inference."""
    _prepare_output_directory(output_dir)
    payload, paths, bundle_sha256 = _load_bundle(bundle_path)
    components = _build_components(payload, paths, code_revision=code_revision)
    run = _build_run_receipt(
        components, code_revision=code_revision, bundle_sha256=bundle_sha256
    )
    _write_outputs(output_dir, components, run)
    return run
