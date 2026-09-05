"""Command-line interface for reconstruction receipts and offline gates."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from concept_filtering.activation_contract import (
    activation_contract_receipt,
    load_activation_contract,
    resolve_contract_example,
    write_activation_receipt,
)
from concept_filtering.activation_contract_transformers import (
    run_activation_contract,
)
from concept_filtering.backbone import (
    BackboneValidationError,
    execute_backbone,
    load_backbone_config,
    plan_backbone,
    status_backbone,
    verify_backbone,
)
from concept_filtering.base_response_aggregate import (
    BaseResponseAggregateError,
    base_response_status,
    finalize_base_responses,
    verify_base_responses,
)
from concept_filtering.base_response_generation import (
    BaseResponseConfig,
    BaseResponseEvidence,
    BaseResponseGenerationError,
    load_base_response_config,
    validate_base_response_evidence,
    verify_artifact_snapshot,
)
from concept_filtering.base_response_join import (
    BaseResponseJoinError,
    JoinedExample,
    load_joined_examples,
)
from concept_filtering.base_response_runner import (
    BaseResponseRunnerError,
    GeneratedResponse,
    run_base_response_shard,
)
from concept_filtering.base_response_transformers import BaseResponseBackendError
from concept_filtering.divergence import (
    DivergenceValidationError,
    compare_expectations,
    load_reference_spec,
    write_divergence_report,
)
from concept_filtering.dry_run import DryRunValidationError, execute_dry_run
from concept_filtering.evaluation import (
    EvaluationValidationError,
    build_evaluation_receipt,
    load_evaluator_config,
    write_evaluation_receipt,
)
from concept_filtering.evaluator_reconstruction import (
    EvaluatorReconstructionError,
    load_evaluator_reconstruction,
    verify_evaluator_reconstruction,
    write_evaluator_reconstruction_receipt,
)
from concept_filtering.generation_bank import (
    GenerationValidationError,
    execute_generation_bank,
    load_generation_plan,
)
from concept_filtering.inspect_archive import (
    InspectArchiveValidationError,
    extract_scores,
    load_extraction_receipt,
)
from concept_filtering.manifest import (
    ManifestValidationError,
    build_manifest,
    load_manifest,
    write_manifest,
)
from concept_filtering.named_training_recipe import (
    NamedTrainingRecipeError,
    load_named_training_recipe,
)
from concept_filtering.organism import OrganismValidationError, load_organism_config
from concept_filtering.prompt_bank import extract_prompt_bank, load_prompt_bank
from concept_filtering.r2_provider_scoring import (
    R2ProviderScoringError,
    build_anthropic_transport,
    build_r2_scoring_items,
    finalize_r2_gate,
    load_r2_owner_acceptance,
    score_r2_items,
)
from concept_filtering.receipts import ReceiptValidationError, load_artifact_receipt
from concept_filtering.sae_compatibility import (
    SAECompatibilityError,
    load_sae_compatibility,
    verify_sae_compatibility,
    write_sae_compatibility_receipt,
)
from concept_filtering.sae_functional_smoke import (
    SAEFunctionalSmokeError,
    execute_sae_functional_smoke,
    load_sae_functional_smoke,
)
from concept_filtering.sae_functional_transformers import run_transformers_sae_smoke
from concept_filtering.serialization import file_sha256
from concept_filtering.smoke import (
    SmokeValidationError,
    execute_smoke,
    load_smoke_plan,
    source_repository_root,
    verify_executor_revision,
)
from concept_filtering.smoke_transformers import (
    run_transformers_generation,
    run_transformers_smoke,
)
from concept_filtering.training_data_package import (
    TrainingDataPackageError,
    load_training_data_package_config,
    materialize_training_data_package,
    verify_training_data_package,
)
from concept_filtering.training_recipe_reconstruction import (
    TrainingRecipeReconstructionError,
    load_training_recipe_reconstruction,
    verify_training_recipe_reconstruction,
    write_training_recipe_receipt,
)
from concept_filtering.training_recovery_smoke import (
    TrainingRecoverySmokeError,
    execute_training_recovery_smoke,
)
from concept_filtering.training_recovery_transformers import (
    run_transformers_training_recovery,
)
from concept_filtering.training_sample_reconstruction import (
    TrainingSampleReconstructionError,
    load_training_sample_reconstruction,
    verify_training_sample_reconstruction,
    write_training_sample_receipt,
)

_USER_ERRORS = (
    BackboneValidationError,
    ReceiptValidationError,
    OrganismValidationError,
    ManifestValidationError,
    DivergenceValidationError,
    EvaluationValidationError,
    EvaluatorReconstructionError,
    DryRunValidationError,
    InspectArchiveValidationError,
    SmokeValidationError,
    GenerationValidationError,
    TrainingSampleReconstructionError,
    TrainingRecipeReconstructionError,
    SAECompatibilityError,
    SAEFunctionalSmokeError,
    NamedTrainingRecipeError,
    TrainingRecoverySmokeError,
    TrainingDataPackageError,
    BaseResponseGenerationError,
    BaseResponseJoinError,
    BaseResponseRunnerError,
    BaseResponseAggregateError,
    BaseResponseBackendError,
    R2ProviderScoringError,
    OSError,
)


class _Subparsers(Protocol):
    def add_parser(self, name: str) -> argparse.ArgumentParser: ...


def _add_validation_commands(subparsers: _Subparsers) -> None:
    receipt = subparsers.add_parser("validate-receipt")
    receipt.add_argument("path")
    organism = subparsers.add_parser("validate-organism")
    organism.add_argument("path")
    extraction = subparsers.add_parser("validate-inspect-extraction")
    extraction.add_argument("receipt")
    extraction.add_argument("--output", required=True)
    smoke = subparsers.add_parser("validate-smoke-plan")
    smoke.add_argument("path")


def _add_manifest_command(subparsers: _Subparsers) -> None:
    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("input")
    manifest.add_argument("--manifest-id", required=True)
    manifest.add_argument("--output", required=True)


def _add_compare_command(subparsers: _Subparsers) -> None:
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--organism", required=True)
    compare.add_argument("--output", required=True)


def _add_evaluate_command(subparsers: _Subparsers) -> None:
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--evaluator", required=True)
    evaluate.add_argument("--code-revision", required=True)
    evaluate.add_argument("--organism", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--sample", required=True)
    evaluate.add_argument("--expected-organism-sha256", required=True)
    evaluate.add_argument("--expected-manifest-sha256", required=True)
    evaluate.add_argument("--output", required=True)


def _add_evaluator_reconstruction_command(subparsers: _Subparsers) -> None:
    verify = subparsers.add_parser("verify-evaluator-reconstruction")
    verify.add_argument("config")
    verify.add_argument("--public-source", required=True)
    verify.add_argument(
        "--archive",
        nargs=2,
        action="append",
        required=True,
        metavar=("BEHAVIOR", "PATH"),
    )
    verify.add_argument("--output", required=True)


def _add_training_sample_reconstruction_command(subparsers: _Subparsers) -> None:
    verify = subparsers.add_parser("verify-training-sample-reconstruction")
    verify.add_argument("config")
    verify.add_argument("--parquet", required=True)
    verify.add_argument("--output", required=True)


def _add_training_recipe_reconstruction_command(subparsers: _Subparsers) -> None:
    verify = subparsers.add_parser("verify-training-recipe-reconstruction")
    verify.add_argument("config")
    verify.add_argument("--artifact-dir", required=True)
    verify.add_argument("--output", required=True)


def _add_training_recovery_smoke_command(subparsers: _Subparsers) -> None:
    smoke = subparsers.add_parser("run-training-recovery-smoke")
    smoke.add_argument("config")
    smoke.add_argument("--code-revision", required=True)
    smoke.add_argument("--base-snapshot", required=True)
    smoke.add_argument("--tokenizer-snapshot", required=True)
    smoke.add_argument("--fixture", required=True)
    smoke.add_argument("--work-dir", required=True)
    smoke.add_argument("--output", required=True)


def _add_training_data_package_command(subparsers: _Subparsers) -> None:
    package = subparsers.add_parser("materialize-training-data-package")
    package.add_argument("config")
    package.add_argument("--training-recipe", required=True)
    package.add_argument("--training-sample-receipt", required=True)
    package.add_argument("--parquet", required=True)
    package.add_argument("--tokenizer-snapshot", required=True)
    package.add_argument("--code-revision", required=True)
    package.add_argument("--package", required=True)
    package.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify-training-data-package")
    verify.add_argument("config")
    verify.add_argument("--training-recipe", required=True)
    verify.add_argument("--training-sample-receipt", required=True)
    verify.add_argument("--parquet", required=True)
    verify.add_argument("--tokenizer-snapshot", required=True)
    verify.add_argument("--code-revision", required=True)
    verify.add_argument("--package", required=True)
    verify.add_argument("--receipt", required=True)


def _add_sae_compatibility_command(subparsers: _Subparsers) -> None:
    verify = subparsers.add_parser("verify-sae-compatibility")
    verify.add_argument("config")
    for role in (
        "base-config",
        "sae-cfg",
        "runner-cfg",
        "release-readme",
        "weights-header",
        "saelens-docs",
        "artifact-receipt",
    ):
        verify.add_argument(f"--{role}", required=True)
    verify.add_argument("--output", required=True)


def _add_sae_functional_smoke_command(subparsers: _Subparsers) -> None:
    smoke = subparsers.add_parser("run-sae-functional-smoke")
    smoke.add_argument("config")
    smoke.add_argument("--code-revision", required=True)
    smoke.add_argument("--base-snapshot", required=True)
    smoke.add_argument("--adapter-snapshot", required=True)
    smoke.add_argument("--sae-directory", required=True)
    smoke.add_argument("--output", required=True)


def _add_dry_run_command(subparsers: _Subparsers) -> None:
    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("bundle")
    dry_run.add_argument("--code-revision", required=True)
    dry_run.add_argument("--output-dir", required=True)


def _add_extract_command(subparsers: _Subparsers) -> None:
    inspect_eval = subparsers.add_parser("extract-inspect-eval")
    inspect_eval.add_argument("archive")
    inspect_eval.add_argument("--scorer", default="rubric_scorer")
    inspect_eval.add_argument("--expected-sha256", required=True)
    inspect_eval.add_argument("--output", required=True)
    inspect_eval.add_argument("--receipt", required=True)
    prompt_bank = subparsers.add_parser("extract-prompt-bank")
    prompt_bank.add_argument("archive")
    prompt_bank.add_argument("--expected-sha256", required=True)
    prompt_bank.add_argument("--output", required=True)
    prompt_bank.add_argument("--receipt", required=True)


def _add_smoke_command(subparsers: _Subparsers) -> None:
    smoke = subparsers.add_parser("run-smoke")
    smoke.add_argument("path")
    smoke.add_argument("--code-revision", required=True)
    smoke.add_argument("--cache-dir", required=True)
    smoke.add_argument("--output", required=True)


def _add_generation_command(subparsers: _Subparsers) -> None:
    generation = subparsers.add_parser("run-generation-bank")
    generation.add_argument("path")
    generation.add_argument("--smoke-plan", required=True)
    generation.add_argument("--code-revision", required=True)
    generation.add_argument("--cache-dir", required=True)
    generation.add_argument(
        "--prompt-bank",
        nargs=2,
        action="append",
        required=True,
        metavar=("RECEIPT", "OUTPUT"),
    )
    generation.add_argument("--limit-per-bank", type=int)
    generation.add_argument("--output-dir", required=True)


def _add_backbone_command(subparsers: _Subparsers) -> None:
    backbone = subparsers.add_parser("backbone")
    commands = backbone.add_subparsers(dest="backbone_command", required=True)
    for name in ("plan", "status", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--run-dir")
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir")
    run.add_argument(
        "--through", choices=[f"B{index}" for index in range(10)], required=True
    )


def _add_base_responses_command(subparsers: _Subparsers) -> None:
    base = subparsers.add_parser("base-responses")
    commands = base.add_subparsers(dest="base_responses_command", required=True)
    for name in ("plan", "run-shard", "status", "finalize", "verify"):
        command = commands.add_parser(name)
        for flag in ("--config", "--package", "--parquet"):
            command.add_argument(flag, required=True)
        if name in ("plan", "run-shard"):
            command.add_argument("--base-snapshot", required=True)
            command.add_argument("--tokenizer-snapshot", required=True)
        if name != "plan":
            command.add_argument("--output-dir", required=True)
        if name == "run-shard":
            command.add_argument("--shard-index", type=int, required=True)
            command.add_argument("--shard-count", type=int, required=True)
            command.add_argument("--device")
            command.add_argument("--dtype")
        if name == "finalize":
            command.add_argument("--code-revision", required=True)
            command.add_argument("--output", required=True)
        if name == "verify":
            command.add_argument("--receipt", required=True)
            command.add_argument("--base-snapshot")
            command.add_argument("--tokenizer-snapshot")


def _add_activation_contract_command(subparsers: _Subparsers) -> None:
    command = subparsers.add_parser("activation-contract-smoke")
    for flag in (
        "--config",
        "--generation-config",
        "--package",
        "--parquet",
        "--output-dir",
        "--base-snapshot",
        "--tokenizer-snapshot",
        "--output",
        "--code-revision",
    ):
        command.add_argument(flag, required=True)
    command.add_argument("--device")
    command.add_argument("--dtype")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="concept-filtering")
    subparsers = cast(_Subparsers, parser.add_subparsers(dest="command", required=True))
    _add_validation_commands(subparsers)
    _add_manifest_command(subparsers)
    _add_compare_command(subparsers)
    _add_evaluate_command(subparsers)
    _add_evaluator_reconstruction_command(subparsers)
    _add_training_sample_reconstruction_command(subparsers)
    _add_training_recipe_reconstruction_command(subparsers)
    _add_training_recovery_smoke_command(subparsers)
    _add_training_data_package_command(subparsers)
    _add_sae_compatibility_command(subparsers)
    _add_sae_functional_smoke_command(subparsers)
    _add_dry_run_command(subparsers)
    _add_extract_command(subparsers)
    _add_smoke_command(subparsers)
    _add_generation_command(subparsers)
    _add_backbone_command(subparsers)
    _add_base_responses_command(subparsers)
    _add_activation_contract_command(subparsers)
    _add_r2_provider_scoring_command(subparsers)
    return parser


def _add_r2_provider_scoring_command(subparsers: _Subparsers) -> None:
    command = subparsers.add_parser("run-r2-provider-scoring")
    for flag in (
        "--decision",
        "--decision-receipt",
        "--acceptance",
        "--evaluator",
        "--bank-dir",
        "--validation-bank-receipt",
        "--validation-bank-output",
        "--refusal-bank-receipt",
        "--refusal-bank-output",
        "--run-dir",
        "--output",
        "--code-revision",
    ):
        command.add_argument(flag, required=True)


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True))


def _handle_receipt(args: argparse.Namespace) -> int:
    receipt = load_artifact_receipt(Path(cast(str, args.path)))
    _print_json(
        {
            "artifact_id": receipt.artifact_id,
            "kind": receipt.kind.value,
            "revision": receipt.revision,
            "status": receipt.status.value,
        }
    )
    return 0


def _handle_organism(args: argparse.Namespace) -> int:
    organism = load_organism_config(Path(cast(str, args.path)))
    _print_json(
        {
            "artifact_count": len(organism.artifacts),
            "organism_id": organism.organism_id,
            "organism_sha256": organism.organism_sha256,
            "parameter_count": len(organism.parameters),
        }
    )
    return 0


def _handle_extraction_receipt(args: argparse.Namespace) -> int:
    receipt = load_extraction_receipt(
        Path(cast(str, args.receipt)), output_path=Path(cast(str, args.output))
    )
    _print_json(
        {
            "behavior": receipt.behavior,
            "output_sha256": receipt.output_sha256,
            "receipt_sha256": receipt.receipt_sha256,
        }
    )
    return 0


def _handle_smoke_plan(args: argparse.Namespace) -> int:
    plan = load_smoke_plan(Path(cast(str, args.path)))
    _print_json(
        {
            "decision_status": plan.decision_status,
            "rank": plan.rank,
            "scientific_use": plan.scientific_use,
            "smoke_id": plan.smoke_id,
            "starts_experiment_timer": plan.starts_experiment_timer,
        }
    )
    return 0


def _handle_training_data_package(args: argparse.Namespace) -> int:
    config = load_training_data_package_config(Path(cast(str, args.config)))
    receipt = materialize_training_data_package(
        config,
        training_recipe_path=Path(cast(str, args.training_recipe)),
        training_sample_receipt_path=Path(cast(str, args.training_sample_receipt)),
        sample_path=Path(cast(str, args.parquet)),
        tokenizer_snapshot=Path(cast(str, args.tokenizer_snapshot)),
        code_revision=cast(str, args.code_revision),
        package_path=Path(cast(str, args.package)),
        receipt_path=Path(cast(str, args.output)),
    )
    _print_json(
        {
            "receipt_sha256": receipt.receipt_sha256,
            "row_count": receipt.row_count,
            "retained_row_count": receipt.retained_row_count,
            "scientific_use": False,
        }
    )
    return 0


def _handle_verify_training_data_package(args: argparse.Namespace) -> int:
    receipt = verify_training_data_package(
        load_training_data_package_config(Path(cast(str, args.config))),
        training_recipe_path=Path(cast(str, args.training_recipe)),
        training_sample_receipt_path=Path(cast(str, args.training_sample_receipt)),
        code_revision=cast(str, args.code_revision),
        sample_path=Path(cast(str, args.parquet)),
        tokenizer_snapshot=Path(cast(str, args.tokenizer_snapshot)),
        package_path=Path(cast(str, args.package)),
        receipt_path=Path(cast(str, args.receipt)),
    )
    _print_json(
        {
            "receipt_sha256": receipt.receipt_sha256,
            "row_count": receipt.row_count,
            "retained_row_count": receipt.retained_row_count,
            "scientific_use": False,
        }
    )
    return 0


def _handle_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(
        Path(cast(str, args.input)), manifest_id=cast(str, args.manifest_id)
    )
    write_manifest(manifest, Path(cast(str, args.output)))
    _print_json(
        {
            "example_count": manifest.example_count,
            "manifest_sha256": manifest.manifest_sha256,
            "token_count": manifest.token_count,
        }
    )
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    reference = load_reference_spec(Path(cast(str, args.reference)))
    organism = load_organism_config(Path(cast(str, args.organism)))
    report = compare_expectations(
        reference.reference_id,
        organism.organism_id,
        list(reference.expectations),
        organism.parameters,
    )
    write_divergence_report(report, Path(cast(str, args.output)))
    _print_json(
        {
            "blocking_count": report.blocking_count,
            "is_blocked": report.is_blocked,
            "report_sha256": report.report_sha256,
        }
    )
    return 3 if report.is_blocked else 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    organism = load_organism_config(Path(cast(str, args.organism)))
    manifest = load_manifest(
        Path(cast(str, args.manifest)), input_path=Path(cast(str, args.sample))
    )
    receipt = build_evaluation_receipt(
        Path(cast(str, args.scores)),
        evaluator=load_evaluator_config(Path(cast(str, args.evaluator))),
        code_revision=cast(str, args.code_revision),
        organism=organism,
        manifest=manifest,
        expected_organism_sha256=cast(str, args.expected_organism_sha256),
        expected_manifest_sha256=cast(str, args.expected_manifest_sha256),
    )
    write_evaluation_receipt(receipt, Path(cast(str, args.output)))
    _print_json(
        {
            "result_sha256": receipt.result_sha256,
            "scientific_use": receipt.scientific_use,
        }
    )
    return 0


def _handle_evaluator_reconstruction(args: argparse.Namespace) -> int:
    archive_args = cast(list[list[str]], args.archive)
    archive_paths = {behavior: Path(path) for behavior, path in archive_args}
    if len(archive_paths) != len(archive_args):
        raise EvaluatorReconstructionError("archive behaviors must be unique")
    receipt = verify_evaluator_reconstruction(
        load_evaluator_reconstruction(Path(cast(str, args.config))),
        public_source_path=Path(cast(str, args.public_source)),
        archive_paths=archive_paths,
    )
    write_evaluator_reconstruction_receipt(receipt, Path(cast(str, args.output)))
    _print_json(receipt.to_dict())
    return 0


def _handle_training_sample_reconstruction(args: argparse.Namespace) -> int:
    receipt = verify_training_sample_reconstruction(
        load_training_sample_reconstruction(Path(cast(str, args.config))),
        sample_path=Path(cast(str, args.parquet)),
    )
    write_training_sample_receipt(receipt, Path(cast(str, args.output)))
    _print_json(receipt.to_dict())
    return 0


def _handle_training_recipe_reconstruction(args: argparse.Namespace) -> int:
    receipt = verify_training_recipe_reconstruction(
        load_training_recipe_reconstruction(Path(cast(str, args.config))),
        artifact_dir=Path(cast(str, args.artifact_dir)),
    )
    write_training_recipe_receipt(receipt, Path(cast(str, args.output)))
    _print_json(receipt.to_dict())
    return 0


def _handle_training_recovery_smoke(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    recipe = load_named_training_recipe(Path(cast(str, args.config)))
    receipt = execute_training_recovery_smoke(
        recipe,
        code_revision=cast(str, args.code_revision),
        output_path=Path(cast(str, args.output)),
        run=lambda value: run_transformers_training_recovery(
            value,
            base_snapshot=Path(cast(str, args.base_snapshot)),
            tokenizer_snapshot=Path(cast(str, args.tokenizer_snapshot)),
            fixture_path=Path(cast(str, args.fixture)),
            work_dir=Path(cast(str, args.work_dir)),
        ),
    )
    _print_json(
        {
            "receipt_sha256": receipt.receipt_sha256,
            "authority_sha256": recipe.smoke_authority_sha256,
            "authority": dict(sorted(recipe.smoke_authority.items())),
        }
    )
    return 0


def _handle_sae_compatibility(args: argparse.Namespace) -> int:
    role_names = (
        "base_config",
        "sae_cfg",
        "runner_cfg",
        "release_readme",
        "weights_header",
        "saelens_docs",
        "artifact_receipt",
    )
    input_paths = {role: Path(cast(str, getattr(args, role))) for role in role_names}
    receipt = verify_sae_compatibility(
        load_sae_compatibility(Path(cast(str, args.config))),
        input_paths=input_paths,
    )
    write_sae_compatibility_receipt(receipt, Path(cast(str, args.output)))
    _print_json(receipt.to_dict())
    return 0


def _handle_sae_functional_smoke(args: argparse.Namespace) -> int:
    plan = load_sae_functional_smoke(Path(cast(str, args.config)))
    receipt = execute_sae_functional_smoke(
        plan,
        code_revision=cast(str, args.code_revision),
        output_path=Path(cast(str, args.output)),
        run=lambda value: run_transformers_sae_smoke(
            value,
            base_snapshot=Path(cast(str, args.base_snapshot)),
            adapter_snapshot=Path(cast(str, args.adapter_snapshot)),
            sae_directory=Path(cast(str, args.sae_directory)),
        ),
    )
    _print_json(
        {
            "receipt_sha256": receipt.receipt_sha256,
            "scientific_use": plan.scientific_use,
            "starts_experiment_timer": plan.starts_experiment_timer,
            "functional_execution": plan.functional_execution,
        }
    )
    return 0


def _handle_dry_run(args: argparse.Namespace) -> int:
    receipt = execute_dry_run(
        Path(cast(str, args.bundle)),
        code_revision=cast(str, args.code_revision),
        output_dir=Path(cast(str, args.output_dir)),
    )
    _print_json(
        {
            "gate_r2_blocked": receipt.gate_r2_blocked,
            "run_sha256": receipt.run_sha256,
            "scientific_use": receipt.scientific_use,
        }
    )
    return 0


def _handle_extract(args: argparse.Namespace) -> int:
    extraction = extract_scores(
        Path(cast(str, args.archive)),
        scorer=cast(str, args.scorer),
        expected_sha256=cast(str, args.expected_sha256),
        output_path=Path(cast(str, args.output)),
        receipt_path=Path(cast(str, args.receipt)),
    )
    _print_json(extraction.to_dict())
    return 0


def _handle_prompt_bank(args: argparse.Namespace) -> int:
    extraction = extract_prompt_bank(
        Path(cast(str, args.archive)),
        expected_sha256=cast(str, args.expected_sha256),
        output_path=Path(cast(str, args.output)),
        receipt_path=Path(cast(str, args.receipt)),
    )
    _print_json(extraction.to_dict())
    return 0


def _handle_smoke(args: argparse.Namespace) -> int:
    plan = load_smoke_plan(Path(cast(str, args.path)))
    cache_dir = Path(cast(str, args.cache_dir))
    receipt = execute_smoke(
        plan,
        code_revision=cast(str, args.code_revision),
        output_path=Path(cast(str, args.output)),
        run=lambda value: run_transformers_smoke(value, cache_dir),
    )
    _print_json(
        {
            "receipt_sha256": receipt.receipt_sha256,
            "scientific_use": plan.scientific_use,
            "starts_experiment_timer": plan.starts_experiment_timer,
        }
    )
    return 0


def _handle_generation(args: argparse.Namespace) -> int:
    plan = load_generation_plan(Path(cast(str, args.path)))
    smoke = load_smoke_plan(Path(cast(str, args.smoke_plan)))
    bank_paths = cast(list[list[str]], args.prompt_bank)
    banks = tuple(
        load_prompt_bank(Path(receipt), output_path=Path(output))
        for receipt, output in bank_paths
    )
    cache_dir = Path(cast(str, args.cache_dir))
    receipt = execute_generation_bank(
        plan,
        smoke,
        code_revision=cast(str, args.code_revision),
        banks=banks,
        limit_per_bank=cast(int | None, args.limit_per_bank),
        output_dir=Path(cast(str, args.output_dir)),
        run=lambda smoke_plan, generation_plan, prompts: run_transformers_generation(
            smoke_plan, generation_plan, prompts, cache_dir
        ),
    )
    _print_json(receipt.to_dict())
    return 0


def _handle_backbone(args: argparse.Namespace) -> int:
    config = load_backbone_config(Path(cast(str, args.config)))
    run_dir_value = cast(str | None, args.run_dir)
    run_dir = (
        Path(run_dir_value)
        if run_dir_value is not None
        else Path(".runs/backbone") / config.backbone_id
    )
    command = cast(str, args.backbone_command)
    if command == "plan":
        result = plan_backbone(config, run_dir)
    elif command == "status":
        result = status_backbone(config, run_dir)
    elif command == "run":
        result = execute_backbone(config, run_dir, through=cast(str, args.through))
    elif command == "verify":
        result = verify_backbone(config, run_dir)
    else:
        raise AssertionError("unhandled backbone command")
    _print_json(result)
    if command == "verify" and result["ready_for_sae_feature_selection"] is False:
        return 3
    return 0


def _source_root_for_base_responses() -> Path:
    return source_repository_root()


def _base_response_executor_revision(value: str) -> str:
    return verify_executor_revision(value)


def _base_response_backend(
    config: BaseResponseConfig,
    base_snapshot: Path,
    tokenizer_snapshot: Path,
    device: str | None,
    dtype: str | None,
) -> Callable[[JoinedExample], GeneratedResponse]:
    from concept_filtering.base_response_transformers import (
        build_transformers_backend,
    )

    return build_transformers_backend(
        config,
        base_snapshot=base_snapshot,
        tokenizer_snapshot=tokenizer_snapshot,
        device=device,
        dtype=dtype,
    )


def _base_response_world(
    args: argparse.Namespace,
) -> tuple[BaseResponseConfig, BaseResponseEvidence, tuple[JoinedExample, ...]]:
    config = load_base_response_config(Path(cast(str, args.config)))
    evidence = validate_base_response_evidence(
        config, _source_root_for_base_responses()
    )
    examples = load_joined_examples(
        evidence.package_payload,
        Path(cast(str, args.package)),
        Path(cast(str, args.parquet)),
    )
    return config, evidence, examples


def _handle_base_response_plan(
    args: argparse.Namespace,
    config: BaseResponseConfig,
    examples: tuple[JoinedExample, ...],
) -> dict[str, object]:
    verify_artifact_snapshot(config.base, Path(cast(str, args.base_snapshot)))
    verify_artifact_snapshot(config.tokenizer, Path(cast(str, args.tokenizer_snapshot)))
    return {
        "generation_id": config.generation_id,
        "config_sha256": config.config_sha256,
        "package_receipt_sha256": config.package.receipt_sha256,
        "b2_receipt_sha256": config.b2_gate.receipt_sha256,
        "retained_examples": len(examples),
        "model_loaded": False,
    }


def _handle_base_response_run_shard(
    args: argparse.Namespace,
    config: BaseResponseConfig,
    examples: tuple[JoinedExample, ...],
) -> dict[str, object]:
    base_snapshot = Path(cast(str, args.base_snapshot))
    tokenizer_snapshot = Path(cast(str, args.tokenizer_snapshot))
    backend = _base_response_backend(
        config,
        base_snapshot,
        tokenizer_snapshot,
        cast("str | None", args.device),
        cast("str | None", args.dtype),
    )
    report = run_base_response_shard(
        config,
        examples,
        shard_index=cast(int, args.shard_index),
        shard_count=cast(int, args.shard_count),
        output_dir=Path(cast(str, args.output_dir)),
        backend=backend,
    )
    return {
        "claimed": report.claimed,
        "generated": report.generated,
        "replayed": report.replayed,
    }


def _handle_base_response_finalize(
    args: argparse.Namespace,
    config: BaseResponseConfig,
    evidence: BaseResponseEvidence,
    examples: tuple[JoinedExample, ...],
) -> dict[str, object]:
    revision = _base_response_executor_revision(cast(str, args.code_revision))
    receipt = finalize_base_responses(
        config,
        evidence,
        examples,
        Path(cast(str, args.output_dir)),
        Path(cast(str, args.output)),
        executor_revision=revision,
    )
    return receipt.to_dict()


def _handle_base_response_verify(
    args: argparse.Namespace,
    config: BaseResponseConfig,
    evidence: BaseResponseEvidence,
    examples: tuple[JoinedExample, ...],
) -> dict[str, object]:
    for binding, argument in (
        (config.base, args.base_snapshot),
        (config.tokenizer, args.tokenizer_snapshot),
    ):
        if argument is not None:
            verify_artifact_snapshot(binding, Path(cast(str, argument)))
    receipt = verify_base_responses(
        config,
        evidence,
        examples,
        Path(cast(str, args.output_dir)),
        Path(cast(str, args.receipt)),
    )
    return {"verified": True, "receipt_sha256": receipt.receipt_sha256}


def _handle_base_responses(args: argparse.Namespace) -> int:
    config, evidence, examples = _base_response_world(args)
    command = cast(str, args.base_responses_command)
    if command == "plan":
        result = _handle_base_response_plan(args, config, examples)
    elif command == "run-shard":
        result = _handle_base_response_run_shard(args, config, examples)
    elif command == "status":
        report = base_response_status(
            config, examples, Path(cast(str, args.output_dir))
        )
        result = dataclasses.asdict(report)
    elif command == "finalize":
        result = _handle_base_response_finalize(args, config, evidence, examples)
    elif command == "verify":
        result = _handle_base_response_verify(args, config, evidence, examples)
    else:
        raise AssertionError("unhandled base-responses command")
    _print_json(result)
    return 0


def _contract_examples(
    args: argparse.Namespace,
) -> tuple[BaseResponseConfig, tuple[JoinedExample, ...]]:
    config = load_base_response_config(Path(cast(str, args.generation_config)))
    evidence = validate_base_response_evidence(
        config, _source_root_for_base_responses()
    )
    return config, load_joined_examples(
        evidence.package_payload,
        Path(cast(str, args.package)),
        Path(cast(str, args.parquet)),
    )


def _handle_activation_contract_smoke(args: argparse.Namespace) -> int:
    plan = load_activation_contract(Path(cast(str, args.config)))
    generation_config, examples = _contract_examples(args)
    example = resolve_contract_example(
        examples,
        plan.probe.example_id,
        Path(cast(str, args.output_dir)),
    )
    observation = run_activation_contract(
        plan,
        example,
        generation_prompt=generation_config.generation.generation_prompt,
        base_snapshot=Path(cast(str, args.base_snapshot)),
        tokenizer_snapshot=Path(cast(str, args.tokenizer_snapshot)),
        device=cast("str | None", args.device),
        dtype=cast("str | None", args.dtype),
    )
    receipt = activation_contract_receipt(
        plan, observation, code_revision=cast(str, args.code_revision)
    )
    write_activation_receipt(receipt, Path(cast(str, args.output)))
    _print_json(receipt.to_dict())
    return 0


def _handle_r2_provider_scoring(args: argparse.Namespace) -> int:
    decision_path = Path(cast(str, args.decision))
    acceptance = load_r2_owner_acceptance(
        Path(cast(str, args.acceptance)),
        decision_path=decision_path,
        decision_receipt_path=Path(cast(str, args.decision_receipt)),
    )
    evaluator_path = Path(cast(str, args.evaluator))
    evaluator = load_evaluator_reconstruction(evaluator_path)
    decision = cast(
        dict[str, Any], json.loads(decision_path.read_text(encoding="utf-8"))
    )
    judge = cast(dict[str, Any], decision["judge_identity"])
    if file_sha256(evaluator_path) != judge["evaluator_reconstruction_config_sha256"]:
        raise R2ProviderScoringError(
            "evaluator config does not match the accepted decision binding"
        )
    rule = cast(dict[str, Any], decision["acceptance_rule"])
    references = {
        behavior: float(cast(dict[str, Any], spec)["mean_score"])
        for behavior, spec in cast(dict[str, Any], decision["reference_scores"]).items()
    }
    expected_counts = {
        behavior: int(cast(dict[str, Any], spec)["sample_count"])
        for behavior, spec in cast(
            dict[str, Any],
            cast(dict[str, Any], decision["response_bank"])["behaviors"],
        ).items()
    }
    _, validation = load_prompt_bank(
        Path(cast(str, args.validation_bank_receipt)),
        output_path=Path(cast(str, args.validation_bank_output)),
    )
    _, refusal = load_prompt_bank(
        Path(cast(str, args.refusal_bank_receipt)),
        output_path=Path(cast(str, args.refusal_bank_output)),
    )
    prompts = {record.question_id: record for record in validation + refusal}
    items = build_r2_scoring_items(
        evaluator,
        Path(cast(str, args.bank_dir)),
        prompts,
        expected_counts=expected_counts,
    )
    transport = build_anthropic_transport()
    run_dir = Path(cast(str, args.run_dir))
    scores = score_r2_items(items, transport, evaluator=evaluator, run_dir=run_dir)
    gate = finalize_r2_gate(
        acceptance=acceptance,
        evaluator=evaluator,
        items=items,
        scores=scores,
        reference_means=references,
        maximum_difference=float(rule["maximum_difference"]),
        code_revision=cast(str, args.code_revision),
        run_dir=run_dir,
        output_path=Path(cast(str, args.output)),
    )
    _print_json(
        {
            "gate_receipt_sha256": gate.receipt_sha256,
            "all_behaviors_passed": gate.all_behaviors_passed,
            "behaviors": gate.behaviors,
            "scientific_use": False,
            "starts_experiment_timer": False,
        }
    )
    return 0


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "activation-contract-smoke": _handle_activation_contract_smoke,
    "run-r2-provider-scoring": _handle_r2_provider_scoring,
    "validate-receipt": _handle_receipt,
    "validate-organism": _handle_organism,
    "validate-inspect-extraction": _handle_extraction_receipt,
    "validate-smoke-plan": _handle_smoke_plan,
    "materialize-training-data-package": _handle_training_data_package,
    "verify-training-data-package": _handle_verify_training_data_package,
    "build-manifest": _handle_manifest,
    "compare": _handle_compare,
    "evaluate": _handle_evaluate,
    "verify-evaluator-reconstruction": _handle_evaluator_reconstruction,
    "verify-training-sample-reconstruction": _handle_training_sample_reconstruction,
    "verify-training-recipe-reconstruction": _handle_training_recipe_reconstruction,
    "run-training-recovery-smoke": _handle_training_recovery_smoke,
    "verify-sae-compatibility": _handle_sae_compatibility,
    "run-sae-functional-smoke": _handle_sae_functional_smoke,
    "dry-run": _handle_dry_run,
    "extract-inspect-eval": _handle_extract,
    "extract-prompt-bank": _handle_prompt_bank,
    "run-smoke": _handle_smoke,
    "run-generation-bank": _handle_generation,
    "backbone": _handle_backbone,
    "base-responses": _handle_base_responses,
}


def _run_command(args: argparse.Namespace) -> int:
    command = cast(str, args.command)
    try:
        handler = _HANDLERS[command]
    except KeyError as error:
        raise AssertionError(f"unhandled command: {command}") from error
    return handler(args)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run_command(args)
    except _USER_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
