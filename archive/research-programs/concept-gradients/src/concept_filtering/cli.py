"""Command-line interface for reconstruction receipts and offline gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast

from concept_filtering.backbone import (
    BackboneValidationError,
    execute_backbone,
    load_backbone_config,
    plan_backbone,
    status_backbone,
    verify_backbone,
)
from concept_filtering.causal_gradient_plan import (
    CausalGradientPlanError,
    load_causal_gradient_plan,
)
from concept_filtering.compute_contract import (
    ComputeContractError,
    plan_compute_contract,
)
from concept_filtering.divergence import (
    DivergenceValidationError,
    compare_expectations,
    load_reference_spec,
    write_divergence_report,
)
from concept_filtering.dry_run import DryRunValidationError, execute_dry_run
from concept_filtering.ekfac_rank32_smoke import (
    Rank32EKFACSmokeError,
    Rank32RuntimePaths,
    execute_rank32_ekfac_smoke,
    load_rank32_ekfac_smoke_plan,
)
from concept_filtering.ekfac_reconstruction import (
    EKFACReconstructionError,
    load_ekfac_reconstruction,
    verify_ekfac_reconstruction,
    write_ekfac_reconstruction_receipt,
)
from concept_filtering.ekfac_smoke import (
    EKFACSmokeError,
    execute_ekfac_backend_smoke,
    load_ekfac_smoke_plan,
)
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
from concept_filtering.organism import OrganismValidationError, load_organism_config
from concept_filtering.prompt_bank import extract_prompt_bank, load_prompt_bank
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
from concept_filtering.smoke import (
    SmokeValidationError,
    execute_smoke,
    load_smoke_plan,
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
from concept_filtering.training_recipe import TrainingRecipeError, load_training_recipe
from concept_filtering.training_recipe_reconstruction import (
    TrainingRecipeReconstructionError,
    load_training_recipe_reconstruction,
    verify_training_recipe_reconstruction,
    write_training_recipe_receipt,
)
from concept_filtering.training_sample_reconstruction import (
    TrainingSampleReconstructionError,
    load_training_sample_reconstruction,
    verify_training_sample_reconstruction,
    write_training_sample_receipt,
)
from concept_filtering.training_smoke import TrainingSmokeError, execute_training_smoke
from concept_filtering.training_smoke_transformers import (
    run_transformers_training_smoke,
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
    TrainingRecipeError,
    ComputeContractError,
    TrainingSmokeError,
    SAECompatibilityError,
    SAEFunctionalSmokeError,
    CausalGradientPlanError,
    TrainingDataPackageError,
    EKFACReconstructionError,
    EKFACSmokeError,
    Rank32EKFACSmokeError,
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
    gradient = subparsers.add_parser("validate-causal-gradient-plan")
    gradient.add_argument("path")
    recipe = subparsers.add_parser("validate-training-recipe")
    recipe.add_argument("path")
    compute = subparsers.add_parser("plan-compute-contract")
    compute.add_argument("config")
    compute.add_argument("--deployment-receipt")


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


def _add_ekfac_reconstruction_command(subparsers: _Subparsers) -> None:
    verify = subparsers.add_parser("verify-ekfac-reconstruction")
    verify.add_argument("config")
    verify.add_argument("--source-root", required=True)
    verify.add_argument("--query-evidence-root", required=True)
    verify.add_argument("--output", required=True)


def _add_ekfac_smoke_command(subparsers: _Subparsers) -> None:
    smoke = subparsers.add_parser("run-ekfac-backend-smoke")
    smoke.add_argument("config")
    smoke.add_argument("--reconstruction-config", required=True)
    smoke.add_argument("--reconstruction-receipt", required=True)
    smoke.add_argument("--attribution-plan", required=True)
    smoke.add_argument("--code-revision", required=True)
    smoke.add_argument("--run-dir", required=True)
    smoke.add_argument("--output", required=True)


def _add_rank32_ekfac_smoke_command(subparsers: _Subparsers) -> None:
    smoke = subparsers.add_parser("run-rank32-ekfac-smoke")
    smoke.add_argument("config")
    smoke.add_argument("--repository-root", required=True)
    smoke.add_argument("--base-snapshot", required=True)
    smoke.add_argument("--adapter-snapshot", required=True)
    smoke.add_argument("--training-sample", required=True)
    smoke.add_argument("--training-package", required=True)
    smoke.add_argument("--code-revision", required=True)
    smoke.add_argument("--run-dir", required=True)
    smoke.add_argument("--output", required=True)


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


def _add_training_smoke_command(subparsers: _Subparsers) -> None:
    smoke = subparsers.add_parser("run-training-smoke")
    smoke.add_argument("config")
    smoke.add_argument("--code-revision", required=True)
    smoke.add_argument("--run-dir", required=True)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="concept-filtering")
    subparsers = cast(_Subparsers, parser.add_subparsers(dest="command", required=True))
    _add_validation_commands(subparsers)
    _add_manifest_command(subparsers)
    _add_compare_command(subparsers)
    _add_evaluate_command(subparsers)
    _add_evaluator_reconstruction_command(subparsers)
    _add_ekfac_reconstruction_command(subparsers)
    _add_ekfac_smoke_command(subparsers)
    _add_rank32_ekfac_smoke_command(subparsers)
    _add_training_sample_reconstruction_command(subparsers)
    _add_training_recipe_reconstruction_command(subparsers)
    _add_training_smoke_command(subparsers)
    _add_training_data_package_command(subparsers)
    _add_sae_compatibility_command(subparsers)
    _add_sae_functional_smoke_command(subparsers)
    _add_dry_run_command(subparsers)
    _add_extract_command(subparsers)
    _add_smoke_command(subparsers)
    _add_generation_command(subparsers)
    _add_backbone_command(subparsers)
    return parser


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


def _handle_causal_gradient_plan(args: argparse.Namespace) -> int:
    plan = load_causal_gradient_plan(Path(cast(str, args.path)))
    _print_json(
        {
            "blocked_objective_ids": plan.blocked_objective_ids,
            "executable_objective_ids": plan.executable_objective_ids,
            "method": plan.method,
            "phase": plan.phase,
            "plan_id": plan.plan_id,
            "plan_sha256": plan.plan_sha256,
            "scientific_use": plan.scientific_use,
        }
    )
    return 0


def _handle_training_recipe(args: argparse.Namespace) -> int:
    recipe = load_training_recipe(Path(cast(str, args.path)))
    _print_json(
        {
            "bounded_smoke_execution_allowed": recipe.bounded_smoke_execution_allowed,
            "contract_id": recipe.contract_id,
            "contract_sha256": recipe.contract_sha256,
            "effective_batch_size": recipe.effective_batch_size,
            "phase": recipe.phase,
            "rank": recipe.rank,
            "recipe_id": recipe.recipe_id,
            "recipe_sha256": recipe.recipe_sha256,
            "scientific_training_allowed": recipe.scientific_training_allowed,
            "scientific_use": recipe.scientific_use,
        }
    )
    return 0


def _handle_training_smoke(args: argparse.Namespace) -> int:
    recipe = load_training_recipe(Path(cast(str, args.config)))
    run_dir = Path(cast(str, args.run_dir))
    receipt = execute_training_smoke(
        recipe,
        code_revision=cast(str, args.code_revision),
        output_path=Path(cast(str, args.output)),
        run=lambda value: run_transformers_training_smoke(value, run_dir),
    )
    _print_json(
        {
            "bounded_smoke_execution": True,
            "receipt_sha256": receipt.receipt_sha256,
            "scientific_training_allowed": False,
            "scientific_use": False,
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


def _handle_ekfac_reconstruction(args: argparse.Namespace) -> int:
    receipt = verify_ekfac_reconstruction(
        load_ekfac_reconstruction(Path(cast(str, args.config))),
        source_root=Path(cast(str, args.source_root)),
        query_evidence_root=Path(cast(str, args.query_evidence_root)),
    )
    write_ekfac_reconstruction_receipt(receipt, Path(cast(str, args.output)))
    _print_json(receipt.to_dict())
    return 0


def _handle_ekfac_smoke(args: argparse.Namespace) -> int:
    plan = load_ekfac_smoke_plan(Path(cast(str, args.config)))
    receipt = execute_ekfac_backend_smoke(
        plan,
        reconstruction_config=Path(cast(str, args.reconstruction_config)),
        reconstruction_receipt=Path(cast(str, args.reconstruction_receipt)),
        attribution_plan=Path(cast(str, args.attribution_plan)),
        code_revision=cast(str, args.code_revision),
        run_dir=Path(cast(str, args.run_dir)),
        output_path=Path(cast(str, args.output)),
    )
    _print_json(
        {
            "receipt_sha256": receipt.receipt_sha256,
            "output_release": receipt.output_release,
            "repeat_count": plan.repeat_count,
            "scientific_use": False,
            "concept_scoring_allowed": False,
        }
    )
    return 0


def _handle_rank32_ekfac_smoke(args: argparse.Namespace) -> int:
    plan = load_rank32_ekfac_smoke_plan(Path(cast(str, args.config)))
    paths = Rank32RuntimePaths(
        repository_root=Path(cast(str, args.repository_root)),
        base_snapshot=Path(cast(str, args.base_snapshot)),
        adapter_snapshot=Path(cast(str, args.adapter_snapshot)),
        training_sample=Path(cast(str, args.training_sample)),
        training_package=Path(cast(str, args.training_package)),
    )
    receipt = execute_rank32_ekfac_smoke(
        plan,
        paths=paths,
        code_revision=cast(str, args.code_revision),
        run_dir=Path(cast(str, args.run_dir)),
        output_path=Path(cast(str, args.output)),
    )
    observation = cast(Mapping[str, object], receipt.payload["observation"])
    _print_json(
        {
            "receipt_sha256": receipt.receipt_sha256,
            "output_release": receipt.output_release,
            "score": observation["score"],
            "scientific_use": False,
            "concept_scoring_allowed": False,
        }
    )
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


def _handle_compute_contract(args: argparse.Namespace) -> int:
    receipt_value = cast(str | None, args.deployment_receipt)
    result = plan_compute_contract(
        Path(cast(str, args.config)),
        Path(receipt_value) if receipt_value is not None else None,
    )
    _print_json(result)
    return 0 if result["ready_for_no_submit"] is True else 3


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "validate-receipt": _handle_receipt,
    "validate-organism": _handle_organism,
    "validate-inspect-extraction": _handle_extraction_receipt,
    "validate-smoke-plan": _handle_smoke_plan,
    "validate-causal-gradient-plan": _handle_causal_gradient_plan,
    "validate-training-recipe": _handle_training_recipe,
    "plan-compute-contract": _handle_compute_contract,
    "run-training-smoke": _handle_training_smoke,
    "materialize-training-data-package": _handle_training_data_package,
    "verify-training-data-package": _handle_verify_training_data_package,
    "build-manifest": _handle_manifest,
    "compare": _handle_compare,
    "evaluate": _handle_evaluate,
    "verify-evaluator-reconstruction": _handle_evaluator_reconstruction,
    "verify-ekfac-reconstruction": _handle_ekfac_reconstruction,
    "run-ekfac-backend-smoke": _handle_ekfac_smoke,
    "run-rank32-ekfac-smoke": _handle_rank32_ekfac_smoke,
    "verify-training-sample-reconstruction": _handle_training_sample_reconstruction,
    "verify-training-recipe-reconstruction": _handle_training_recipe_reconstruction,
    "verify-sae-compatibility": _handle_sae_compatibility,
    "run-sae-functional-smoke": _handle_sae_functional_smoke,
    "dry-run": _handle_dry_run,
    "extract-inspect-eval": _handle_extract,
    "extract-prompt-bank": _handle_prompt_bank,
    "run-smoke": _handle_smoke,
    "run-generation-bank": _handle_generation,
    "backbone": _handle_backbone,
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
