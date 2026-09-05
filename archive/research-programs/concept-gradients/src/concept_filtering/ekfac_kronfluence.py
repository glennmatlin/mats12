"""Kronfluence executor for the deterministic local EKFAC smoke."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import TensorDataset

from concept_filtering.ekfac_smoke import (
    EKFACBackendObservation,
    EKFACQueryObservation,
    EKFACSmokeError,
    EKFACSmokePlan,
)
from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    json_loads_unique,
)

Batch = tuple[torch.Tensor, torch.Tensor]


class _TinyClassificationTask(Task):
    def __init__(self) -> None:
        self.objective_kind = "behavior_loss"

    def compute_train_loss(
        self, batch: Batch, model: nn.Module, sample: bool = False
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if sample:
            with torch.no_grad():
                labels = torch.multinomial(
                    torch.softmax(logits.detach(), dim=-1), 1
                ).flatten()
        return functional.cross_entropy(logits, labels, reduction="sum")

    def compute_measurement(self, batch: Batch, model: nn.Module) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if self.objective_kind == "behavior_loss":
            return functional.cross_entropy(logits, labels, reduction="sum")
        if self.objective_kind == "synthetic_scalar":
            return logits[:, 0].sum()
        raise EKFACSmokeError("unknown EKFAC query objective")

    def get_influence_tracked_modules(self) -> list[str]:
        return ["0", "2"]


@dataclass(frozen=True)
class _Fixture:
    model: nn.Module
    training_dataset: TensorDataset
    query_dataset: TensorDataset
    task: _TinyClassificationTask


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EKFACSmokeError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _float_matrix(value: object) -> list[list[float]]:
    rows = cast(list[object], value)
    return [[_float(item) for item in cast(list[object], row)] for row in rows]


def _float_vector(value: object) -> list[float]:
    return [_float(item) for item in cast(list[object], value)]


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EKFACSmokeError("fixture tensor values must be numeric")
    return float(value)


def _copy_model_parameters(model: nn.Sequential, payload: dict[str, object]) -> None:
    first = cast(nn.Linear, model[0])
    second = cast(nn.Linear, model[2])
    with torch.no_grad():
        first.weight.copy_(
            torch.tensor(_float_matrix(payload["first_weight"]), dtype=torch.float64)
        )
        first.bias.copy_(
            torch.tensor(_float_vector(payload["first_bias"]), dtype=torch.float64)
        )
        second.weight.copy_(
            torch.tensor(_float_matrix(payload["second_weight"]), dtype=torch.float64)
        )
        second.bias.copy_(
            torch.tensor(_float_vector(payload["second_bias"]), dtype=torch.float64)
        )


def _build_model(plan: EKFACSmokePlan) -> nn.Module:
    model_payload = _mapping(plan.fixture["model"], "fixture.model")
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
    _copy_model_parameters(model, model_payload)
    return model


def _build_dataset(value: object) -> TensorDataset:
    payload = _mapping(value, "fixture dataset")
    inputs = torch.tensor(_float_matrix(payload["inputs"]), dtype=torch.float64)
    labels = torch.tensor(cast(list[int], payload["labels"]), dtype=torch.long)
    return TensorDataset(inputs, labels)


def _build_fixture(plan: EKFACSmokePlan) -> _Fixture:
    return _Fixture(
        _build_model(plan),
        _build_dataset(plan.fixture["training_dataset"]),
        _build_dataset(plan.fixture["query_dataset"]),
        _TinyClassificationTask(),
    )


def _runtime_versions() -> dict[str, str]:
    try:
        return {
            name: importlib.metadata.version(name)
            for name in ("kronfluence", "safetensors", "torch")
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise EKFACSmokeError(f"missing EKFAC runtime package: {error}") from error


def _factor_arguments() -> FactorArguments:
    return FactorArguments(
        strategy="ekfac",
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
        eigendecomposition_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        lambda_dtype=torch.float64,
    )


def _score_arguments() -> ScoreArguments:
    return ScoreArguments(
        damping_factor=0.001,
        query_gradient_accumulation_steps=1,
        query_gradient_low_rank=None,
        use_full_svd=True,
        query_gradient_svd_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        precondition_dtype=torch.float64,
        score_dtype=torch.float64,
    )


def _analyzer(fixture: _Fixture, run_dir: Path) -> Analyzer:
    prepared = prepare_model(fixture.model, fixture.task)
    return Analyzer(
        analysis_name="tiny-ekfac",
        model=prepared,
        task=fixture.task,
        cpu=True,
        disable_tqdm=True,
        disable_model_save=True,
        output_dir=str(run_dir),
    )


def _fit_factors(analyzer: Analyzer, fixture: _Fixture) -> None:
    analyzer.fit_all_factors(
        factors_name="backend-factors",
        dataset=fixture.training_dataset,
        per_device_batch_size=2,
        factor_args=_factor_arguments(),
        overwrite_output_dir=True,
    )


def _compute_scores(
    analyzer: Analyzer, fixture: _Fixture, objective_kind: str
) -> tuple[float, ...]:
    fixture.task.objective_kind = objective_kind
    scores_name = f"backend-scores-{objective_kind}"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name="backend-factors",
        query_dataset=fixture.query_dataset,
        train_dataset=fixture.training_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=2,
        score_args=_score_arguments(),
        overwrite_output_dir=True,
    )
    loaded = analyzer.load_pairwise_scores(scores_name)
    if loaded is None or "all_modules" not in loaded:
        raise EKFACSmokeError("Kronfluence score output is missing")
    value = loaded["all_modules"]
    scores = cast(list[list[float]], value.tolist())
    if len(scores) != 1 or any(not math.isfinite(item) for item in scores[0]):
        raise EKFACSmokeError("Kronfluence produced invalid pairwise scores")
    return tuple(scores[0])


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    payload = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes_sha256": bytes_sha256(tensor.numpy().tobytes()),
    }
    return canonical_sha256(payload)


def _model_sha256(model: nn.Module) -> str:
    manifest = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for name, value in model.state_dict().items()
    ]
    return canonical_sha256(manifest)


def _gradient_sha256(plan: EKFACSmokePlan, *, objective_kind: str | None) -> str:
    fixture = _build_fixture(plan)
    query = objective_kind is not None
    dataset = fixture.query_dataset if query else fixture.training_dataset
    inputs, labels = cast(Batch, dataset[0])
    batch = (inputs.unsqueeze(0), labels.unsqueeze(0))
    if objective_kind is not None:
        fixture.task.objective_kind = objective_kind
        loss = fixture.task.compute_measurement(batch, fixture.model)
    else:
        loss = fixture.task.compute_train_loss(batch, fixture.model)
    gradients = torch.autograd.grad(loss, tuple(fixture.model.parameters()))
    return canonical_sha256([_tensor_sha256(value) for value in gradients])


def _safetensors_semantic_sha256(path: Path) -> str:
    tensors = load_file(path)
    manifest = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for name, value in sorted(tensors.items())
    ]
    return canonical_sha256(manifest)


def _file_semantic_sha256(path: Path) -> str:
    if path.suffix == ".safetensors":
        return _safetensors_semantic_sha256(path)
    if path.suffix == ".json":
        try:
            payload = json_loads_unique(path.read_text(encoding="utf-8"))
            return canonical_sha256(payload)
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            raise EKFACSmokeError(
                f"invalid Kronfluence JSON artifact: {path}"
            ) from error
    return bytes_sha256(path.read_bytes())


def _store_semantic_sha256(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files or any(item.is_symlink() for item in files):
        raise EKFACSmokeError(f"Kronfluence store is invalid: {path}")
    manifest = [
        {
            "path": item.relative_to(path).as_posix(),
            "semantic_sha256": _file_semantic_sha256(item),
        }
        for item in files
    ]
    return canonical_sha256(manifest)


def _dataset_sha256(plan: EKFACSmokePlan, role: str) -> str:
    return canonical_sha256(plan.fixture[role])


def _inverse_application_sha256(
    factor_sha256: str, query_gradient_sha256: str, scores: tuple[float, ...]
) -> str:
    return canonical_sha256(
        {
            "factor_store_semantic_sha256": factor_sha256,
            "query_gradient_sha256": query_gradient_sha256,
            "score_arguments": _SCORE_IDENTITY,
            "score_vector": list(scores),
        }
    )


_SCORE_IDENTITY = {
    "damping_factor": 0.001,
    "query_gradient_accumulation_steps": 1,
    "query_gradient_low_rank": None,
    "use_full_svd": True,
    "dtype": "float64",
}


def _factor_store_path(run_dir: Path) -> Path:
    root = run_dir / "tiny-ekfac"
    return root / "factors_backend-factors"


def _score_store_path(run_dir: Path, objective_kind: str) -> Path:
    root = run_dir / "tiny-ekfac"
    return root / f"scores_backend-scores-{objective_kind}"


def _score_shared_queries(
    analyzer: Analyzer, fixture: _Fixture, run_dir: Path
) -> tuple[dict[str, tuple[float, ...]], str]:
    factor_path = _factor_store_path(run_dir)
    factor_sha256 = _store_semantic_sha256(factor_path)
    scores: dict[str, tuple[float, ...]] = {}
    for kind in ("behavior_loss", "synthetic_scalar"):
        scores[kind] = _compute_scores(analyzer, fixture, kind)
        if _store_semantic_sha256(factor_path) != factor_sha256:
            raise EKFACSmokeError("EKFAC factor store changed between queries")
    return scores, factor_sha256


def _query_observation(
    plan: EKFACSmokePlan,
    run_dir: Path,
    objective_kind: str,
    factor_sha256: str,
    training_gradient_sha256: str,
    scores: tuple[float, ...],
) -> EKFACQueryObservation:
    query_gradient_sha256 = _gradient_sha256(plan, objective_kind=objective_kind)
    return EKFACQueryObservation(
        objective_kind=objective_kind,
        factor_store_semantic_sha256=factor_sha256,
        training_gradient_sha256=training_gradient_sha256,
        score_store_semantic_sha256=_store_semantic_sha256(
            _score_store_path(run_dir, objective_kind)
        ),
        query_gradient_sha256=query_gradient_sha256,
        inverse_application_sha256=_inverse_application_sha256(
            factor_sha256, query_gradient_sha256, scores
        ),
        score_vector=scores,
    )


def _observation(
    plan: EKFACSmokePlan,
    fixture: _Fixture,
    run_dir: Path,
    scores: dict[str, tuple[float, ...]],
    factor_sha256: str,
) -> EKFACBackendObservation:
    train_sha256 = _gradient_sha256(plan, objective_kind=None)
    queries = tuple(
        _query_observation(
            plan, run_dir, kind, factor_sha256, train_sha256, scores[kind]
        )
        for kind in ("behavior_loss", "synthetic_scalar")
    )
    return EKFACBackendObservation(
        implementation="kronfluence",
        runtime_versions=_runtime_versions(),
        fixture_sha256=plan.fixture_sha256,
        model_sha256=_model_sha256(fixture.model),
        training_data_sha256=_dataset_sha256(plan, "training_dataset"),
        query_data_sha256=_dataset_sha256(plan, "query_dataset"),
        query_observations=queries,
    )


def run_kronfluence_ekfac_smoke(
    plan: EKFACSmokePlan, run_dir: Path
) -> EKFACBackendObservation:
    """Run one exact tiny EKFAC factor fit and pairwise score pass on CPU."""
    if run_dir.exists():
        raise EKFACSmokeError(f"refusing to reuse backend run directory: {run_dir}")
    versions = _runtime_versions()
    if versions != plan.runtime_versions:
        raise EKFACSmokeError("installed EKFAC runtime differs from the plan")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            torch.use_deterministic_algorithms(True)
            fixture = _build_fixture(plan)
            analyzer = _analyzer(fixture, run_dir)
            _fit_factors(analyzer, fixture)
            scores, factor_sha256 = _score_shared_queries(analyzer, fixture, run_dir)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    return _observation(plan, fixture, run_dir, scores, factor_sha256)
