from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import concept_filtering.ekfac_smoke as ekfac_smoke
import concept_filtering.isolated_io as isolated_io
import concept_filtering.smoke as smoke
from concept_filtering.cli import main
from concept_filtering.ekfac_smoke import (
    BackendRunner,
    EKFACBackendObservation,
    EKFACQueryObservation,
    EKFACSmokeError,
    execute_ekfac_backend_smoke,
    load_ekfac_smoke_plan,
)
from concept_filtering.serialization import bytes_sha256, canonical_sha256

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/attribution/fixtures/public-ekfac-backend-smoke-v1.json"
RECONSTRUCTION = (
    ROOT
    / "configs/reconstruction/ekfac/public-rank32-kronfluence-reconstruction-v1.json"
)
RECONSTRUCTION_RECEIPT = (
    ROOT
    / "reports/reconstruction/ekfac"
    / "public-rank32-kronfluence-reconstruction-v1.receipt.json"
)
ATTRIBUTION_PLAN = (
    ROOT / "configs/attribution/public-rank32-causal-gradient-pre-feature-v1.json"
)
RETAINED_SMOKE_RECEIPT = (
    ROOT / "reports/reconstruction/ekfac/public-ekfac-backend-smoke-v1.receipt.json"
)


def _clean_git_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "executor.txt").write_text("reviewed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "executor.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "Test executor",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _observation() -> EKFACBackendObservation:
    plan = load_ekfac_smoke_plan(CONFIG)
    queries = (
        EKFACQueryObservation(
            "behavior_loss",
            "e" * 64,
            "f" * 64,
            "1" * 64,
            "2" * 64,
            "3" * 64,
            plan.expected_score_vectors["behavior_loss"],
        ),
        EKFACQueryObservation(
            "synthetic_scalar",
            "e" * 64,
            "f" * 64,
            "4" * 64,
            "5" * 64,
            "6" * 64,
            plan.expected_score_vectors["synthetic_scalar"],
        ),
    )
    return EKFACBackendObservation(
        implementation="kronfluence",
        runtime_versions=dict(plan.runtime_versions),
        fixture_sha256=plan.fixture_sha256,
        model_sha256="b" * 64,
        training_data_sha256="c" * 64,
        query_data_sha256="d" * 64,
        query_observations=queries,
    )


def _write_mutation(tmp_path: Path, field: str, value: object) -> Path:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["boundaries"][field] = value
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _patch_backend_runner(
    monkeypatch: pytest.MonkeyPatch, runner: BackendRunner
) -> None:
    monkeypatch.setattr(ekfac_smoke, "_load_backend_runner", lambda: runner)


def test_loads_exact_non_scientific_ekfac_smoke_plan() -> None:
    plan = load_ekfac_smoke_plan(CONFIG)

    assert plan.smoke_id == "public-ekfac-backend-smoke-v1"
    assert plan.implementation_version == "1.0.1"
    assert plan.repeat_count == 2
    assert plan.local_backend_smoke_execution is True
    assert plan.scientific_use is False
    assert plan.concept_scoring_allowed is False
    assert plan.scheduler_execution_allowed is False
    assert len(plan.expected_score_vector) == 4
    assert set(plan.expected_score_vectors) == {"behavior_loss", "synthetic_scalar"}


def test_retained_smoke_receipt_binds_clean_executor_and_false_boundaries() -> None:
    source = RETAINED_SMOKE_RECEIPT.read_bytes()
    assert bytes_sha256(source) == (
        "65fcc89eff8c9b0d830e5eb1318166e7ab5d924aec04a780c8022de4c8f3128c"
    )
    payload = json.loads(source)
    receipt_sha256 = payload.pop("receipt_sha256")

    assert canonical_sha256(payload) == receipt_sha256
    assert receipt_sha256 == (
        "6838cef814fb909278dc90c837480189f75ffb4cb467966d388db2f103d1abbc"
    )
    assert payload["releases"]["executor_release"] == (
        "03546ac89ab19c78cf4a79cc016c71310f8bebb4"
    )
    assert payload["releases"]["output_release"] == (
        "40015421661d274d38525286b3bdeb0ed793ce375e7d40383587b59184aaede4"
    )
    assert payload["determinism"]["maximum_observed_score_difference"] == 0.0
    for field in (
        "scientific_use",
        "starts_experiment_timer",
        "provider_execution_allowed",
        "scheduler_execution_allowed",
        "sae_feature_selection_allowed",
        "concept_scoring_allowed",
        "concept_selected_training_allowed",
    ):
        assert payload[field] is False


def test_rejects_aliased_smoke_config(tmp_path: Path) -> None:
    target = tmp_path / "smoke.json"
    target.write_bytes(CONFIG.read_bytes())
    alias = tmp_path / "smoke-alias.json"
    alias.symlink_to(target)

    with pytest.raises(EKFACSmokeError, match="isolated regular file"):
        load_ekfac_smoke_plan(alias)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scientific_use", True),
        ("starts_experiment_timer", 0),
        ("concept_scoring_allowed", None),
        ("local_backend_smoke_execution", False),
    ],
)
def test_rejects_authority_expansion_or_disabled_smoke(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(EKFACSmokeError, match=field):
        load_ekfac_smoke_plan(_write_mutation(tmp_path, field, value))


def test_executes_twice_and_writes_receipt_bound_to_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    calls: list[Path] = []

    def run(_plan: object, run_dir: Path) -> EKFACBackendObservation:
        calls.append(run_dir)
        return _observation()

    _patch_backend_runner(monkeypatch, run)
    output = tmp_path / "receipt.json"
    receipt = execute_ekfac_backend_smoke(
        plan,
        reconstruction_config=RECONSTRUCTION,
        reconstruction_receipt=RECONSTRUCTION_RECEIPT,
        attribution_plan=ATTRIBUTION_PLAN,
        code_revision=revision,
        run_dir=tmp_path / "runs",
        output_path=output,
    )

    assert calls == [tmp_path / "runs/run-1", tmp_path / "runs/run-2"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["receipt_sha256"] == receipt.receipt_sha256
    assert payload["releases"]["executor_release"] == revision
    assert payload["releases"]["coordinator_release"] == revision
    assert payload["releases"]["output_release"] == receipt.output_release
    assert payload["scientific_use"] is False
    assert payload["concept_scoring_allowed"] is False


def test_rejects_nondeterministic_backend_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    observations = [
        _observation(),
        replace(_observation(), model_sha256="9" * 64),
    ]
    _patch_backend_runner(monkeypatch, lambda _plan, _run_dir: observations.pop(0))

    with pytest.raises(EKFACSmokeError, match="deterministic"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rejects_aliased_bound_receipt_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    original = tmp_path / "reconstruction.receipt.original.json"
    original.write_bytes(RECONSTRUCTION_RECEIPT.read_bytes())
    alias = tmp_path / "reconstruction.receipt.json"
    alias.hardlink_to(original)

    with pytest.raises(EKFACSmokeError, match="isolated regular file"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=alias,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rejects_reconstruction_receipt_with_substituted_backend_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = json.loads(RECONSTRUCTION_RECEIPT.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt["implementation_source_revision"] = "1" * 40
    receipt_sha256 = canonical_sha256(receipt)
    receipt["receipt_sha256"] = receipt_sha256
    receipt_path = tmp_path / "reconstruction.receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    smoke_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    smoke_payload["bindings"]["reconstruction_receipt_sha256"] = receipt_sha256
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps(smoke_payload), encoding="utf-8")
    plan = load_ekfac_smoke_plan(smoke_path)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )

    with pytest.raises(EKFACSmokeError, match="receipt contract"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=receipt_path,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rejects_attribution_plan_with_substituted_backend_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attribution = json.loads(ATTRIBUTION_PLAN.read_text(encoding="utf-8"))
    attribution["backend"]["factor_config_sha256"] = "9" * 64
    attribution_path = tmp_path / "attribution.json"
    attribution_path.write_text(json.dumps(attribution), encoding="utf-8")
    smoke_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    smoke_payload["bindings"]["attribution_plan_sha256"] = canonical_sha256(attribution)
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps(smoke_payload), encoding="utf-8")
    plan = load_ekfac_smoke_plan(smoke_path)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )

    with pytest.raises(EKFACSmokeError, match="backend config"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=attribution_path,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rejects_runtime_smoke_authority_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = replace(load_ekfac_smoke_plan(CONFIG), scientific_use=True)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )

    with pytest.raises(EKFACSmokeError, match="canonical file"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rechecks_bound_inputs_after_backend_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    receipt = tmp_path / "reconstruction.receipt.json"
    receipt.write_bytes(RECONSTRUCTION_RECEIPT.read_bytes())
    call_count = 0

    def mutate_after_first_run(
        _plan: object, _run_dir: Path
    ) -> EKFACBackendObservation:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            receipt.write_text("{}\n", encoding="utf-8")
        return _observation()

    _patch_backend_runner(monkeypatch, mutate_after_first_run)
    with pytest.raises(EKFACSmokeError, match="reconstruction receipt"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=receipt,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rechecks_executor_revision_after_backend_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    repository = tmp_path / "repository"
    revision = _clean_git_repo(repository)
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository)
    calls = 0

    def mutate_executor(_plan: object, _run_dir: Path) -> EKFACBackendObservation:
        nonlocal calls
        calls += 1
        if calls == 1:
            (repository / "executor.txt").write_text("changed\n", encoding="utf-8")
        return _observation()

    _patch_backend_runner(monkeypatch, mutate_executor)
    with pytest.raises(smoke.SmokeValidationError, match="clean"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rejects_output_parent_replacement_during_backend_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    original_parent = tmp_path / "output-original"
    calls = 0

    def replace_parent(_plan: object, _run_dir: Path) -> EKFACBackendObservation:
        nonlocal calls
        calls += 1
        if calls == 1:
            output_parent.rename(original_parent)
            output_parent.mkdir()
        return _observation()

    _patch_backend_runner(monkeypatch, replace_parent)
    output = output_parent / "receipt.json"
    with pytest.raises(EKFACSmokeError, match="parent identity changed"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=output,
        )
    assert not output.exists()


def test_rejects_output_parent_replacement_during_receipt_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    original_parent = tmp_path / "output-original"
    original_stage = cast(
        Callable[[int, Path, bytes, str], str],
        vars(isolated_io)["_stage_bytes"],
    )

    def replace_parent(parent: int, path: Path, value: bytes, label: str) -> str:
        output_parent.rename(original_parent)
        output_parent.mkdir()
        return original_stage(parent, path, value, label)

    monkeypatch.setattr(isolated_io, "_stage_bytes", replace_parent)
    _patch_backend_runner(monkeypatch, lambda _plan, _run_dir: _observation())
    output = output_parent / "receipt.json"
    with pytest.raises(EKFACSmokeError, match="parent identity changed"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=output,
        )
    assert not output.exists()


def test_rejects_symlinked_receipt_output_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    real_output_parent = tmp_path / "real-output"
    real_output_parent.mkdir()
    output_parent = tmp_path / "output-alias"
    output_parent.symlink_to(real_output_parent, target_is_directory=True)
    _patch_backend_runner(monkeypatch, lambda _plan, _run_dir: _observation())

    with pytest.raises(EKFACSmokeError, match="plain parent"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=output_parent / "receipt.json",
        )


def test_rejects_symlinked_run_directory_parent_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    real_parent = tmp_path / "real-run-parent"
    real_parent.mkdir()
    alias = tmp_path / "run-parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    _patch_backend_runner(monkeypatch, lambda _plan, _run_dir: _observation())

    with pytest.raises(EKFACSmokeError, match="plain parent"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=alias / "runs",
            output_path=tmp_path / "receipt.json",
        )
    assert not (real_parent / "runs").exists()


def test_rejects_symlinked_output_ancestor_before_creating_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    alias = tmp_path / "output-parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    _patch_backend_runner(monkeypatch, lambda _plan, _run_dir: _observation())

    with pytest.raises(EKFACSmokeError, match="plain parent"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=alias / "nested" / "receipt.json",
        )
    assert not (real_parent / "nested").exists()


def test_rejects_score_outside_frozen_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    changed = list(plan.expected_score_vector)
    changed[0] += 1e-6
    ordinary, synthetic = _observation().query_observations
    observation = replace(
        _observation(),
        query_observations=(
            replace(ordinary, score_vector=tuple(changed)),
            synthetic,
        ),
    )
    _patch_backend_runner(monkeypatch, lambda _plan, _run_dir: observation)

    with pytest.raises(EKFACSmokeError, match="expected score"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rejects_query_measurement_that_changes_shared_factors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_ekfac_smoke_plan(CONFIG)
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    ordinary, synthetic = _observation().query_observations
    observation = replace(
        _observation(),
        query_observations=(
            ordinary,
            replace(synthetic, factor_store_semantic_sha256="9" * 64),
        ),
    )
    _patch_backend_runner(monkeypatch, lambda _plan, _run_dir: observation)

    with pytest.raises(EKFACSmokeError, match="reuse factors"):
        execute_ekfac_backend_smoke(
            plan,
            reconstruction_config=RECONSTRUCTION,
            reconstruction_receipt=RECONSTRUCTION_RECEIPT,
            attribution_plan=ATTRIBUTION_PLAN,
            code_revision=revision,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_cli_reports_missing_attribution_extra_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )

    def missing_backend(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'kronfluence'")

    monkeypatch.setattr(ekfac_smoke.importlib, "import_module", missing_backend)
    exit_code = main(
        [
            "run-ekfac-backend-smoke",
            str(CONFIG),
            "--reconstruction-config",
            str(RECONSTRUCTION),
            "--reconstruction-receipt",
            str(RECONSTRUCTION_RECEIPT),
            "--attribution-plan",
            str(ATTRIBUTION_PLAN),
            "--code-revision",
            revision,
            "--run-dir",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "install the attribution extra" in captured.err
    assert "Traceback" not in captured.err


def test_runs_real_kronfluence_ekfac_factor_to_score_path(tmp_path: Path) -> None:
    pytest.importorskip("kronfluence")
    import torch

    from concept_filtering.ekfac_kronfluence import run_kronfluence_ekfac_smoke

    plan = load_ekfac_smoke_plan(CONFIG)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    observation = run_kronfluence_ekfac_smoke(plan, tmp_path / "run")

    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    assert observation.implementation == "kronfluence"
    assert observation.runtime_versions == plan.runtime_versions
    assert observation.fixture_sha256 == plan.fixture_sha256
    ordinary, synthetic = observation.query_observations
    assert ordinary.score_vector == pytest.approx(
        plan.expected_score_vectors["behavior_loss"], abs=plan.absolute_tolerance, rel=0
    )
    assert synthetic.score_vector == pytest.approx(
        plan.expected_score_vectors["synthetic_scalar"],
        abs=plan.absolute_tolerance,
        rel=0,
    )
    assert ordinary.factor_store_semantic_sha256 == (
        synthetic.factor_store_semantic_sha256
    )
    assert ordinary.training_gradient_sha256 == synthetic.training_gradient_sha256
    assert ordinary.query_gradient_sha256 != synthetic.query_gradient_sha256
    hashes = {
        observation.model_sha256,
        observation.training_data_sha256,
        observation.query_data_sha256,
        ordinary.factor_store_semantic_sha256,
        ordinary.score_store_semantic_sha256,
        ordinary.training_gradient_sha256,
        ordinary.query_gradient_sha256,
        ordinary.inverse_application_sha256,
        synthetic.score_store_semantic_sha256,
        synthetic.query_gradient_sha256,
        synthetic.inverse_application_sha256,
    }
    assert len(hashes) == 11
    assert all(len(value) == 64 for value in hashes)


def test_restores_torch_determinism_after_backend_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("kronfluence")
    import torch

    import concept_filtering.ekfac_kronfluence as backend

    plan = load_ekfac_smoke_plan(CONFIG)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def fail_fixture(_plan: object) -> object:
        raise RuntimeError("fixture failed")

    monkeypatch.setattr(backend, "_build_fixture", fail_fixture)
    with pytest.raises(RuntimeError, match="fixture failed"):
        backend.run_kronfluence_ekfac_smoke(plan, tmp_path / "run")

    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only


def test_backend_rejects_factor_store_change_between_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("kronfluence")
    import concept_filtering.ekfac_kronfluence as backend

    plan = load_ekfac_smoke_plan(CONFIG)
    original = cast(Callable[[Path], str], vars(backend)["_store_semantic_sha256"])
    factor_reads = 0

    def changing_factor(path: Path) -> str:
        nonlocal factor_reads
        if path.name == "factors_backend-factors":
            factor_reads += 1
            return ("a" if factor_reads == 1 else "b") * 64
        return original(path)

    monkeypatch.setattr(backend, "_store_semantic_sha256", changing_factor)
    with pytest.raises(EKFACSmokeError, match="factor store changed"):
        backend.run_kronfluence_ekfac_smoke(plan, tmp_path / "run")


def test_cli_runs_real_backend_twice_without_scientific_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("kronfluence")
    revision = _clean_git_repo(tmp_path / "repository")
    monkeypatch.setattr(
        smoke, "source_repository_root", lambda: tmp_path / "repository"
    )
    output = tmp_path / "receipt.json"

    exit_code = main(
        [
            "run-ekfac-backend-smoke",
            str(CONFIG),
            "--reconstruction-config",
            str(RECONSTRUCTION),
            "--reconstruction-receipt",
            str(RECONSTRUCTION_RECEIPT),
            "--attribution-plan",
            str(ATTRIBUTION_PLAN),
            "--code-revision",
            revision,
            "--run-dir",
            str(tmp_path / "runs"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["scientific_use"] is False
    assert summary["concept_scoring_allowed"] is False
    assert summary["repeat_count"] == 2
    assert len(summary["output_release"]) == 64
    assert output.exists()
