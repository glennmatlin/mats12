from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, cast

import pytest

import concept_filtering.ekfac_rank32_smoke as rank32_smoke
import concept_filtering.isolated_io as isolated_io
from concept_filtering.cli import main
from concept_filtering.ekfac_rank32_smoke import (
    Rank32EKFACObservation,
    Rank32EKFACSmokeError,
    Rank32EKFACSmokePlan,
    Rank32RuntimePaths,
    execute_rank32_ekfac_smoke,
    load_rank32_ekfac_smoke_plan,
)
from concept_filtering.isolated_io import IsolatedIOError, sha256_isolated_file
from concept_filtering.serialization import bytes_sha256, canonical_sha256

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/attribution/public-rank32-ekfac-bounded-smoke-v1.json"
QUERY_SAMPLE = (
    ROOT
    / "reports/reconstruction/ekfac/inputs/public-rank32-validation-query-0082.json"
)
RETAINED_RECEIPT = (
    ROOT
    / "reports/reconstruction/ekfac/public-rank32-ekfac-bounded-smoke-v1.receipt.json"
)


def _mutated_config(tmp_path: Path, mutate: object) -> Path:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_exact_rank32_bounded_smoke_plan() -> None:
    plan = load_rank32_ekfac_smoke_plan(CONFIG)

    assert plan.smoke_id == "public-rank32-ekfac-bounded-smoke-v1"
    assert plan.config_sha256 == (
        "fb83c621ad0d3c08698f3496ab3c6610b72c58ebd666ab1acb50d794bf677a40"
    )
    assert plan.tracked_modules == (
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.default",
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.default",
    )
    assert plan.expected_adapter_linear_modules == 448
    assert plan.training_example["row_ordinal"] == 17_763
    assert plan.query_example["index"] == 82
    assert plan.factor_arguments["fisher"] == (
        "true_fisher_sampled_from_model_predictions"
    )
    assert plan.factor_arguments["covariance_max_examples"] == 1
    assert plan.score_arguments["query_gradient_low_rank"] == 64
    assert plan.runtime["device"] == "cpu"
    assert plan.repeat_count == 2
    assert plan.bounded_rank32_smoke_execution is True
    assert plan.scientific_use is False


def test_bound_rank32_query_sample_retains_exact_bytes_and_identity() -> None:
    plan = load_rank32_ekfac_smoke_plan(CONFIG)
    value = QUERY_SAMPLE.read_bytes()
    payload = json.loads(value)

    assert sha256(value).hexdigest() == plan.bindings["generation_sample"].file_sha256
    assert payload["sample_sha256"] == plan.query_example["sample_sha256"]


def test_rejects_rank32_smoke_module_substitution(tmp_path: Path) -> None:
    def mutate(payload: dict[str, object]) -> None:
        topology = payload["topology"]
        assert isinstance(topology, dict)
        modules = cast(list[object], topology["tracked_modules"])
        assert isinstance(modules, list)
        assert isinstance(modules[0], str)
        modules[0] = modules[0].replace("k_proj", "q_proj")

    with pytest.raises(Rank32EKFACSmokeError, match="tracked_modules"):
        load_rank32_ekfac_smoke_plan(_mutated_config(tmp_path, mutate))


def test_rejects_rank32_smoke_profile_substitution(tmp_path: Path) -> None:
    def mutate(payload: dict[str, object]) -> None:
        arguments = payload["factor_arguments"]
        assert isinstance(arguments, dict)
        arguments["use_empirical_fisher"] = True

    with pytest.raises(Rank32EKFACSmokeError, match="factor_arguments"):
        load_rank32_ekfac_smoke_plan(_mutated_config(tmp_path, mutate))


def test_rejects_self_consistent_rank32_artifact_substitution(tmp_path: Path) -> None:
    def mutate(payload: dict[str, object]) -> None:
        artifacts = cast(dict[str, object], payload["artifacts"])
        adapter = cast(dict[str, object], artifacts["adapter"])
        adapter["revision"] = "a" * 40

    with pytest.raises(Rank32EKFACSmokeError, match="canonical identity"):
        load_rank32_ekfac_smoke_plan(_mutated_config(tmp_path, mutate))


@pytest.mark.parametrize(
    "field",
    [
        "scientific_use",
        "starts_experiment_timer",
        "sae_feature_selection_allowed",
        "concept_scoring_allowed",
        "provider_execution_allowed",
        "scheduler_execution_allowed",
        "concept_selected_training_allowed",
        "training_execution_allowed",
    ],
)
def test_rejects_rank32_smoke_authority_expansion(tmp_path: Path, field: str) -> None:
    def mutate(payload: dict[str, object]) -> None:
        boundaries = payload["boundaries"]
        assert isinstance(boundaries, dict)
        boundaries[field] = True

    with pytest.raises(Rank32EKFACSmokeError, match=field):
        load_rank32_ekfac_smoke_plan(_mutated_config(tmp_path, mutate))


def test_rejects_aliased_rank32_smoke_config(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(CONFIG.read_bytes())
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)

    with pytest.raises(Rank32EKFACSmokeError, match="isolated regular file"):
        load_rank32_ekfac_smoke_plan(alias)


def test_streams_isolated_artifact_hash_without_accepting_aliases(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bounded artifact")

    assert (
        sha256_isolated_file(artifact, "artifact")
        == sha256(b"bounded artifact").hexdigest()
    )
    alias = tmp_path / "alias.bin"
    alias.symlink_to(artifact)
    with pytest.raises(IsolatedIOError, match="isolated regular file"):
        sha256_isolated_file(alias, "artifact")


def test_streamed_artifact_hash_rejects_hardlinks_and_fifos(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bounded artifact")
    hardlink = tmp_path / "hardlink.bin"
    os.link(artifact, hardlink)

    with pytest.raises(IsolatedIOError, match="isolated regular file"):
        sha256_isolated_file(artifact, "artifact")
    fifo = tmp_path / "artifact.fifo"
    os.mkfifo(fifo)
    with pytest.raises(IsolatedIOError, match="isolated regular file"):
        sha256_isolated_file(fifo, "artifact")


def test_streamed_artifact_hash_rejects_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bounded artifact")
    original = cast(
        Callable[[BinaryIO], str],
        getattr(isolated_io, "_stream_sha256"),  # noqa: B009
    )

    def swap(handle: BinaryIO) -> str:
        digest = original(handle)
        artifact.rename(tmp_path / "original.bin")
        artifact.write_bytes(b"replacement")
        return digest

    monkeypatch.setattr(isolated_io, "_stream_sha256", swap)
    with pytest.raises(IsolatedIOError, match="identity changed"):
        sha256_isolated_file(artifact, "artifact")


def _observation() -> Rank32EKFACObservation:
    plan = load_rank32_ekfac_smoke_plan(CONFIG)
    return Rank32EKFACObservation(
        runtime_versions=tuple(
            sorted(cast(dict[str, str], plan.runtime["versions"]).items())
        ),
        artifact_manifest_sha256=plan.artifact_manifest_sha256,
        topology_sha256=plan.topology_sha256,
        tracked_parameter_sha256="1" * 64,
        tracked_module_shapes=(
            (plan.tracked_modules[0], (32, 4096)),
            (plan.tracked_modules[1], (4096, 32)),
        ),
        training_input_sha256=plan.training_input_sha256,
        query_input_sha256=plan.query_input_sha256,
        factor_store_semantic_sha256="2" * 64,
        training_gradient_sha256="3" * 64,
        query_gradient_sha256="4" * 64,
        score_store_semantic_sha256="5" * 64,
        inverse_application_sha256="6" * 64,
        score=0.25,
    )


def _manifest(_plan: Rank32EKFACSmokePlan, _paths: Rank32RuntimePaths) -> str:
    return "a" * 64


def _revision(value: str) -> str:
    return value


def _validate_plan(_plan: Rank32EKFACSmokePlan) -> None:
    return None


def _identity_stage(
    _plan: Rank32EKFACSmokePlan, run_paths: Rank32RuntimePaths, _run_dir: Path
) -> Rank32RuntimePaths:
    return run_paths


def _patch_executor_seams(
    monkeypatch: pytest.MonkeyPatch, *, stage_identity: bool = True
) -> None:
    monkeypatch.setattr(rank32_smoke, "_verify_runtime_inputs", _manifest)
    monkeypatch.setattr(rank32_smoke, "verify_executor_revision", _revision)
    monkeypatch.setattr(rank32_smoke, "_validate_plan_identity", _validate_plan)
    if stage_identity:
        monkeypatch.setattr(rank32_smoke, "_stage_runtime_paths", _identity_stage)


def test_executes_rank32_backend_twice_and_publishes_false_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_rank32_ekfac_smoke_plan(CONFIG)
    paths = Rank32RuntimePaths(
        repository_root=ROOT,
        base_snapshot=tmp_path / "base",
        adapter_snapshot=tmp_path / "adapter",
        training_sample=tmp_path / "train.parquet",
        training_package=tmp_path / "package.jsonl",
    )
    calls: list[Path] = []

    def run(_plan: object, _paths: object, run_dir: Path) -> Rank32EKFACObservation:
        calls.append(run_dir)
        return _observation()

    monkeypatch.setattr(rank32_smoke, "_load_backend_runner", lambda: run)
    _patch_executor_seams(monkeypatch)
    output = tmp_path / "receipt.json"
    receipt = execute_rank32_ekfac_smoke(
        plan,
        paths=paths,
        code_revision="b" * 40,
        run_dir=tmp_path / "runs",
        output_path=output,
    )

    assert calls == [tmp_path / "runs/run-1", tmp_path / "runs/run-2"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["receipt_sha256"] == receipt.receipt_sha256
    assert payload["determinism"]["maximum_observed_score_difference"] == 0.0
    assert payload["bounded_rank32_smoke_execution"] is True
    identities = payload["releases"]["input_artifact_identities"]
    assert identities["base_snapshot"] == plan.artifacts["base"].semantic_sha256
    assert identities["adapter_snapshot"] == (plan.artifacts["adapter"].semantic_sha256)
    for field in (
        "scientific_use",
        "starts_experiment_timer",
        "sae_feature_selection_allowed",
        "concept_scoring_allowed",
        "provider_execution_allowed",
        "scheduler_execution_allowed",
        "concept_selected_training_allowed",
        "training_execution_allowed",
    ):
        assert payload[field] is False


def test_rejects_rank32_inverse_identity_drift_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_rank32_ekfac_smoke_plan(CONFIG)
    observations = iter(
        (_observation(), replace(_observation(), inverse_application_sha256="7" * 64))
    )

    def run(
        _plan: Rank32EKFACSmokePlan, _paths: Rank32RuntimePaths, _run_dir: Path
    ) -> Rank32EKFACObservation:
        return next(observations)

    monkeypatch.setattr(rank32_smoke, "_load_backend_runner", lambda: run)
    _patch_executor_seams(monkeypatch)
    paths = Rank32RuntimePaths(
        ROOT,
        tmp_path / "base",
        tmp_path / "adapter",
        tmp_path / "sample",
        tmp_path / "package",
    )

    with pytest.raises(Rank32EKFACSmokeError, match="not deterministic"):
        execute_rank32_ekfac_smoke(
            plan,
            paths=paths,
            code_revision="b" * 40,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def test_rejects_rank32_tracked_shape_drift_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_rank32_ekfac_smoke_plan(CONFIG)
    changed = replace(
        _observation(),
        tracked_module_shapes=(
            (plan.tracked_modules[0], (64, 4096)),
            (plan.tracked_modules[1], (4096, 64)),
        ),
    )

    def run(
        _plan: Rank32EKFACSmokePlan, _paths: Rank32RuntimePaths, _run_dir: Path
    ) -> Rank32EKFACObservation:
        return changed

    monkeypatch.setattr(rank32_smoke, "_load_backend_runner", lambda: run)
    _patch_executor_seams(monkeypatch)
    paths = Rank32RuntimePaths(
        ROOT,
        tmp_path / "base",
        tmp_path / "adapter",
        tmp_path / "sample",
        tmp_path / "package",
    )

    with pytest.raises(Rank32EKFACSmokeError, match="tracked module shapes"):
        execute_rank32_ekfac_smoke(
            plan,
            paths=paths,
            code_revision="b" * 40,
            run_dir=tmp_path / "runs",
            output_path=tmp_path / "receipt.json",
        )


def _fake_snapshot_identity(
    role: str, _snapshot: Path, _artifact: rank32_smoke.SnapshotBinding
) -> tuple[str, str]:
    return role, "f" * 64


def _fixture_snapshot(
    tmp_path: Path, names: tuple[str, ...]
) -> tuple[Path, rank32_smoke.SnapshotBinding]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    files: list[tuple[str, str]] = []
    for index, name in enumerate(names):
        value = f"artifact-{index}".encode()
        (snapshot / name).write_bytes(value)
        files.append((name, sha256(value).hexdigest()))
    return snapshot, rank32_smoke.SnapshotBinding("repo/model", "c" * 40, tuple(files))


def test_rejects_unexpected_snapshot_loader_file(tmp_path: Path) -> None:
    snapshot, binding = _fixture_snapshot(tmp_path, ("config.json", "model.bin"))
    verify = rank32_smoke._verify_snapshot  # pyright: ignore[reportPrivateUsage]
    role, digest = verify("base", snapshot, binding)

    assert role == "base"
    assert len(digest) == 64
    (snapshot / "chat_template.jinja").write_bytes(b"{{ injected }}")
    with pytest.raises(Rank32EKFACSmokeError, match="unexpected loader"):
        verify("base", snapshot, binding)


def test_rejects_unexpected_snapshot_subdirectory_and_missing_file(
    tmp_path: Path,
) -> None:
    snapshot, binding = _fixture_snapshot(tmp_path, ("config.json", "model.bin"))
    verify = rank32_smoke._verify_snapshot  # pyright: ignore[reportPrivateUsage]
    (snapshot / "extra").mkdir()

    with pytest.raises(Rank32EKFACSmokeError, match="unexpected loader"):
        verify("base", snapshot, binding)
    (snapshot / "extra").rmdir()
    (snapshot / "model.bin").unlink()
    with pytest.raises(Rank32EKFACSmokeError, match="missing"):
        verify("base", snapshot, binding)


def test_stages_verified_snapshot_copies_for_consumption(tmp_path: Path) -> None:
    snapshot, binding = _fixture_snapshot(tmp_path, ("config.json", "model.bin"))
    stage = rank32_smoke._stage_snapshot  # pyright: ignore[reportPrivateUsage]
    staged = stage("base", snapshot, binding, tmp_path / "staged/base")

    assert staged == tmp_path / "staged/base"
    for name, _ in binding.files:
        assert (staged / name).read_bytes() == (snapshot / name).read_bytes()


def test_rejects_swap_then_restore_during_snapshot_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, binding = _fixture_snapshot(tmp_path, ("config.json",))
    copy = rank32_smoke._copy_snapshot_file  # pyright: ignore[reportPrivateUsage]

    def swap_then_restore(source: Path, destination: Path, label: str) -> None:
        genuine = source.read_bytes()
        source.write_bytes(b"swapped loader payload")
        copy(source, destination, label)
        source.write_bytes(genuine)

    monkeypatch.setattr(rank32_smoke, "_copy_snapshot_file", swap_then_restore)
    stage = rank32_smoke._stage_snapshot  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(Rank32EKFACSmokeError, match="SHA-256 differs"):
        stage("base", snapshot, binding, tmp_path / "staged/base")


def test_backend_consumes_pinned_staged_snapshot_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_rank32_ekfac_smoke_plan(CONFIG)
    paths = Rank32RuntimePaths(
        repository_root=ROOT,
        base_snapshot=tmp_path / "base",
        adapter_snapshot=tmp_path / "adapter",
        training_sample=tmp_path / "train.parquet",
        training_package=tmp_path / "package.jsonl",
    )
    for directory, artifact in (
        (paths.base_snapshot, plan.artifacts["base"]),
        (paths.adapter_snapshot, plan.artifacts["adapter"]),
    ):
        directory.mkdir()
        for name, _ in artifact.files:
            (directory / name).write_bytes(b"loader-input")
    run_dir = tmp_path / "runs"
    target = paths.base_snapshot / "config.json"

    def run(
        _plan: Rank32EKFACSmokePlan, run_paths: Rank32RuntimePaths, _run_dir: Path
    ) -> Rank32EKFACObservation:
        assert run_paths.base_snapshot.is_relative_to(run_dir)
        assert run_paths.adapter_snapshot.is_relative_to(run_dir)
        genuine = target.read_bytes()
        target.write_bytes(b"swapped loader payload")
        staged = (run_paths.base_snapshot / "config.json").read_bytes()
        target.write_bytes(genuine)
        assert staged == b"loader-input"
        return _observation()

    monkeypatch.setattr(rank32_smoke, "_load_backend_runner", lambda: run)
    _patch_executor_seams(monkeypatch, stage_identity=False)
    monkeypatch.setattr(rank32_smoke, "_verify_snapshot", _fake_snapshot_identity)
    receipt = execute_rank32_ekfac_smoke(
        plan,
        paths=paths,
        code_revision="b" * 40,
        run_dir=run_dir,
        output_path=tmp_path / "receipt.json",
    )

    assert receipt.output_release


def _kron_paths(tmp_path: Path) -> Rank32RuntimePaths:
    return Rank32RuntimePaths(
        ROOT,
        tmp_path / "base",
        tmp_path / "adapter",
        tmp_path / "sample",
        tmp_path / "package",
    )


def _patch_kron_state_seams(
    monkeypatch: pytest.MonkeyPatch, plan: Rank32EKFACSmokePlan
) -> None:
    import concept_filtering.ekfac_rank32_kronfluence as kron

    versions = tuple(sorted(cast(dict[str, str], plan.runtime["versions"]).items()))

    def fake_versions() -> tuple[tuple[str, str], ...]:
        return versions

    def fake_weights(_path: Path) -> None:
        return None

    def fake_inputs(_plan: Rank32EKFACSmokePlan, _paths: Rank32RuntimePaths) -> object:
        return object()

    def fake_model(_plan: Rank32EKFACSmokePlan, _paths: Rank32RuntimePaths) -> object:
        return object()

    def fake_topology(_plan: Rank32EKFACSmokePlan, _model: object) -> None:
        return None

    monkeypatch.setattr(kron, "_runtime_versions", fake_versions)
    monkeypatch.setattr(kron, "_validate_adapter_weights", fake_weights)
    monkeypatch.setattr(kron, "_prepare_inputs", fake_inputs)
    monkeypatch.setattr(kron, "_load_model", fake_model)
    monkeypatch.setattr(kron, "_validate_loaded_topology", fake_topology)


def test_rank32_backend_restores_rng_and_determinism_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    import concept_filtering.ekfac_rank32_kronfluence as kron

    plan = load_rank32_ekfac_smoke_plan(CONFIG)
    _patch_kron_state_seams(monkeypatch, plan)
    expected = _observation()

    def fake_analysis(
        _plan: Rank32EKFACSmokePlan,
        _model: object,
        _inputs: object,
        _run_dir: Path,
    ) -> Rank32EKFACObservation:
        return expected

    monkeypatch.setattr(kron, "_run_analysis", fake_analysis)
    torch.manual_seed(1234)  # pyright: ignore[reportUnknownMemberType]
    state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    result = kron.run_rank32_kronfluence_smoke(
        plan, _kron_paths(tmp_path), tmp_path / "run-a"
    )

    assert result is expected
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.are_deterministic_algorithms_enabled() is deterministic


def test_rank32_backend_restores_rng_and_determinism_state_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    import concept_filtering.ekfac_rank32_kronfluence as kron

    plan = load_rank32_ekfac_smoke_plan(CONFIG)
    _patch_kron_state_seams(monkeypatch, plan)

    def fail(
        _plan: Rank32EKFACSmokePlan,
        _model: object,
        _inputs: object,
        _run_dir: Path,
    ) -> Rank32EKFACObservation:
        raise Rank32EKFACSmokeError("backend failure")

    monkeypatch.setattr(kron, "_run_analysis", fail)
    torch.manual_seed(1234)  # pyright: ignore[reportUnknownMemberType]
    state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()

    with pytest.raises(Rank32EKFACSmokeError, match="backend failure"):
        kron.run_rank32_kronfluence_smoke(
            plan, _kron_paths(tmp_path), tmp_path / "run-b"
        )
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.are_deterministic_algorithms_enabled() is deterministic


def test_rank32_backend_freezes_all_adapter_weight_keys() -> None:
    from concept_filtering.ekfac_rank32_kronfluence import (
        expected_adapter_weight_keys,
    )

    keys = expected_adapter_weight_keys()

    assert len(keys) == 448
    assert len(set(keys)) == 448
    assert "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight" in keys
    assert "base_model.model.model.layers.31.mlp.up_proj.lora_B.weight" in keys


def test_rank32_cli_reports_invalid_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        boundaries = cast(dict[str, object], payload["boundaries"])
        boundaries["scientific_use"] = True

    path = _mutated_config(tmp_path, mutate)
    exit_code = main(
        [
            "run-rank32-ekfac-smoke",
            str(path),
            "--repository-root",
            str(ROOT),
            "--base-snapshot",
            str(tmp_path / "base"),
            "--adapter-snapshot",
            str(tmp_path / "adapter"),
            "--training-sample",
            str(tmp_path / "train.parquet"),
            "--training-package",
            str(tmp_path / "package.jsonl"),
            "--code-revision",
            "a" * 40,
            "--run-dir",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert exit_code == 2
    assert "scientific_use" in capsys.readouterr().err


def test_retained_rank32_receipt_binds_clean_executor_and_false_boundaries() -> None:
    source = RETAINED_RECEIPT.read_bytes()
    assert bytes_sha256(source) == (
        "ca0e635669bff69b1825a319a49c22bae08c8431adecd297c39c1c9dba73071f"
    )
    payload = json.loads(source)
    receipt_sha256 = payload.pop("receipt_sha256")

    assert canonical_sha256(payload) == receipt_sha256
    assert receipt_sha256 == (
        "b103b327bf3e1dc1112b52054efe1b513fab7e80a9cc38f53642534a977119c3"
    )
    assert payload["smoke_id"] == "public-rank32-ekfac-bounded-smoke-v1"
    assert payload["config_sha256"] == (
        "fb83c621ad0d3c08698f3496ab3c6610b72c58ebd666ab1acb50d794bf677a40"
    )
    releases = payload["releases"]
    assert releases["executor_release"] == ("8d4f7f23f5fb6098295c145d5a693f7b9ac36c5b")
    assert releases["coordinator_release"] == releases["executor_release"]
    assert releases["output_release"] == (
        "c413ef1a64f273c17aeedb6025bf22c2df119a260ca148bd37244a7c50968089"
    )
    determinism = payload["determinism"]
    assert determinism["repeat_count"] == 2
    assert determinism["absolute_tolerance"] == 1e-5
    assert determinism["maximum_observed_score_difference"] == 0.0
    assert payload["observation"]["score"] == 31488.0
    assert payload["bounded_rank32_smoke_execution"] is True
    for field in (
        "scientific_use",
        "starts_experiment_timer",
        "provider_execution_allowed",
        "scheduler_execution_allowed",
        "sae_feature_selection_allowed",
        "concept_scoring_allowed",
        "concept_selected_training_allowed",
        "training_execution_allowed",
    ):
        assert payload[field] is False
