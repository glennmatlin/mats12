from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

import concept_filtering.cli as cli
import concept_filtering.smoke as smoke
from concept_filtering.cli import main
from concept_filtering.sae_functional_smoke import (
    ArtifactSpec,
    SAEFunctionalObservation,
    SAEFunctionalSmokeError,
    TensorObservation,
    execute_sae_functional_smoke,
    load_sae_functional_smoke,
)
from concept_filtering.sae_functional_transformers import (
    ProbeExecution,
    run_transformers_sae_smoke,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/reconstruction/sae/public-layer16-functional-smoke-v1.json"


def _mutated_config(tmp_path: Path, payload: object) -> Path:
    root = tmp_path / "reconstruction"
    shutil.copytree(ROOT / "configs/reconstruction", root)
    candidate = root / "sae/public-layer16-functional-smoke-v1.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    return candidate


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


def _fake_observation() -> SAEFunctionalObservation:
    plan = load_sae_functional_smoke(CONFIG)
    tensors = {
        name: TensorObservation(spec.shape, spec.dtype, True, character * 64)
        for (name, spec), character in zip(
            plan.probe.expected_tensors.items(), "abc", strict=True
        )
    }
    return SAEFunctionalObservation(
        artifact_sha256={
            role: dict(spec.files) for role, spec in plan.artifacts.items()
        },
        runtime_versions=dict(plan.runtime.versions),
        input_token_ids=plan.probe.input_token_ids,
        active_adapter=plan.probe.active_adapter,
        resolved_hook_name=plan.probe.resolved_hook_name,
        sae_architecture=plan.probe.sae_architecture,
        model_forward_count=1,
        hook_capture_count=1,
        encode_count=1,
        decode_count=1,
        tensors=tensors,
    )


def test_loads_frozen_non_scientific_functional_smoke() -> None:
    plan = load_sae_functional_smoke(CONFIG)

    assert plan.smoke_id == "public-layer16-functional-smoke-v1"
    assert plan.scientific_use is False
    assert plan.starts_experiment_timer is False
    assert plan.functional_execution is True
    assert plan.feature_selection is False
    assert plan.intervention_execution is False
    assert plan.provider_execution is False
    assert plan.probe.input_token_ids == (9906,)
    assert plan.probe.hook_name == "model.layers.16"
    assert len(plan.config_sha256) == 64


def test_inference_extra_and_lock_pin_saelens_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    inference = project["project"]["optional-dependencies"]["inference"]
    assert "sae-lens==6.42.0" in inference
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] == "sae-lens"
    }
    assert versions == {"sae-lens": "6.42.0"}


def test_rejects_non_boolean_tokenizer_policy(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["probe"]["add_special_tokens"] = "false"

    with pytest.raises(SAEFunctionalSmokeError, match="add_special_tokens"):
        load_sae_functional_smoke(_mutated_config(tmp_path, payload))


def test_rejects_artifact_identity_not_bound_to_organism(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["artifacts"]["adapter"]["revision"] = "0" * 40

    with pytest.raises(SAEFunctionalSmokeError, match="adapter artifact"):
        load_sae_functional_smoke(_mutated_config(tmp_path, payload))


def test_rejects_unsupported_functional_runtime(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["runtime"]["device"] = "tpu"

    with pytest.raises(SAEFunctionalSmokeError, match=r"runtime\.device"):
        load_sae_functional_smoke(_mutated_config(tmp_path, payload))


def test_executes_once_and_writes_exclusive_boundary_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_sae_functional_smoke(CONFIG)
    repository_root = tmp_path / "repository"
    revision = _clean_git_repo(repository_root)
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository_root)
    output = tmp_path / "receipt.json"

    receipt = execute_sae_functional_smoke(
        plan,
        code_revision=revision,
        output_path=output,
        run=lambda _: _fake_observation(),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["receipt_sha256"] == receipt.receipt_sha256
    assert payload["scientific_use"] is False
    assert payload["starts_experiment_timer"] is False
    assert payload["functional_execution"] is True
    assert payload["feature_selection"] is False
    assert payload["intervention_execution"] is False
    assert payload["provider_execution"] is False
    with pytest.raises(SAEFunctionalSmokeError, match="overwrite"):
        execute_sae_functional_smoke(
            plan,
            code_revision=revision,
            output_path=output,
            run=lambda _: _fake_observation(),
        )


def test_runtime_binds_exact_artifact_bytes_around_probe(tmp_path: Path) -> None:
    plan = load_sae_functional_smoke(CONFIG)
    roots = {role: tmp_path / role for role in ("base", "adapter", "sae")}
    artifacts: dict[str, ArtifactSpec] = {}
    for role, root in roots.items():
        root.mkdir()
        content = f"{role} bytes".encode()
        (root / "artifact.bin").write_bytes(content)
        original = plan.artifacts[role]
        artifacts[role] = ArtifactSpec(
            original.repository,
            original.revision,
            {"artifact.bin": sha256(content).hexdigest()},
        )
    observation = _fake_observation()
    versions = {"runtime": "1"}
    runtime_plan = replace(
        plan,
        artifacts=artifacts,
        runtime=replace(plan.runtime, versions=versions),
    )
    execution = ProbeExecution(
        input_token_ids=observation.input_token_ids,
        active_adapter=observation.active_adapter,
        resolved_hook_name=observation.resolved_hook_name,
        sae_architecture=observation.sae_architecture,
        model_forward_count=1,
        hook_capture_count=1,
        encode_count=1,
        decode_count=1,
        tensors=observation.tensors,
    )

    def fake_execute(
        _plan: object, _base: Path, _adapter: Path, _sae: Path
    ) -> ProbeExecution:
        return execution

    result = run_transformers_sae_smoke(
        runtime_plan,
        base_snapshot=roots["base"],
        adapter_snapshot=roots["adapter"],
        sae_directory=roots["sae"],
        execute_probe=fake_execute,
        version_reader=lambda: versions,
    )

    assert result.artifact_sha256 == {
        role: spec.files for role, spec in artifacts.items()
    }
    assert result.runtime_versions == versions


def test_cli_runs_functional_smoke_and_prints_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    revision = _clean_git_repo(repository_root)
    monkeypatch.setattr(smoke, "source_repository_root", lambda: repository_root)

    def fake_runtime(
        _plan: object,
        *,
        base_snapshot: Path,
        adapter_snapshot: Path,
        sae_directory: Path,
    ) -> SAEFunctionalObservation:
        del base_snapshot, adapter_snapshot, sae_directory
        return _fake_observation()

    monkeypatch.setattr(cli, "run_transformers_sae_smoke", fake_runtime)
    output = tmp_path / "receipt.json"

    exit_code = main(
        [
            "run-sae-functional-smoke",
            str(CONFIG),
            "--code-revision",
            revision,
            "--base-snapshot",
            str(tmp_path / "base"),
            "--adapter-snapshot",
            str(tmp_path / "adapter"),
            "--sae-directory",
            str(tmp_path / "sae"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["functional_execution"] is True
    assert summary["scientific_use"] is False
    assert summary["starts_experiment_timer"] is False
    assert len(summary["receipt_sha256"]) == 64
