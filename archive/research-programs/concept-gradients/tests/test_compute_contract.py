from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import cast

import pytest

from concept_filtering.cli import main
from concept_filtering.serialization import canonical_sha256

_STAGE_IDS = (
    "G2_EKFAC_SMOKE",
    "G3_FACTOR_FIT",
    "G3_TRAINING_GRADIENTS",
    "G3_BEHAVIOR_QUERY_SCORES",
    "G3_ORDINARY_REMOVAL_MANIFEST",
    "B7_TRAIN_REPLICATION_CONDITIONS",
    "B8_EVALUATE_REPLICATION_CONDITIONS",
)
_INPUT_ROLES = {
    "G2_EKFAC_SMOKE": (
        "organism",
        "training_data_package",
        "attribution_plan",
        "ekfac_backend_config",
    ),
    "G3_FACTOR_FIT": (
        "training_data_package",
        "factor_config",
        "parameter_selection",
    ),
    "G3_TRAINING_GRADIENTS": (
        "training_data_package",
        "training_gradient_config",
    ),
    "G3_BEHAVIOR_QUERY_SCORES": (
        "validation_query_bank",
        "refusal_query_bank",
        "score_config",
    ),
    "G3_ORDINARY_REMOVAL_MANIFEST": (
        "training_data_package",
        "selection_policy",
        "condition_multiplicity_decision",
    ),
    "B7_TRAIN_REPLICATION_CONDITIONS": (
        "training_data_package",
        "training_recipe",
        "dense_probe_manifest",
        "matched_random_manifest",
        "condition_multiplicity_decision",
    ),
    "B8_EVALUATE_REPLICATION_CONDITIONS": (
        "evaluator_contract",
        "accepted_evaluator_receipt",
        "validation_prompt_bank",
        "refusal_prompt_bank",
        "condition_multiplicity_decision",
    ),
}
_COMMAND_IDS = {
    "G2_EKFAC_SMOKE": "run-ekfac-backend-smoke",
    "G3_FACTOR_FIT": "fit-ekfac-factors",
    "G3_TRAINING_GRADIENTS": "build-training-gradient-store",
    "G3_BEHAVIOR_QUERY_SCORES": "score-behavior-queries",
    "G3_ORDINARY_REMOVAL_MANIFEST": "freeze-ordinary-ekfac-manifest",
    "B7_TRAIN_REPLICATION_CONDITIONS": "train-replication-conditions",
    "B8_EVALUATE_REPLICATION_CONDITIONS": "evaluate-replication-conditions",
}
_WORK_UNITS = {
    "G2_EKFAC_SMOKE": ["bounded_smoke"],
    "G3_FACTOR_FIT": ["factor_fit"],
    "G3_TRAINING_GRADIENTS": ["training_gradient_shard_0000"],
    "G3_BEHAVIOR_QUERY_SCORES": ["validation", "refusal"],
    "G3_ORDINARY_REMOVAL_MANIFEST": ["validation"],
    "B7_TRAIN_REPLICATION_CONDITIONS": [
        "full_data",
        "dense_probe_removal",
        "matched_random_removal",
        "ordinary_ekfac_removal",
    ],
    "B8_EVALUATE_REPLICATION_CONDITIONS": [
        "full_data",
        "dense_probe_removal",
        "matched_random_removal",
        "ordinary_ekfac_removal",
    ],
}
_FIXTURE_CONTRACT = Path(
    "configs/compute/fixtures/phoenix-no-submit-compute-contract-v1.json"
)
_FIXTURE_DEPLOYMENT_RECEIPT = Path(
    "configs/compute/fixtures/pace-code-release-v3-receipt.json"
)
_PRODUCTION_CONTRACT = Path(
    "configs/compute/public-rank32-phoenix-compute-contract-v1.json"
)
_DETAILED_LEDGER = Path("reports/compute/ledger.jsonl")


def _mutable_object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _mutable_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


def _stage(
    stage_id: str,
    dependencies: list[str],
    *,
    array_size: int = 1,
) -> dict[str, object]:
    input_releases = {
        role: hashlib.sha256(role.encode()).hexdigest()
        for role in _INPUT_ROLES[stage_id]
    }
    return {
        "stage_id": stage_id,
        "name": stage_id.replace("_", " ").title(),
        "command_id": _COMMAND_IDS[stage_id],
        "command_receipt_sha256": hashlib.sha256(
            _COMMAND_IDS[stage_id].encode()
        ).hexdigest(),
        "work_units": _WORK_UNITS[stage_id],
        "dependencies": dependencies,
        "release_tuple": {
            "executor_release": "a" * 40,
            "coordinator_release": "a" * 40,
            "input_producer_releases": input_releases,
            "output_release_policy": "compute-stage-output-release-v1",
            "pre_side_effect_release_gate": "compute-release-tuple-v1",
        },
        "resources": {
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 8,
            "gpus_per_node": 1,
            "gpu_type": "H100",
            "memory_per_node_gib": 64,
            "wall_time_minutes": 60,
            "array_size": array_size,
            "max_concurrency": min(array_size, 2),
        },
        "recovery": {
            "checkpoint_cadence_minutes": 15,
            "max_attempts": 2,
            "retry_policy": "resume-from-verified-checkpoint-v1",
            "retryable_states": ["PREEMPTED", "TIMEOUT", "NODE_FAIL"],
        },
    }


def _valid_contract(receipt_sha256: str) -> dict[str, object]:
    dependencies = {
        "G2_EKFAC_SMOKE": [],
        "G3_FACTOR_FIT": ["G2_EKFAC_SMOKE"],
        "G3_TRAINING_GRADIENTS": ["G3_FACTOR_FIT"],
        "G3_BEHAVIOR_QUERY_SCORES": ["G3_TRAINING_GRADIENTS"],
        "G3_ORDINARY_REMOVAL_MANIFEST": ["G3_BEHAVIOR_QUERY_SCORES"],
        "B7_TRAIN_REPLICATION_CONDITIONS": ["G3_ORDINARY_REMOVAL_MANIFEST"],
        "B8_EVALUATE_REPLICATION_CONDITIONS": ["B7_TRAIN_REPLICATION_CONDITIONS"],
    }
    stages = [
        _stage(
            stage_id,
            dependencies[stage_id],
            array_size=(
                4
                if stage_id.startswith("B")
                else 2
                if stage_id == "G3_BEHAVIOR_QUERY_SCORES"
                else 1
            ),
        )
        for stage_id in _STAGE_IDS
    ]
    return {
        "schema_version": 1,
        "contract_id": "fixture-phoenix-compute-contract-v1",
        "mode": "fixture",
        "site": {
            "cluster": "pace-phoenix",
            "host": "pace-phoenix-agent",
            "account": "gts-example",
            "qos": "inferno",
            "constraint": "gpu-h100",
            "module_names": ["anaconda3", "cuda"],
        },
        "deployment": {
            "receipt_contract": "pace-code-release-v3",
            "origin": "https://github.com/glennmatlin/mats12-concept-filtering.git",
            "revision": "a" * 40,
            "release_root": "/storage/example/code/releases",
            "receipt_sha256": receipt_sha256,
            "tracked_identities": ["pyproject.toml", "uv.lock"],
            "verification_mode": "pace-code-release-v3-verify-only",
        },
        "storage": {
            "shared_project_root": "/storage/example/mats12-concept-filtering",
            "node_local_root": "${TMPDIR}",
            "retained_outputs_relative_root": "runs/retained",
            "checkpoints_relative_root": "runs/checkpoints",
            "logs_relative_root": "runs/logs",
            "cleanup_policy": "node-local-after-receipted-copy-v1",
        },
        "ledger": {
            "summary_path": "docs/COMPUTE_LEDGER.md",
            "detailed_path": "reports/compute/ledger.jsonl",
            "accounting_mode": "allocation-only-sacct-v1",
            "include_failed": True,
            "include_retried": True,
            "include_debug": True,
            "fields": [
                "JobIDRaw",
                "JobName",
                "State",
                "ExitCode",
                "ElapsedRaw",
                "AllocTRES",
            ],
        },
        "stages": stages,
        "boundaries": {
            "offline_plan_allowed": True,
            "deployment_execution_allowed": False,
            "scheduler_submission_allowed": False,
            "provider_execution_allowed": False,
            "destructive_cleanup_allowed": False,
            "scientific_use": False,
            "sae_feature_selection_allowed": False,
            "concept_scoring_allowed": False,
            "concept_selected_training_allowed": False,
        },
    }


def _deployment_receipt() -> dict[str, object]:
    return {
        "receipt_version": 3,
        "contract": "pace-code-release-v3",
        "origin": "https://github.com/glennmatlin/mats12-concept-filtering.git",
        "revision": "a" * 40,
        "deployment_path": "/storage/example/code/releases/" + "a" * 40,
        "tracked_identities": [
            {
                "path": "pyproject.toml",
                "git_object": "c" * 40,
                "object_type": "blob",
            },
            {
                "path": "uv.lock",
                "git_object": "d" * 40,
                "object_type": "blob",
            },
        ],
        "git_tree": "e" * 40,
        "verified_at": "2026-08-16T12:00:00Z",
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    receipt_bytes = json.dumps(_deployment_receipt(), sort_keys=True).encode()
    receipt_path = tmp_path / "deployment-receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    payload = _valid_contract(hashlib.sha256(receipt_bytes).hexdigest())
    config_path = tmp_path / "compute-contract.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path, receipt_path, payload


def _write_mutated_receipt(
    config_path: Path,
    receipt_path: Path,
    payload: dict[str, object],
    receipt: dict[str, object],
) -> None:
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
    receipt_path.write_bytes(receipt_bytes)
    deployment = payload["deployment"]
    assert isinstance(deployment, dict)
    deployment["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_plans_ready_fixture_without_submission(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["contract_id"] == "fixture-phoenix-compute-contract-v1"
    assert summary["contract_sha256"] == canonical_sha256(payload)
    assert summary["ready_for_no_submit"] is True
    assert summary["blocked_fields"] == []
    assert summary["stage_order"] == list(_STAGE_IDS)
    assert set(summary["stage_output_releases"]) == set(_STAGE_IDS)
    assert len(set(summary["stage_output_releases"].values())) == len(_STAGE_IDS)
    assert summary["mode"] == "fixture"
    assert summary["site"]["qos"] == "inferno"
    assert summary["stage_plans"][0]["resources"]["gpus_per_node"] == 1
    assert (
        summary["stage_plans"][0]["output_release"]
        == summary["stage_output_releases"]["G2_EKFAC_SMOKE"]
    )
    assert summary["submission_allowed"] is False


def test_cli_reports_unresolved_production_contract_as_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_contract("VERIFY_ON_PACE")
    payload["mode"] = "production"
    site = payload["site"]
    deployment = payload["deployment"]
    storage = payload["storage"]
    stages = _mutable_list(payload["stages"])
    assert isinstance(site, dict)
    assert isinstance(deployment, dict)
    assert isinstance(storage, dict)
    site["account"] = "VERIFY_ON_PACE"
    deployment["revision"] = "VERIFY_ON_PACE"
    deployment["release_root"] = "VERIFY_ON_PACE"
    storage["shared_project_root"] = "VERIFY_ON_PACE"
    for raw_stage in stages:
        stage = _mutable_object(raw_stage)
        release_tuple = _mutable_object(stage["release_tuple"])
        release_tuple["executor_release"] = "VERIFY_ON_PACE"
        release_tuple["coordinator_release"] = "VERIFY_ON_PACE"
    config_path = tmp_path / "production-contract.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["plan-compute-contract", str(config_path)])

    assert exit_code == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["ready_for_no_submit"] is False
    assert summary["submission_allowed"] is False
    assert "site.account" in summary["blocked_fields"]
    assert "deployment.receipt_file" in summary["blocked_fields"]
    assert "deployment.revision" in summary["blocked_fields"]
    assert "storage.shared_project_root" in summary["blocked_fields"]
    assert all(value is None for value in summary["stage_output_releases"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scheduler_submission_allowed", True),
        ("deployment_execution_allowed", True),
        ("scientific_use", True),
        ("offline_plan_allowed", False),
        ("provider_execution_allowed", 0),
    ],
)
def test_rejects_boundary_authority_or_non_boolean_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    boundaries = payload["boundaries"]
    assert isinstance(boundaries, dict)
    boundaries[field] = value
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["missing-stage", "wrong-edge", "unknown-field"])
def test_rejects_incomplete_or_mutated_stage_graph(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    stages = _mutable_list(payload["stages"])
    if mutation == "missing-stage":
        stages.pop()
    elif mutation == "wrong-edge":
        stage = _mutable_object(stages[-1])
        stage["dependencies"] = ["G3_FACTOR_FIT"]
    else:
        stage = _mutable_object(stages[0])
        stage["partition"] = "invented"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    error = capsys.readouterr().err.lower()
    assert "stage" in error or "dependencies" in error


@pytest.mark.parametrize(
    "mutation",
    [
        "executor-mismatch",
        "coordinator-mismatch",
        "bad-input-release",
        "duplicate-input-release",
        "unnamed-input-releases",
        "bad-release-gate",
        "bad-output-policy",
        "wrong-command-id",
        "bad-command-receipt",
        "duplicate-command-receipt",
        "wrong-work-units",
        "missing-input-role",
        "extra-input-role",
        "zero-cpus",
        "boolean-walltime",
        "missing-node-memory",
        "excess-concurrency",
        "bad-retry-policy",
        "cadence-exceeds-walltime",
    ],
)
def test_rejects_invalid_stage_execution_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    stages = _mutable_list(payload["stages"])
    stage = _mutable_object(stages[0])
    release_tuple = _mutable_object(stage["release_tuple"])
    resources = _mutable_object(stage["resources"])
    recovery = _mutable_object(stage["recovery"])
    if mutation == "executor-mismatch":
        release_tuple["executor_release"] = "f" * 40
    elif mutation == "coordinator-mismatch":
        release_tuple["coordinator_release"] = "f" * 40
    elif mutation == "bad-input-release":
        release_tuple["input_producer_releases"] = {"input": "b" * 40}
    elif mutation == "duplicate-input-release":
        release_tuple["input_producer_releases"] = {
            "first": "b" * 64,
            "second": "b" * 64,
        }
    elif mutation == "unnamed-input-releases":
        release_tuple["input_producer_releases"] = ["b" * 64]
    elif mutation == "bad-release-gate":
        release_tuple["pre_side_effect_release_gate"] = "none"
    elif mutation == "bad-output-policy":
        release_tuple["output_release_policy"] = "copy-input"
    elif mutation == "wrong-command-id":
        stage["command_id"] = "unrelated-command"
    elif mutation == "bad-command-receipt":
        stage["command_receipt_sha256"] = "f" * 40
    elif mutation == "duplicate-command-receipt":
        other_stage = _mutable_object(stages[1])
        other_stage["command_receipt_sha256"] = stage["command_receipt_sha256"]
    elif mutation == "wrong-work-units":
        stage["work_units"] = ["unrelated"]
    elif mutation == "missing-input-role":
        inputs = _mutable_object(release_tuple["input_producer_releases"])
        inputs.pop("organism")
    elif mutation == "extra-input-role":
        inputs = _mutable_object(release_tuple["input_producer_releases"])
        inputs["irrelevant"] = "f" * 64
    elif mutation == "zero-cpus":
        resources["cpus_per_task"] = 0
    elif mutation == "boolean-walltime":
        resources["wall_time_minutes"] = True
    elif mutation == "missing-node-memory":
        resources.pop("memory_per_node_gib")
    elif mutation == "excess-concurrency":
        resources["max_concurrency"] = 2
        resources["array_size"] = 1
    elif mutation == "cadence-exceeds-walltime":
        recovery["checkpoint_cadence_minutes"] = 60
    else:
        recovery["retry_policy"] = "blind-retry"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.parametrize(
    "mutation",
    [
        "bad-mode",
        "boolean-schema-version",
        "bad-host",
        "bad-account-marker",
        "bad-qos",
        "versioned-module",
        "credentialed-origin",
        "relative-release-root",
        "bad-receipt-hash",
        "duplicate-tracked-identity",
        "ambiguous-tracked-identity",
        "relative-shared-root",
        "control-shared-root",
        "noncanonical-shared-root",
        "bad-node-local-root",
        "absolute-retained-root",
        "aliased-relative-root-dot",
        "aliased-relative-root-trailing",
        "nested-relative-root",
        "bad-cleanup-policy",
        "ledger-omits-failures",
        "absolute-ledger-path",
        "bad-ledger-fields",
    ],
)
def test_rejects_invalid_site_deployment_storage_or_ledger_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    site = payload["site"]
    deployment = payload["deployment"]
    storage = payload["storage"]
    ledger = payload["ledger"]
    assert isinstance(site, dict)
    assert isinstance(deployment, dict)
    assert isinstance(storage, dict)
    assert isinstance(ledger, dict)
    mutations: dict[str, tuple[dict[str, object], str, object]] = {
        "bad-mode": (payload, "mode", "live"),
        "boolean-schema-version": (payload, "schema_version", True),
        "bad-host": (site, "host", "login-phoenix"),
        "bad-account-marker": (site, "account", "VERIFY_ON_SITE"),
        "bad-qos": (site, "qos", "debug"),
        "versioned-module": (site, "module_names", ["anaconda3/2026", "cuda"]),
        "credentialed-origin": (
            deployment,
            "origin",
            "https://secret@github.com/glennmatlin/mats12-concept-filtering.git",
        ),
        "relative-release-root": (deployment, "release_root", "releases"),
        "bad-receipt-hash": (deployment, "receipt_sha256", "A" * 64),
        "duplicate-tracked-identity": (
            deployment,
            "tracked_identities",
            ["uv.lock", "uv.lock"],
        ),
        "ambiguous-tracked-identity": (
            deployment,
            "tracked_identities",
            ["pyproject.toml", "src//concept_filtering"],
        ),
        "relative-shared-root": (storage, "shared_project_root", "runs"),
        "control-shared-root": (
            storage,
            "shared_project_root",
            "/storage/example\nother",
        ),
        "noncanonical-shared-root": (
            storage,
            "shared_project_root",
            "/storage/./example",
        ),
        "bad-node-local-root": (storage, "node_local_root", "/tmp"),
        "absolute-retained-root": (
            storage,
            "retained_outputs_relative_root",
            "/retained",
        ),
        "aliased-relative-root-dot": (
            storage,
            "logs_relative_root",
            "runs/./retained",
        ),
        "aliased-relative-root-trailing": (
            storage,
            "logs_relative_root",
            "runs/retained/",
        ),
        "nested-relative-root": (
            storage,
            "logs_relative_root",
            "runs/retained/logs",
        ),
        "bad-cleanup-policy": (storage, "cleanup_policy", "delete-all"),
        "ledger-omits-failures": (ledger, "include_failed", False),
        "absolute-ledger-path": (ledger, "summary_path", "/tmp/ledger.md"),
        "bad-ledger-fields": (ledger, "fields", ["State"]),
    }
    target, field, value = mutations[mutation]
    target[field] = value
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy-version",
        "unknown-field",
        "bad-git-tree",
        "bad-verified-at",
        "identity-path-mismatch",
        "duplicate-identity",
        "bad-object-type",
        "unknown-identity-field",
    ],
)
def test_rejects_self_contradictory_deployment_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    receipt = _deployment_receipt()
    identities = _mutable_list(receipt["tracked_identities"])
    if mutation == "legacy-version":
        receipt["receipt_version"] = 2
    elif mutation == "unknown-field":
        receipt["tree_digest"] = "f" * 64
    elif mutation == "bad-git-tree":
        receipt["git_tree"] = "E" * 40
    elif mutation == "bad-verified-at":
        receipt["verified_at"] = 0
    elif mutation == "identity-path-mismatch":
        identity = _mutable_object(identities[0])
        identity["path"] = "README.md"
    elif mutation == "duplicate-identity":
        identities[1] = identities[0]
    elif mutation == "bad-object-type":
        identity = _mutable_object(identities[0])
        identity["object_type"] = "commit"
    else:
        identity = _mutable_object(identities[0])
        identity["sha256"] = "f" * 64
    _write_mutated_receipt(config_path, receipt_path, payload, receipt)

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert "receipt" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
@pytest.mark.parametrize("target_kind", ["config", "receipt"])
def test_rejects_aliased_contract_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    alias_kind: str,
    target_kind: str,
) -> None:
    config_path, receipt_path, _ = _write_inputs(tmp_path)
    target = config_path if target_kind == "config" else receipt_path
    alias = tmp_path / f"{target_kind}-{alias_kind}.json"
    if alias_kind == "symlink":
        alias.symlink_to(target)
    else:
        os.link(target, alias)
    selected_config = alias if target_kind == "config" else config_path
    selected_receipt = alias if target_kind == "receipt" else receipt_path

    exit_code = main(
        [
            "plan-compute-contract",
            str(selected_config),
            "--deployment-receipt",
            str(selected_receipt),
        ]
    )

    assert exit_code == 2
    assert "isolated regular file" in capsys.readouterr().err


@pytest.mark.parametrize("target_kind", ["config", "receipt"])
def test_rejects_contract_inputs_beneath_symlinked_parent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target_kind: str,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    config_path, receipt_path, _ = _write_inputs(real_parent)
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    selected_config = (
        alias_parent / config_path.name if target_kind == "config" else config_path
    )
    selected_receipt = (
        alias_parent / receipt_path.name if target_kind == "receipt" else receipt_path
    )

    exit_code = main(
        [
            "plan-compute-contract",
            str(selected_config),
            "--deployment-receipt",
            str(selected_receipt),
        ]
    )

    assert exit_code == 2
    assert "plain parent directories" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["name", "checkpoint_cadence_minutes"])
def test_rejects_verify_marker_outside_live_phoenix_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    stages = _mutable_list(payload["stages"])
    stage = _mutable_object(stages[0])
    if field == "name":
        stage[field] = "VERIFY_ON_PACE"
    else:
        recovery = _mutable_object(stage["recovery"])
        recovery[field] = "VERIFY_ON_PACE"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert "VERIFY_ON_PACE" in capsys.readouterr().err


def test_present_receipt_is_not_reported_missing_when_another_field_is_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    payload["mode"] = "production"
    site = payload["site"]
    assert isinstance(site, dict)
    site["account"] = "VERIFY_ON_PACE"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["blocked_fields"] == ["site.account"]


def test_present_receipt_stays_unverified_while_deployment_is_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    payload["mode"] = "production"
    deployment = _mutable_object(payload["deployment"])
    for field in ("revision", "release_root", "receipt_sha256"):
        deployment[field] = "VERIFY_ON_PACE"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 3
    summary = json.loads(capsys.readouterr().out)
    assert "deployment.receipt_file" in summary["blocked_fields"]


@pytest.mark.parametrize("receipt_kind", ["symlink", "fifo"])
def test_rejects_unreadable_receipt_even_when_deployment_is_blocked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    receipt_kind: str,
) -> None:
    receipt_path = tmp_path / "receipt-input"
    if receipt_kind == "symlink":
        receipt_path.symlink_to(_FIXTURE_DEPLOYMENT_RECEIPT.absolute())
    else:
        os.mkfifo(receipt_path)

    exit_code = main(
        [
            "plan-compute-contract",
            str(_PRODUCTION_CONTRACT),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert "isolated regular file" in capsys.readouterr().err


@pytest.mark.parametrize("target_kind", ["config", "receipt"])
def test_rejects_fifo_contract_inputs_without_blocking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target_kind: str,
) -> None:
    config_path, receipt_path, _ = _write_inputs(tmp_path)
    fifo_path = tmp_path / f"{target_kind}.fifo"
    os.mkfifo(fifo_path)
    selected_config = fifo_path if target_kind == "config" else config_path
    selected_receipt = fifo_path if target_kind == "receipt" else receipt_path

    exit_code = main(
        [
            "plan-compute-contract",
            str(selected_config),
            "--deployment-receipt",
            str(selected_receipt),
        ]
    )

    assert exit_code == 2
    assert "isolated regular file" in capsys.readouterr().err


@pytest.mark.parametrize("target_kind", ["config", "receipt"])
def test_rejects_duplicate_json_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target_kind: str,
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    if target_kind == "config":
        source = config_path.read_text(encoding="utf-8")
        config_path.write_text(
            source.replace(
                '"schema_version": 1',
                '"schema_version": 1, "schema_version": 1',
                1,
            ),
            encoding="utf-8",
        )
    else:
        receipt_bytes = receipt_path.read_bytes().replace(
            b'"receipt_version": 3',
            b'"receipt_version": 3, "receipt_version": 3',
            1,
        )
        receipt_path.write_bytes(receipt_bytes)
        deployment = payload["deployment"]
        assert isinstance(deployment, dict)
        deployment["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
        config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert "duplicate JSON key" in capsys.readouterr().err


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_rejects_nonstandard_json_constants_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    constant: str,
) -> None:
    config_path, receipt_path, _ = _write_inputs(tmp_path)
    source = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        source.replace('"name": "G2 Ekfac Smoke"', f'"name": {constant}', 1),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err.startswith("error:")


def test_rejects_excessive_json_nesting_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, receipt_path, _ = _write_inputs(tmp_path)
    source = config_path.read_text(encoding="utf-8")
    nested_value = "[" * 2_000 + "0" + "]" * 2_000
    config_path.write_text(
        source.replace('"name": "G2 Ekfac Smoke"', f'"name": {nested_value}', 1),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cpu_only_manifest_stage_is_explicit_and_valid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    stages = _mutable_list(payload["stages"])
    manifest_stage = _mutable_object(stages[4])
    resources = _mutable_object(manifest_stage["resources"])
    resources["gpus_per_node"] = 0
    resources["gpu_type"] = None
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["ready_for_no_submit"] is True


def test_accepts_separate_behavior_condition_work_units(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    stages = [_mutable_object(item) for item in _mutable_list(payload["stages"])]
    stages[4]["work_units"] = ["validation", "refusal"]
    condition_units = [
        "full_data",
        "validation_dense_probe_removal",
        "validation_matched_random_removal",
        "validation_ordinary_ekfac_removal",
        "refusal_dense_probe_removal",
        "refusal_matched_random_removal",
        "refusal_ordinary_ekfac_removal",
    ]
    for stage in stages[5:]:
        stage["work_units"] = condition_units
    for stage, array_size in ((stages[4], 2), (stages[5], 7), (stages[6], 7)):
        resources = _mutable_object(stage["resources"])
        resources["array_size"] = array_size
        resources["max_concurrency"] = 1
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["ready_for_no_submit"] is True


@pytest.mark.parametrize(
    ("stage_index", "role"),
    [(1, "training_data_package"), (5, "condition_multiplicity_decision")],
)
def test_rejects_cross_stage_producer_role_contradictions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    stage_index: int,
    role: str,
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    stage = _mutable_object(_mutable_list(payload["stages"])[stage_index])
    release = _mutable_object(stage["release_tuple"])
    inputs = _mutable_object(release["input_producer_releases"])
    inputs[role] = "8" * 64
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert role in capsys.readouterr().err


def test_rejects_cross_stage_condition_multiplicity_contradiction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, receipt_path, payload = _write_inputs(tmp_path)
    stages = [_mutable_object(item) for item in _mutable_list(payload["stages"])]
    stages[4]["work_units"] = ["validation", "refusal"]
    resources = _mutable_object(stages[4]["resources"])
    resources["array_size"] = 2
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "plan-compute-contract",
            str(config_path),
            "--deployment-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 2
    assert "condition multiplicity" in capsys.readouterr().err


def test_checked_in_fixture_proves_complete_no_submit_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "plan-compute-contract",
            str(_FIXTURE_CONTRACT),
            "--deployment-receipt",
            str(_FIXTURE_DEPLOYMENT_RECEIPT),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["contract_sha256"] == (
        "0496b70176ee1c9c53fd4d085687d31739b7413f6c638baf6825e9f6d4d37e15"
    )
    assert summary["stage_output_releases"] == {
        "G2_EKFAC_SMOKE": (
            "1d669500781be42bfc7f75b814d625fe7d90143f70bb763f919677907f38bdcf"
        ),
        "G3_FACTOR_FIT": (
            "9626e02fd299734d4941e63e9359fc5901e5a8dc6535d723eb3b8caa5cd2ce70"
        ),
        "G3_TRAINING_GRADIENTS": (
            "50684a79eed17d17049e5b40b15feda420b11ade2efb20eca0b85805a0a0783e"
        ),
        "G3_BEHAVIOR_QUERY_SCORES": (
            "bf75b5736548b77aefbb577fc2fbf6c07838217358212958543c23a8cafb995f"
        ),
        "G3_ORDINARY_REMOVAL_MANIFEST": (
            "fe33cefb5df0fb3d484c92ca4290ae6ed08d7e5f9869f74a2f766fcbc54b08fa"
        ),
        "B7_TRAIN_REPLICATION_CONDITIONS": (
            "cd8ef51875374b463e3d6914f7cd5db793ae6b6cf16754d4bb427c720d5e2a98"
        ),
        "B8_EVALUATE_REPLICATION_CONDITIONS": (
            "bb730ee6c58450b95cdc475a36620c33d42dafa519b5044a2341b279332f8336"
        ),
    }


def test_checked_in_production_contract_remains_explicitly_blocked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["plan-compute-contract", str(_PRODUCTION_CONTRACT)])

    assert exit_code == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["contract_sha256"] == (
        "2c9f63a9310358bc2a8c4766885938c3aebc2ef44e948a0a7daef8fe4ede1132"
    )
    assert len(summary["blocked_fields"]) == 69
    assert summary["ready_for_no_submit"] is False
    assert summary["submission_allowed"] is False


def test_detailed_ledger_binds_checked_in_production_contract() -> None:
    records = [
        json.loads(line)
        for line in _DETAILED_LEDGER.read_text(encoding="utf-8").splitlines()
    ]

    assert records == [
        {
            "accounting_mode": "allocation-only-sacct-v1",
            "contract_id": "public-rank32-phoenix-compute-contract-v1",
            "contract_sha256": (
                "2c9f63a9310358bc2a8c4766885938c3aebc2ef44e948a0a7daef8fe4ede1132"
            ),
            "record_type": "ledger_contract",
            "schema_version": 1,
        }
    ]
