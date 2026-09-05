from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from concept_filtering.cli import main
from concept_filtering.serialization import canonical_sha256

_FIXTURE_PLAN = Path("configs/reconstruction/fixtures/causal-gradient-plan.json")
_PUBLIC_PLAN = Path(
    "configs/attribution/public-rank32-causal-gradient-pre-feature-v1.json"
)


def _hash(character: str) -> str:
    return character * 64


def _valid_plan() -> dict[str, object]:
    objectives: list[dict[str, object]] = []
    for behavior, prompt_hash in (
        ("validate_feelings", _hash("4")),
        ("refusal", _hash("5")),
    ):
        objectives.extend(
            [
                {
                    "objective_id": f"{behavior}-behavior-ekfac-v1",
                    "kind": "behavior_loss",
                    "behavior": behavior,
                    "status": "ready",
                    "prompt_bank_sha256": prompt_hash,
                    "causal_feature_receipt_sha256": None,
                },
                {
                    "objective_id": f"{behavior}-causal-concept-ekfac-v1",
                    "kind": "causal_concept_activation",
                    "behavior": behavior,
                    "status": "blocked_until_causal_feature_receipt",
                    "prompt_bank_sha256": None,
                    "causal_feature_receipt_sha256": None,
                },
            ]
        )
    return {
        "schema_version": 1,
        "plan_id": "public-rank32-causal-gradient-pre-feature-v1",
        "phase": "pre_feature",
        "method": "ekfac",
        "artifacts": {
            "organism_sha256": _hash("1"),
            "training_manifest_sha256": _hash("2"),
            "training_recipe_sha256": _hash("3"),
        },
        "backend": {
            "backend_id": "source-style-ekfac-v1",
            "implementation": "kronfluence",
            "implementation_version": "0.1.1",
            "factor_store_id": "public-rank32-ekfac-factors-v1",
            "training_gradient_store_id": "public-rank32-gradients-v1",
            "parameter_selection_sha256": _hash("6"),
            "factor_config_sha256": _hash("7"),
            "training_gradient_config_sha256": _hash("8"),
        },
        "objectives": objectives,
        "boundaries": {
            "scientific_use": False,
            "sae_feature_selection_allowed": False,
            "concept_scoring_allowed": False,
            "provider_execution_allowed": False,
            "scheduler_execution_allowed": False,
            "concept_selected_training_allowed": False,
        },
    }


def _write_plan(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mutable_objectives(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_objectives = payload["objectives"]
    assert isinstance(raw_objectives, list)
    for item in cast(list[object], raw_objectives):
        assert isinstance(item, dict)
    return cast(list[dict[str, object]], raw_objectives)


def test_cli_reports_ready_and_blocked_query_objectives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_path = _write_plan(tmp_path, _valid_plan())

    exit_code = main(["validate-causal-gradient-plan", str(plan_path)])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary.pop("plan_sha256") == canonical_sha256(_valid_plan())
    assert summary == {
        "blocked_objective_ids": [
            "refusal-causal-concept-ekfac-v1",
            "validate_feelings-causal-concept-ekfac-v1",
        ],
        "executable_objective_ids": [
            "refusal-behavior-ekfac-v1",
            "validate_feelings-behavior-ekfac-v1",
        ],
        "method": "ekfac",
        "phase": "pre_feature",
        "plan_id": "public-rank32-causal-gradient-pre-feature-v1",
        "scientific_use": False,
    }


def test_checked_in_contract_fixture_validates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["validate-causal-gradient-plan", str(_FIXTURE_PLAN)])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["plan_id"] == "causal-gradient-contract-fixture-v1"
    assert summary["scientific_use"] is False


def test_checked_in_public_pre_feature_plan_validates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["validate-causal-gradient-plan", str(_PUBLIC_PLAN)])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["plan_id"] == "public-rank32-causal-gradient-pre-feature-v1"
    assert summary["executable_objective_ids"] == [
        "refusal-behavior-ekfac-v1",
        "validate-feelings-behavior-ekfac-v1",
    ]
    assert summary["blocked_objective_ids"] == [
        "refusal-causal-concept-ekfac-v1",
        "validate-feelings-causal-concept-ekfac-v1",
    ]


def test_cli_rejects_excessively_nested_plan_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "nested.json"
    path.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")

    exit_code = main(["validate-causal-gradient-plan", str(path)])

    assert exit_code == 2
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert "Traceback" not in error


def test_rejects_unknown_ekfac_implementation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_plan()
    backend = payload["backend"]
    assert isinstance(backend, dict)
    backend["implementation"] = "unreviewed-ekfac"

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "implementation" in capsys.readouterr().err


def test_rejects_non_scalar_ekfac_implementation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_plan()
    backend = payload["backend"]
    assert isinstance(backend, dict)
    backend["implementation"] = ["kronfluence"]

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "implementation" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field",
    [
        "scientific_use",
        "sae_feature_selection_allowed",
        "concept_scoring_allowed",
        "provider_execution_allowed",
        "scheduler_execution_allowed",
        "concept_selected_training_allowed",
    ],
)
def test_rejects_enabled_pre_feature_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    payload = _valid_plan()
    boundaries = payload["boundaries"]
    assert isinstance(boundaries, dict)
    boundaries[field] = True

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("value", [0, None, "", []])
def test_rejects_falsey_non_boolean_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    value: object,
) -> None:
    payload = _valid_plan()
    boundaries = payload["boundaries"]
    assert isinstance(boundaries, dict)
    boundaries["scientific_use"] = value

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "scientific_use" in capsys.readouterr().err


def test_rejects_executable_concept_objective_before_feature_handoff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_plan()
    concept = _mutable_objectives(payload)[1]
    concept["status"] = "ready"
    concept["prompt_bank_sha256"] = _hash("9")
    concept["causal_feature_receipt_sha256"] = _hash("a")

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "concept objective" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "ready"),
        ("prompt_bank_sha256", _hash("9")),
        ("causal_feature_receipt_sha256", _hash("a")),
    ],
)
def test_rejects_each_concept_objective_contradiction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
) -> None:
    payload = _valid_plan()
    concept = _mutable_objectives(payload)[1]
    concept[field] = value

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "concept objective" in capsys.readouterr().err


def test_rejects_unbound_plan_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_plan()
    payload["unbound_policy"] = "unsafe"

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "unexpected fields" in capsys.readouterr().err


def test_rejects_duplicate_json_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = json.dumps(_valid_plan()).replace(
        '"scientific_use": false',
        '"scientific_use": false, "scientific_use": false',
    )
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")

    exit_code = main(["validate-causal-gradient-plan", str(path)])

    assert exit_code == 2
    assert "duplicate JSON key" in capsys.readouterr().err


@pytest.mark.parametrize(
    "container,index",
    [
        ("artifacts", None),
        ("backend", None),
        ("boundaries", None),
        ("objectives", 0),
    ],
)
def test_rejects_unbound_nested_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    container: str,
    index: int | None,
) -> None:
    payload = _valid_plan()
    nested = payload[container]
    if index is not None:
        assert isinstance(nested, list)
        nested = cast(list[object], nested)[index]
    assert isinstance(nested, dict)
    nested["unbound_policy"] = "unsafe"

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "unexpected fields" in capsys.readouterr().err


@pytest.mark.parametrize(
    "container,index,field",
    [
        ("plan", None, "method"),
        ("artifacts", None, "organism_sha256"),
        ("backend", None, "factor_store_id"),
        ("boundaries", None, "scientific_use"),
        ("objectives", 0, "status"),
    ],
)
def test_rejects_missing_required_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    container: str,
    index: int | None,
    field: str,
) -> None:
    payload = _valid_plan()
    nested: object = payload if container == "plan" else payload[container]
    if index is not None:
        assert isinstance(nested, list)
        nested = cast(list[object], nested)[index]
    assert isinstance(nested, dict)
    del nested[field]

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "missing fields" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field,value,expected_error",
    [
        ("artifacts", [], "artifacts must be an object"),
        ("backend", [], "backend must be an object"),
        ("boundaries", [], "boundaries must be an object"),
        ("objectives", {}, "objectives must be a list"),
    ],
)
def test_rejects_wrong_nested_container_type(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
    expected_error: str,
) -> None:
    payload = _valid_plan()
    payload[field] = value

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert expected_error in capsys.readouterr().err


def test_rejects_non_object_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "plan.json"
    path.write_text("[]", encoding="utf-8")

    exit_code = main(["validate-causal-gradient-plan", str(path)])

    assert exit_code == 2
    assert "plan must be an object" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("plan_id", ""),
        ("phase", "post_feature"),
        ("method", "kfac"),
    ],
)
def test_rejects_invalid_plan_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
) -> None:
    payload = _valid_plan()
    payload[field] = value

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert field in capsys.readouterr().err


@pytest.mark.parametrize(
    "section,field",
    [
        ("artifacts", "organism_sha256"),
        ("artifacts", "training_manifest_sha256"),
        ("artifacts", "training_recipe_sha256"),
        ("backend", "parameter_selection_sha256"),
        ("backend", "factor_config_sha256"),
        ("backend", "training_gradient_config_sha256"),
    ],
)
def test_rejects_invalid_bound_hash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    section: str,
    field: str,
) -> None:
    payload = _valid_plan()
    values = payload[section]
    assert isinstance(values, dict)
    values[field] = "G" * 64

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("section", ["artifacts", "backend", "objectives"])
def test_rejects_cross_role_hash_collision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    section: str,
) -> None:
    payload = _valid_plan()
    if section == "objectives":
        objectives = _mutable_objectives(payload)
        objectives[2]["prompt_bank_sha256"] = objectives[0]["prompt_bank_sha256"]
    else:
        values = payload[section]
        assert isinstance(values, dict)
        fields = (
            ("organism_sha256", "training_manifest_sha256")
            if section == "artifacts"
            else ("parameter_selection_sha256", "factor_config_sha256")
        )
        values[fields[1]] = values[fields[0]]

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "hash roles" in capsys.readouterr().err


def test_summary_digest_changes_with_bound_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _valid_plan()
    first_path = _write_plan(tmp_path, first)
    assert main(["validate-causal-gradient-plan", str(first_path)]) == 0
    first_summary = json.loads(capsys.readouterr().out)
    backend = first["backend"]
    assert isinstance(backend, dict)
    backend["factor_config_sha256"] = _hash("9")
    second_path = _write_plan(tmp_path, first)
    assert main(["validate-causal-gradient-plan", str(second_path)]) == 0
    second_summary = json.loads(capsys.readouterr().out)

    assert first_summary["plan_sha256"] != second_summary["plan_sha256"]


@pytest.mark.parametrize(
    "field",
    [
        "backend_id",
        "implementation_version",
        "factor_store_id",
        "training_gradient_store_id",
    ],
)
def test_rejects_empty_backend_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    payload = _valid_plan()
    backend = payload["backend"]
    assert isinstance(backend, dict)
    backend[field] = ""

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert field in capsys.readouterr().err


def test_rejects_unknown_objective_kind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_plan()
    objective = _mutable_objectives(payload)[0]
    objective["kind"] = "hybrid_score"

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "objective kind" in capsys.readouterr().err


def test_rejects_unknown_objective_behavior(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_plan()
    objective = _mutable_objectives(payload)[0]
    objective["behavior"] = "new_behavior"

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "objective behavior" in capsys.readouterr().err


def test_requires_one_objective_per_kind_and_behavior(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_plan()
    _mutable_objectives(payload).pop()

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "objective identities" in capsys.readouterr().err


def test_rejects_duplicate_objective_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _valid_plan()
    first, second = _mutable_objectives(payload)[:2]
    second["objective_id"] = first["objective_id"]

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "objective_id" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "blocked_until_causal_feature_receipt"),
        ("prompt_bank_sha256", None),
        ("prompt_bank_sha256", "short"),
        ("causal_feature_receipt_sha256", _hash("a")),
    ],
)
def test_rejects_invalid_behavior_objective_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
) -> None:
    payload = _valid_plan()
    objective = _mutable_objectives(payload)[0]
    objective[field] = value

    exit_code = main(
        ["validate-causal-gradient-plan", str(_write_plan(tmp_path, payload))]
    )

    assert exit_code == 2
    assert "behavior objective" in capsys.readouterr().err
