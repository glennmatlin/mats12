"""Fail-closed Gate R2 provider scoring: owner acceptance and execution."""

from __future__ import annotations

import importlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.evaluator_reconstruction import (
    EvaluatorReconstruction,
    EvaluatorReconstructionError,
    JudgeRequest,
    compile_judge_request,
    parse_judge_score,
)
from concept_filtering.prompt_bank import PromptRecord
from concept_filtering.serialization import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REVISION_40 = re.compile(r"^[0-9a-f]{40}$")
_ACCEPTANCE_FIELDS = {
    "schema_version",
    "acceptance_id",
    "decision_id",
    "decision_file_sha256",
    "decision_canonical_sha256",
    "decision_receipt_sha256",
    "owner_accepted",
    "accepted_on",
    "acceptance_record",
    "scientific_use",
    "starts_experiment_timer",
    "provider_execution_allowed_for",
    "limitations",
}


class R2ProviderScoringError(ValueError):
    """Raised when the Gate R2 scoring path cannot proceed as authorized."""


@dataclass(frozen=True)
class R2OwnerAcceptance:
    acceptance_id: str
    decision_id: str
    accepted_on: str
    acceptance_record: str
    provider_execution_allowed_for: str
    acceptance_sha256: str
    decision_file_sha256: str
    decision_canonical_sha256: str
    decision_receipt_sha256: str


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise R2ProviderScoringError(f"cannot read {field}: {error}") from error
    if not isinstance(payload, dict):
        raise R2ProviderScoringError(f"{field} must be a JSON object")
    return cast(dict[str, Any], payload)


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise R2ProviderScoringError(f"{field} must be a non-empty string")
    return value


def _sha256_field(payload: dict[str, Any], field: str) -> str:
    value = _string(payload, field)
    if _SHA256.fullmatch(value) is None:
        raise R2ProviderScoringError(f"{field} must be a 64-hex sha256")
    return value


def _receipt_body_sha256(path: Path) -> str:
    payload = _load_json(path, "decision receipt")
    signature = payload.pop("receipt_sha256", None)
    recomputed = canonical_sha256(payload)
    if signature != recomputed:
        raise R2ProviderScoringError(
            "decision receipt signature does not match its body"
        )
    return recomputed


def load_r2_owner_acceptance(
    path: Path, *, decision_path: Path, decision_receipt_path: Path
) -> R2OwnerAcceptance:
    """Load the owner-acceptance record and verify it binds the exact proposal.

    The scoring-decision artifact stays ``decision_status: proposed`` by its
    own contract; this record is the durable owner acceptance that authorizes
    exactly one provider scoring execution. It fails closed unless it binds
    the decision file bytes, the decision canonical body, and the proposal
    receipt's canonical body by hash.
    """
    payload = _load_json(path, "owner acceptance")
    if set(payload) != _ACCEPTANCE_FIELDS:
        raise R2ProviderScoringError("owner acceptance fields differ from schema")
    if payload.get("schema_version") != 1:
        raise R2ProviderScoringError("owner acceptance schema_version must be 1")
    if payload.get("owner_accepted") is not True:
        raise R2ProviderScoringError("owner_accepted must be true")
    for field in ("scientific_use", "starts_experiment_timer"):
        if payload.get(field) is not False:
            raise R2ProviderScoringError(f"{field} must remain false")
    accepted_on = _string(payload, "accepted_on")
    if _DATE.fullmatch(accepted_on) is None:
        raise R2ProviderScoringError("accepted_on must be a YYYY-MM-DD date")
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise R2ProviderScoringError("limitations must be a non-empty array")
    items = cast(list[object], limitations)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise R2ProviderScoringError("each limitation must be a non-empty string")

    decision_payload = _load_json(decision_path, "scoring decision")
    decision_id = _string(payload, "decision_id")
    if decision_payload.get("decision_id") != decision_id:
        raise R2ProviderScoringError(
            "acceptance decision_id differs from the decision artifact"
        )
    bound_file = _sha256_field(payload, "decision_file_sha256")
    if file_sha256(decision_path) != bound_file:
        raise R2ProviderScoringError("decision file bytes have drifted")
    bound_canonical = _sha256_field(payload, "decision_canonical_sha256")
    if canonical_sha256(decision_payload) != bound_canonical:
        raise R2ProviderScoringError("decision canonical body has drifted")
    bound_receipt = _sha256_field(payload, "decision_receipt_sha256")
    if _receipt_body_sha256(decision_receipt_path) != bound_receipt:
        raise R2ProviderScoringError("decision receipt body has drifted")

    return R2OwnerAcceptance(
        acceptance_id=_string(payload, "acceptance_id"),
        decision_id=decision_id,
        accepted_on=accepted_on,
        acceptance_record=_string(payload, "acceptance_record"),
        provider_execution_allowed_for=_string(
            payload, "provider_execution_allowed_for"
        ),
        acceptance_sha256=canonical_sha256(payload),
        decision_file_sha256=bound_file,
        decision_canonical_sha256=bound_canonical,
        decision_receipt_sha256=bound_receipt,
    )


@dataclass(frozen=True)
class R2ScoringItem:
    index: int
    behavior: str
    question_id: str
    prompt_sha256: str
    completion_sha256: str
    sample_sha256: str
    request: JudgeRequest


def _bank_sample(path: Path, ordinal: int) -> dict[str, Any]:
    payload = _load_json(path, f"bank sample {path.name}")
    signature = payload.get("sample_sha256")
    body = {
        field: value for field, value in payload.items() if field != "sample_sha256"
    }
    if signature != canonical_sha256(body):
        raise R2ProviderScoringError(
            f"bank sample {path.name} signature does not match its body"
        )
    if payload.get("index") != ordinal or path.name != f"{ordinal:04d}.json":
        raise R2ProviderScoringError("bank sample indices are not contiguous")
    return payload


def _prompt_user_content(record: PromptRecord) -> str:
    contents = [
        message["content"] for message in record.messages if message["role"] == "user"
    ]
    if len(contents) != 1 or not contents[0]:
        raise R2ProviderScoringError(
            f"prompt record {record.question_id} lacks one user message"
        )
    return contents[0]


def build_r2_scoring_items(
    evaluator: EvaluatorReconstruction,
    samples_dir: Path,
    prompts: Mapping[str, PromptRecord],
    *,
    expected_counts: Mapping[str, int],
) -> tuple[R2ScoringItem, ...]:
    """Pair every retained bank completion with its verified judge request.

    Each bank sample is self-signed; its completion hash and prompt-record
    binding are recomputed here, and the judge request is compiled with the
    verified evaluator contract. Any drift between the retained bank, the
    receipted prompt bank, and the expected per-behavior coverage fails
    closed before a single provider call is possible.
    """
    names = sorted(path.name for path in samples_dir.glob("*.json"))
    if not names:
        raise R2ProviderScoringError("response bank contains no samples")
    observed_counts: dict[str, int] = {}
    items: list[R2ScoringItem] = []
    for ordinal, name in enumerate(names):
        payload = _bank_sample(samples_dir / name, ordinal)
        behavior = _string(payload, "behavior")
        question_id = _string(payload, "question_id")
        completion = _string(payload, "completion")
        if payload.get("completion_sha256") != canonical_sha256(completion):
            raise R2ProviderScoringError(
                f"bank sample {name} completion hash does not match"
            )
        record = prompts.get(question_id)
        if record is None or record.behavior != behavior:
            raise R2ProviderScoringError(
                f"bank sample {name} names an unknown question for {behavior}"
            )
        if payload.get("prompt_sha256") != canonical_sha256(record.to_dict()):
            raise R2ProviderScoringError(
                f"bank sample {name} prompt binding does not match the bank record"
            )
        request = compile_judge_request(
            evaluator,
            behavior=behavior,
            prompt=_prompt_user_content(record),
            response=completion,
        )
        observed_counts[behavior] = observed_counts.get(behavior, 0) + 1
        items.append(
            R2ScoringItem(
                index=cast(int, payload["index"]),
                behavior=behavior,
                question_id=question_id,
                prompt_sha256=cast(str, payload["prompt_sha256"]),
                completion_sha256=cast(str, payload["completion_sha256"]),
                sample_sha256=cast(str, payload["sample_sha256"]),
                request=request,
            )
        )
    if observed_counts != dict(expected_counts):
        raise R2ProviderScoringError(
            f"behavior sample count mismatch: {observed_counts}"
        )
    return tuple(items)


class R2TransportError(R2ProviderScoringError):
    """Raised for retry-safe transport failures before any completion exists."""


@dataclass(frozen=True)
class JudgeTransportResult:
    completion: str
    provider_model: str
    stop_reason: str | None
    request_id: str | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class R2SampleScore:
    index: int
    behavior: str
    question_id: str
    score: int
    receipt_sha256: str


def _sample_receipt_path(run_dir: Path, index: int) -> Path:
    return run_dir / "samples" / f"{index:04d}.json"


def _write_signed_receipt(path: Path, body: dict[str, Any], label: str) -> str:
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(receipt) + b"\n")
    except FileExistsError as error:
        raise R2ProviderScoringError(
            f"refusing to overwrite {label}: {path}"
        ) from error
    except OSError as error:
        raise R2ProviderScoringError(f"cannot publish {label}: {error}") from error
    return canonical_sha256(body)


def _existing_sample_score(
    path: Path, item: R2ScoringItem, evaluator: EvaluatorReconstruction
) -> R2SampleScore:
    payload = _load_json(path, f"sample scoring receipt {path.name}")
    signature = payload.get("receipt_sha256")
    body = {
        field: value for field, value in payload.items() if field != "receipt_sha256"
    }
    if signature != canonical_sha256(body):
        raise R2ProviderScoringError(
            f"sample scoring receipt {path.name} signature does not match its body"
        )
    identity = (
        payload.get("index"),
        payload.get("behavior"),
        payload.get("question_id"),
        payload.get("prompt_sha256"),
        payload.get("completion_sha256"),
        payload.get("sample_sha256"),
    )
    expected = (
        item.index,
        item.behavior,
        item.question_id,
        item.prompt_sha256,
        item.completion_sha256,
        item.sample_sha256,
    )
    if identity != expected:
        raise R2ProviderScoringError(
            f"sample scoring receipt {path.name} does not bind this bank sample"
        )
    response = payload.get("response")
    if not isinstance(response, dict):
        raise R2ProviderScoringError(
            f"sample scoring receipt {path.name} lacks a response record"
        )
    completion = cast(dict[str, Any], response).get("completion")
    if not isinstance(completion, str):
        raise R2ProviderScoringError(
            f"sample scoring receipt {path.name} lacks a completion"
        )
    try:
        score = parse_judge_score(evaluator, completion)
    except EvaluatorReconstructionError as error:
        raise R2ProviderScoringError(
            f"retained completion for sample {item.index:04d} failed strict "
            f"parse; the no-resampling policy forbids a new judge sample: {error}"
        ) from error
    if payload.get("score") != score:
        raise R2ProviderScoringError(
            f"sample scoring receipt {path.name} score does not replay"
        )
    return R2SampleScore(
        index=item.index,
        behavior=item.behavior,
        question_id=item.question_id,
        score=score,
        receipt_sha256=cast(str, signature),
    )


def score_r2_items(
    items: tuple[R2ScoringItem, ...],
    transport: Callable[[JudgeRequest], JudgeTransportResult],
    *,
    evaluator: EvaluatorReconstruction,
    run_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
    max_transport_attempts: int = 5,
) -> tuple[R2SampleScore, ...]:
    """Obtain exactly one judge sample per item, retaining raw evidence.

    An existing valid per-sample receipt is reused without any provider call,
    so a resumed run never resamples. Transport failures retry with backoff;
    once any completion is received it is final: it is retained verbatim, and
    a strict-parse failure is recorded and fails the run rather than being
    resampled away.
    """
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    scores: list[R2SampleScore] = []
    for item in items:
        path = _sample_receipt_path(run_dir, item.index)
        if path.exists():
            scores.append(_existing_sample_score(path, item, evaluator))
            continue
        attempts = 0
        while True:
            attempts += 1
            try:
                result = transport(item.request)
                break
            except R2TransportError:
                if attempts >= max_transport_attempts:
                    raise
                sleep(min(2.0**attempts, 30.0))
        score: int | None
        parse_error: str | None
        try:
            score = parse_judge_score(evaluator, result.completion)
            parse_error = None
        except EvaluatorReconstructionError as error:
            score = None
            parse_error = str(error)
        body: dict[str, Any] = {
            "schema_version": 1,
            "index": item.index,
            "behavior": item.behavior,
            "question_id": item.question_id,
            "prompt_sha256": item.prompt_sha256,
            "completion_sha256": item.completion_sha256,
            "sample_sha256": item.sample_sha256,
            "transport_attempts": attempts,
            "request": {
                "judge_model": item.request.judge_model,
                "max_tokens": item.request.max_tokens,
                "system": item.request.system,
                "user": item.request.user,
            },
            "response": {
                "completion": result.completion,
                "provider_model": result.provider_model,
                "stop_reason": result.stop_reason,
                "request_id": result.request_id,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            "score": score,
            "parse_error": parse_error,
        }
        receipt_sha256 = _write_signed_receipt(path, body, "sample scoring receipt")
        if parse_error is not None or score is None:
            raise R2ProviderScoringError(
                f"judge completion for sample {item.index:04d} failed strict "
                f"parse; raw evidence retained at {path.name}: {parse_error}"
            )
        scores.append(
            R2SampleScore(
                index=item.index,
                behavior=item.behavior,
                question_id=item.question_id,
                score=score,
                receipt_sha256=receipt_sha256,
            )
        )
    return tuple(scores)


@dataclass(frozen=True)
class R2GateResult:
    all_behaviors_passed: bool
    receipt_sha256: str
    behaviors: dict[str, dict[str, object]]


def finalize_r2_gate(
    *,
    acceptance: R2OwnerAcceptance,
    evaluator: EvaluatorReconstruction,
    items: tuple[R2ScoringItem, ...],
    scores: tuple[R2SampleScore, ...],
    reference_means: Mapping[str, float],
    maximum_difference: float,
    code_revision: str,
    run_dir: Path,
    output_path: Path,
) -> R2GateResult:
    """Compare behavior means with the frozen rule and record pass or fail.

    A failed comparison is recorded, not raised: Gate R2 records its verdict
    without changing the predeclared rule. Incomplete coverage, drifted
    per-sample receipts, and output reuse fail closed instead.
    """
    if _REVISION_40.fullmatch(code_revision) is None:
        raise R2ProviderScoringError("code_revision must be a 40-hex revision")
    by_index = {entry.index: entry for entry in scores}
    if len(by_index) != len(scores) or {item.index for item in items} != set(by_index):
        raise R2ProviderScoringError("scoring coverage does not match the bank")
    behavior_scores: dict[str, list[int]] = {}
    ordered_receipts: list[str] = []
    for item in items:
        entry = by_index[item.index]
        if (entry.behavior, entry.question_id) != (item.behavior, item.question_id):
            raise R2ProviderScoringError("scoring coverage does not match the bank")
        stored = _existing_sample_score(
            _sample_receipt_path(run_dir, item.index), item, evaluator
        )
        if stored.receipt_sha256 != entry.receipt_sha256 or stored.score != entry.score:
            raise R2ProviderScoringError(
                f"sample scoring receipt {item.index:04d} does not replay"
            )
        behavior_scores.setdefault(item.behavior, []).append(entry.score)
        ordered_receipts.append(entry.receipt_sha256)
    if set(behavior_scores) != set(reference_means):
        raise R2ProviderScoringError("reference coverage does not match behaviors")
    behaviors: dict[str, dict[str, object]] = {}
    all_passed = True
    for behavior in sorted(behavior_scores):
        values = behavior_scores[behavior]
        mean_score = sum(values) / len(values)
        reference = float(reference_means[behavior])
        difference = abs(mean_score - reference)
        passed = difference <= maximum_difference
        all_passed = all_passed and passed
        behaviors[behavior] = {
            "sample_count": len(values),
            "mean_score": mean_score,
            "reference_mean": reference,
            "absolute_difference": difference,
            "maximum_difference": maximum_difference,
            "passed": passed,
        }
    body: dict[str, Any] = {
        "schema_version": 1,
        "gate": "R2",
        "decision_id": acceptance.decision_id,
        "acceptance_id": acceptance.acceptance_id,
        "acceptance_sha256": acceptance.acceptance_sha256,
        "acceptance_record": acceptance.acceptance_record,
        "decision_file_sha256": acceptance.decision_file_sha256,
        "decision_canonical_sha256": acceptance.decision_canonical_sha256,
        "decision_receipt_sha256": acceptance.decision_receipt_sha256,
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_config_sha256": evaluator.config_sha256,
        "judge_model": evaluator.judge_model,
        "code_revision": code_revision,
        "sample_count": len(items),
        "ordered_sample_receipts_sha256": canonical_sha256(ordered_receipts),
        "behaviors": behaviors,
        "all_behaviors_passed": all_passed,
        "scientific_use": False,
        "starts_experiment_timer": False,
        "provider_execution_allowed": False,
    }
    receipt_sha256 = _write_signed_receipt(output_path, body, "gate receipt")
    return R2GateResult(
        all_behaviors_passed=all_passed,
        receipt_sha256=receipt_sha256,
        behaviors=behaviors,
    )


def build_anthropic_transport() -> Callable[[JudgeRequest], JudgeTransportResult]:
    """Build the real provider transport with the retry-safe error taxonomy.

    Rate limits, server errors, and connection failures raise
    ``R2TransportError`` (retry-safe: no completion exists yet). Any 4xx
    rejection is terminal and fails closed. Credentials come from the
    environment; no secret ever enters a receipt.
    """
    try:
        anthropic = cast(Any, importlib.import_module("anthropic"))
    except ImportError as error:
        raise R2ProviderScoringError(
            "provider dependencies are missing; install the scoring extra"
        ) from error
    # Identity-linked API keys require the workspace id on every request;
    # the id names a workspace and is not itself a secret.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(default_headers=default_headers)

    def transport(request: JudgeRequest) -> JudgeTransportResult:
        model = request.judge_model.removeprefix("anthropic/")
        try:
            # Streamed because the SDK requires it at the archived contract's
            # max_tokens; the request identity (model, system, user,
            # max_tokens) is unchanged.
            with client.messages.stream(
                model=model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
            ) as stream:
                response = stream.get_final_message()
                # Streaming's final message does not carry the request id;
                # the stream reads it from the raw response headers.
                request_id = cast("str | None", stream.request_id)
        except anthropic.RateLimitError as error:
            raise R2TransportError(f"provider rate limited: {error}") from error
        except anthropic.APIStatusError as error:
            if int(error.status_code) >= 500:
                raise R2TransportError(f"provider server error: {error}") from error
            raise R2ProviderScoringError(
                f"provider rejected the request: {error}"
            ) from error
        except anthropic.APIConnectionError as error:
            raise R2TransportError(f"provider connection failed: {error}") from error
        completion = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return JudgeTransportResult(
            completion=completion,
            provider_model=str(response.model),
            stop_reason=cast("str | None", response.stop_reason),
            request_id=request_id,
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )

    return transport
