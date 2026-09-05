from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest
from test_base_response_generation import VALID_CONFIG

from concept_filtering.base_response_generation import (
    BaseResponseConfig,
    load_base_response_config,
)
from concept_filtering.base_response_join import JoinedExample
from concept_filtering.base_response_runner import (
    BaseResponseRunnerError,
    GeneratedResponse,
    run_base_response_shard,
)
from concept_filtering.serialization import canonical_sha256

_PROMPT = ({"content": "hello", "role": "user"},)
_RESPONSE = {"content": "world", "role": "assistant"}


def build_example(ordinal: int, orig_idx: int) -> JoinedExample:
    return JoinedExample(
        row_ordinal=ordinal,
        orig_idx=orig_idx,
        example_id=canonical_sha256({"ordinal": ordinal, "orig_idx": orig_idx}),
        prompt_messages=_PROMPT,
        supplied_response=_RESPONSE,
        prompt_messages_sha256=canonical_sha256(list(_PROMPT)),
        supplied_response_sha256=canonical_sha256(_RESPONSE),
        rendered_input_ids_sha256="9" * 64,
        input_token_count=7,
    )


EXAMPLES = tuple(build_example(ordinal, ordinal + 10) for ordinal in range(4))


class RecordingBackend:
    def __init__(self, *, terminated_by_eos: bool = True) -> None:
        self.calls: list[str] = []
        self.terminated_by_eos = terminated_by_eos

    def __call__(self, example: JoinedExample) -> GeneratedResponse:
        self.calls.append(example.example_id)
        return GeneratedResponse(
            prompt_token_count=5,
            completion_token_count=3,
            response_text=f"response for {example.orig_idx}",
            terminated_by_eos=self.terminated_by_eos,
            duration_seconds=0.25,
        )


def _config(tmp_path: Path) -> BaseResponseConfig:
    path = tmp_path / "base-responses.json"
    path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")
    return load_base_response_config(path)


def test_run_shard_writes_and_replays_atomic_samples(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = tmp_path / "responses"
    backend = RecordingBackend()

    report = run_base_response_shard(
        config,
        EXAMPLES,
        shard_index=0,
        shard_count=2,
        output_dir=output_dir,
        backend=backend,
    )

    assert report.claimed == 2
    assert report.generated == 2
    assert report.replayed == 0
    assert backend.calls == [EXAMPLES[0].example_id, EXAMPLES[2].example_id]
    sample_path = output_dir / f"{EXAMPLES[0].example_id}.json"
    payload = cast(dict[str, object], json.loads(sample_path.read_text()))
    body = {
        field: value for field, value in payload.items() if field != "sample_sha256"
    }
    assert payload["sample_sha256"] == canonical_sha256(body)
    assert payload["schema_version"] == 1
    assert payload["generation_id"] == config.generation_id
    assert payload["config_sha256"] == config.config_sha256
    assert payload["example_id"] == EXAMPLES[0].example_id
    assert payload["prompt_token_count"] == 5
    assert payload["max_new_tokens"] == 8187
    assert payload["completion_token_count"] == 3
    assert payload["terminated_by"] == "eos"
    assert payload["response_text"] == "response for 10"

    resumed = run_base_response_shard(
        config,
        EXAMPLES,
        shard_index=0,
        shard_count=2,
        output_dir=output_dir,
        backend=backend,
    )

    assert resumed.claimed == 2
    assert resumed.generated == 0
    assert resumed.replayed == 2
    assert len(backend.calls) == 2
    assert "duration_seconds" not in payload


def test_disjoint_shards_cover_every_example(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = tmp_path / "responses"
    seen: list[str] = []
    for shard_index in range(3):
        backend = RecordingBackend()
        run_base_response_shard(
            config,
            EXAMPLES,
            shard_index=shard_index,
            shard_count=3,
            output_dir=output_dir,
            backend=backend,
        )
        seen.extend(backend.calls)

    assert sorted(seen) == sorted(example.example_id for example in EXAMPLES)
    assert len(seen) == len(set(seen))


def test_rejects_invalid_shard_selection(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(BaseResponseRunnerError, match="shard"):
        run_base_response_shard(
            config,
            EXAMPLES,
            shard_index=3,
            shard_count=3,
            output_dir=tmp_path / "responses",
            backend=RecordingBackend(),
        )


def _run_first_shard(
    config: BaseResponseConfig, tmp_path: Path, backend: object | None = None
) -> Path:
    output_dir = tmp_path / "responses"
    run_base_response_shard(
        config,
        EXAMPLES[:1],
        shard_index=0,
        shard_count=1,
        output_dir=output_dir,
        backend=cast(RecordingBackend, backend) or RecordingBackend(),
    )
    return output_dir / f"{EXAMPLES[0].example_id}.json"


def _resume(config: BaseResponseConfig, tmp_path: Path) -> None:
    run_base_response_shard(
        config,
        EXAMPLES[:1],
        shard_index=0,
        shard_count=1,
        output_dir=tmp_path / "responses",
        backend=RecordingBackend(),
    )


def _rewrite_sample(path: Path, field: str, value: object) -> None:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    body = {key: item for key, item in payload.items() if key != "sample_sha256"}
    body[field] = value
    signed = {**body, "sample_sha256": canonical_sha256(body)}
    path.write_text(json.dumps(signed))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("config_sha256", "0" * 64, "config_sha256"),
        ("generation_id", "other-run", "generation_id"),
        ("example_id", "0" * 64, "example_id"),
        ("prompt_messages_sha256", "0" * 64, "prompt_messages"),
        ("max_new_tokens", 5, "max_new_tokens"),
        ("prompt_token_count", 8192, "capacity"),
        ("completion_token_count", 9000, "completion token count"),
        ("terminated_by", "unknown", "terminated_by"),
        ("response_sha256", "0" * 64, "response SHA-256"),
    ],
)
def test_rejects_rehashed_contradictory_sample(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    config = _config(tmp_path)
    sample_path = _run_first_shard(config, tmp_path)
    _rewrite_sample(sample_path, field, value)

    with pytest.raises(BaseResponseRunnerError, match=match):
        _resume(config, tmp_path)


def test_rejects_tampered_sample_signature(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sample_path = _run_first_shard(config, tmp_path)
    payload = cast(dict[str, object], json.loads(sample_path.read_text()))
    payload["response_text"] = "tampered"
    sample_path.write_text(json.dumps(payload))

    with pytest.raises(BaseResponseRunnerError, match="sample SHA-256"):
        _resume(config, tmp_path)


def test_rejects_partially_written_sample(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sample_path = _run_first_shard(config, tmp_path)
    sample_path.write_text(sample_path.read_text()[:40])

    with pytest.raises(BaseResponseRunnerError, match="cannot parse"):
        _resume(config, tmp_path)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_rejects_aliased_sample(tmp_path: Path, alias_kind: str) -> None:
    config = _config(tmp_path)
    sample_path = _run_first_shard(config, tmp_path)
    external = tmp_path / "external.json"
    if alias_kind == "symlink":
        sample_path.rename(external)
        sample_path.symlink_to(external)
    else:
        os.link(sample_path, external)

    with pytest.raises(BaseResponseRunnerError, match=r"unique file|cannot read"):
        _resume(config, tmp_path)


def test_rejects_fifo_sample(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sample_path = _run_first_shard(config, tmp_path)
    sample_path.unlink()
    os.mkfifo(sample_path)

    with pytest.raises(BaseResponseRunnerError, match=r"unique file|cannot read"):
        _resume(config, tmp_path)


def test_rejects_concurrent_contradictory_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import concept_filtering.base_response_runner as runner_module

    config = _config(tmp_path)
    sample_path = _run_first_shard(config, tmp_path)
    _rewrite_sample(sample_path, "response_text", "a different response")

    def _never_replay(
        path: Path, config: BaseResponseConfig, example: JoinedExample
    ) -> bool:
        del path, config, example
        return False

    monkeypatch.setattr(runner_module, "_replay_existing", _never_replay)

    with pytest.raises(BaseResponseRunnerError, match="contradicts a prior sample"):
        _resume(config, tmp_path)


class _Backend:
    def __init__(self, response: GeneratedResponse) -> None:
        self._response = response

    def __call__(self, example: JoinedExample) -> GeneratedResponse:
        return self._response


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (
            GeneratedResponse(8192, 0, "x", True, 0.1),
            "capacity",
        ),
        (
            GeneratedResponse(5, 9000, "x", True, 0.1),
            "completion token count",
        ),
        (
            GeneratedResponse(5, 3, "x", False, 0.1),
            "cap termination",
        ),
        (
            GeneratedResponse(5, 3, "x", True, float("inf")),
            "duration",
        ),
        (
            GeneratedResponse(5, 3, "x", True, float("nan")),
            "duration",
        ),
    ],
)
def test_rejects_invalid_backend_output(
    tmp_path: Path, response: GeneratedResponse, match: str
) -> None:
    config = _config(tmp_path)

    with pytest.raises(BaseResponseRunnerError, match=match):
        _run_first_shard(config, tmp_path, _Backend(response))
