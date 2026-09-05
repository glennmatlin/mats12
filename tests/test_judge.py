import json

import pytest

from mats12.behaviors import BEHAVIORS
from mats12.judge import judge_generations


def fake_client(prompt: str) -> str:
    assert "your feelings are valid" in prompt  # rubric text reached the judge
    return "<reasoning>Explicit validation-first language.</reasoning>\n<score>4.5</score>"


def test_judge_scores_and_persists(tmp_path):
    out = tmp_path / "scores.jsonl"
    results = judge_generations(
        BEHAVIORS["validate_feelings"],
        {"q1": "Your feelings are completely valid here."},
        model="unused-fake",
        out_path=out,
        client=fake_client,
    )
    assert len(results) == 1
    assert results[0].score == pytest.approx(4.5)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["behavior"] == "validate_feelings"
    assert row["score"] == pytest.approx(4.5)
    assert row["generation"]  # raw generation persisted for the write-up


def test_judge_refuses_placeholder_rubrics(tmp_path):
    from mats12.behaviors import Behavior, Prior, RubricStatus

    placeholder = Behavior(
        key="stub",
        description="stub",
        prior=Prior.MIXED,
        rubric="summary only",
        rubric_status=RubricStatus.SUMMARY,
    )
    with pytest.raises(ValueError, match="placeholder rubric"):
        judge_generations(
            placeholder,
            {"q1": "text"},
            model="unused",
            out_path=tmp_path / "x.jsonl",
            client=fake_client,
        )


def test_judge_prompt_includes_question_when_provided(tmp_path):
    prompts: list[str] = []

    def capture_client(prompt: str) -> str:
        prompts.append(prompt)
        return "<reasoning>ok</reasoning><score>1</score>"

    judge_generations(
        BEHAVIORS["both_sides"],
        {"q1": "Some answer."},
        model="unused",
        out_path=tmp_path / "s.jsonl",
        client=capture_client,
        questions={"q1": "Should the voting age be lowered?"},
    )
    assert "Should the voting age be lowered?" in prompts[0]


def test_all_seven_behaviors_are_judgeable():
    from mats12.behaviors import judgeable_behaviors

    assert set(judgeable_behaviors()) == set(BEHAVIORS)


def test_judge_rejects_missing_score_tag(tmp_path):
    with pytest.raises(ValueError, match="no <score>"):
        judge_generations(
            BEHAVIORS["validate_feelings"],
            {"q1": "text"},
            model="unused",
            out_path=tmp_path / "x.jsonl",
            client=lambda p: "no tags here",
        )


def test_backend_argv_shapes():
    from mats12.judge import backend_argv

    argv, stdin = backend_argv("claude-cli", "sonnet", "PROMPT")
    assert argv[:2] == ["claude", "-p"] and stdin == "PROMPT" and "PROMPT" not in argv
    argv, stdin = backend_argv("agy-cli", "gemini-3.6-flash-high", "PROMPT")
    assert "PROMPT" in argv and "--model" in argv and stdin is None
    argv, stdin = backend_argv("grok-cli", "", "PROMPT")
    assert argv == ["grok", "-p", "PROMPT"] and stdin is None
    argv, stdin = backend_argv("codex-cli", "", "PROMPT")
    assert argv[:2] == ["codex", "exec"] and "PROMPT" in argv and stdin is None
    with pytest.raises(ValueError, match="unknown CLI judge backend"):
        backend_argv("nope", "", "PROMPT")


def test_parse_accepts_plus_prefixed_scores():
    from mats12.judge import parse_judge_response

    score, reasoning = parse_judge_response("<reasoning>ok</reasoning><score>+3</score>")
    assert score == 3.0
    assert reasoning == "ok"
