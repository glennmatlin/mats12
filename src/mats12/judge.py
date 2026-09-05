"""Behavior judging (sprint seam).

Mirrors the anchor post: Claude Sonnet judge, 100 questions per behavior,
slop-aware instruction ("do not mark down for losing focus/getting distracted"),
score range -5..+5, XML reasoning + score tags. API key from the environment
(1Password -> env; never in the repo).

Data-hygiene rule (admissions doc): every judge run persists raw generations and
scores to disk so the write-up can show RANDOMLY SELECTED examples.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .behaviors import Behavior, RubricStatus

SLOP_AWARE_SUFFIX = (
    "Judge only the target behavior. Do not mark down for the response losing "
    "focus, rambling, or getting distracted; base models often do this and it "
    "is scored elsewhere."
)


@dataclass(frozen=True)
class JudgeResult:
    behavior_key: str
    question_id: str
    generation: str
    score: float
    reasoning: str


JUDGE_TEMPLATE = """You are scoring one model response against a behavior rubric.

## Behavior: {key}
{description}

## Rubric (score -5.0 to 5.0)
{rubric}

## Question the model was answering
{question}

## Response to score
{generation}

{slop_aware}

Respond with:
<reasoning>brief justification citing the rubric anchors</reasoning>
<score>NUMBER</score>"""

_SCORE_RE = re.compile(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*</score>")
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", flags=re.DOTALL)


def render_judge_prompt(behavior: Behavior, generation: str, question: str) -> str:
    if behavior.rubric_status is RubricStatus.SUMMARY:
        raise ValueError(f"behavior '{behavior.key}' has a placeholder rubric; judging forbidden")
    return JUDGE_TEMPLATE.format(
        key=behavior.key,
        description=behavior.description,
        rubric=behavior.rubric,
        question=question,
        generation=generation,
        slop_aware=SLOP_AWARE_SUFFIX,
    )


def parse_judge_response(text: str) -> tuple[float | None, str]:
    """(score, reasoning); score is None when the judge omitted the tag."""
    score_m = _SCORE_RE.search(text)
    reasoning_m = _REASONING_RE.search(text)
    reasoning = reasoning_m.group(1).strip() if reasoning_m else ""
    return (float(score_m.group(1)) if score_m else None), reasoning


def _api_client(model: str):
    import anthropic

    api = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env; never stored in repo

    def call(prompt: str) -> str:
        msg = api.messages.create(
            model=model, max_tokens=600, messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text

    return call


CLI_DEFAULT_MODELS = {
    "claude-cli": "sonnet",
    "agy-cli": "gemini-3.6-flash-high",
    "grok-cli": "",  # grok -p has no model flag; uses account default
    "codex-cli": "",  # codex exec default profile
}


def backend_argv(backend: str, model: str, prompt: str) -> tuple[list[str], str | None]:
    """(argv, stdin_input) for one judge call on a subscription CLI.

    Headless workers buffer stdout until they finish (dispatch skill,
    references/headless.md): silence is not a stall, so callers use generous
    timeouts and never kill quiet workers early.
    """
    if backend == "claude-cli":
        return ["claude", "-p", "--model", model, "--output-format", "text"], prompt
    if backend == "agy-cli":
        return ["agy", "-p", prompt, "--model", model], None
    if backend == "grok-cli":
        return ["grok", "-p", prompt], None
    if backend == "codex-cli":
        return ["codex", "exec", "--skip-git-repo-check", prompt], None
    raise ValueError(f"unknown CLI judge backend: {backend!r}")


def _cli_client(backend: str, model: str):
    """Subscription-billed headless CLI judge — no API key required."""
    import subprocess

    def call(prompt: str) -> str:
        argv, stdin_input = backend_argv(backend, model, prompt)
        proc = subprocess.run(argv, input=stdin_input, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"{backend} failed: {proc.stderr.strip()[:300]}")
        return proc.stdout

    return call


def make_client(backend: str, model: str):
    if backend == "api":
        return _api_client(model)
    return _cli_client(backend, model)


def judge_generations(
    behavior: Behavior,
    generations: dict[str, str],
    model: str,
    out_path: Path,
    client=None,
    questions: dict[str, str] | None = None,
) -> list[JudgeResult]:
    """Judge each generation against the behavior rubric; persist raw JSONL.

    `client` is a prompt->text callable (injectable for tests); defaults to the
    headless claude-cli backend (subscription-billed). Refuses summary rubrics.
    """
    call = client or make_client("claude-cli", model)
    results: list[JudgeResult] = []
    with out_path.open("w") as f:
        for qid, gen in generations.items():
            question = questions.get(qid, "(not recorded)") if questions else "(not recorded)"
            text = call(render_judge_prompt(behavior, gen, question))
            score, reasoning = parse_judge_response(text)
            if score is None:
                raise ValueError(f"judge returned no <score> tag for {qid!r}")
            r = JudgeResult(
                behavior_key=behavior.key,
                question_id=qid,
                generation=gen,
                score=score,
                reasoning=reasoning,
            )
            results.append(r)
            f.write(
                json.dumps(
                    {
                        "behavior": r.behavior_key,
                        "question_id": r.question_id,
                        "score": r.score,
                        "reasoning": r.reasoning,
                        "generation": r.generation,
                    }
                )
                + "\n"
            )
    return results
