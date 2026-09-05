"""Add a structured installation receipt to the R1 analysis artifact.

The receipt pairs the committed full-fine-tuning and base scores by question,
reuses the Step-1 bootstrap convention, checks generation and judgment
completeness, and binds every direct input and producer file by SHA-256.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from judge_sharded import completion_sha
from summarize_step1 import paired_delta_ci

log = logging.getLogger("summarize_r1")

BEHAVIORS = ("both_sides", "refusal", "validate_feelings")
CONDITION = "r1_full_ft"
EXPECTED_PER_BEHAVIOR = 100
EXPECTED_TOTAL = EXPECTED_PER_BEHAVIOR * len(BEHAVIORS)

JsonRow: TypeAlias = dict[str, Any]
FileReceipt: TypeAlias = dict[str, str]
JudgmentKey: TypeAlias = tuple[str, str, str]


@dataclass(frozen=True)
class BehaviorInputs:
    behavior: str
    base_path: Path
    full_ft_path: Path
    generation_path: Path
    base_scores: dict[str, float]
    full_ft_scores: dict[str, float]
    full_ft_rows: list[JsonRow]
    generations: list[JsonRow]


@dataclass
class ReceiptSections:
    behaviors: dict[str, dict[str, object]]
    base_receipts: dict[str, FileReceipt]
    full_ft_receipts: dict[str, FileReceipt]
    generation_receipts: dict[str, FileReceipt]
    generation_rows: list[JsonRow]
    score_rows: int = 0
    content_bound_judgments: int = 0


def sha256(path: Path) -> str:
    """Return a full SHA-256 hex digest for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[JsonRow]:
    """Load nonempty JSONL rows."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_scores(path: Path) -> tuple[dict[str, float], list[JsonRow]]:
    """Load score rows and reject duplicate question identifiers."""
    rows = load_jsonl(path)
    scores = {row["question_id"]: float(row["score"]) for row in rows}
    if len(scores) != len(rows):
        raise ValueError(f"duplicate question ids in {path}")
    return scores, rows


def file_receipt(repo_root: Path, relative_path: Path) -> FileReceipt:
    """Bind a repository-relative path to its current bytes."""
    path = repo_root / relative_path
    return {"path": relative_path.as_posix(), "sha256": sha256(path)}


def read_behavior_inputs(repo_root: Path, behavior: str) -> BehaviorInputs:
    """Load the paired scores and full-FT generations for one behavior."""
    base_path = Path(f"results/step1/judging/scores_base_{behavior}.jsonl")
    full_ft_path = Path(f"results/r1/judging/scores_r1_full_ft_{behavior}.jsonl")
    generation_path = Path(f"results/r1/gens/{behavior}.jsonl")
    base_scores, _ = load_scores(repo_root / base_path)
    full_ft_scores, full_ft_rows = load_scores(repo_root / full_ft_path)
    return BehaviorInputs(
        behavior=behavior,
        base_path=base_path,
        full_ft_path=full_ft_path,
        generation_path=generation_path,
        base_scores=base_scores,
        full_ft_scores=full_ft_scores,
        full_ft_rows=full_ft_rows,
        generations=load_jsonl(repo_root / generation_path),
    )


def index_judgments(rows: list[JsonRow]) -> dict[JudgmentKey, JsonRow]:
    """Index one primary judgment per condition, behavior, and question."""
    required = ("cond", "behavior", "question_id", "provider", "completion_sha", "score", "error")
    indexed: dict[JudgmentKey, JsonRow] = {}
    for row in rows:
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"judgment row missing fields: {missing}")
        key = (row["cond"], row["behavior"], row["question_id"])
        if key in indexed:
            raise ValueError(f"duplicate judgment for {key}")
        indexed[key] = row
    return indexed


def rounded_stats(stats: dict[str, float | int | bool]) -> dict[str, object]:
    """Normalize floating output to the R1 reporting precision."""
    return {
        "n": stats["n"],
        "base_mean": round(float(stats["base_mean"]), 3),
        "full_ft_mean": round(float(stats["sft_mean"]), 3),
        "paired_delta": round(float(stats["delta_mean"]), 3),
        "ci95": [round(float(stats["ci95_lo"]), 3), round(float(stats["ci95_hi"]), 3)],
        "ci_excludes_zero": stats["ci_excludes_zero"],
    }


def validate_question_ids(
    behavior: str,
    base_ids: set[str],
    full_ft_ids: set[str],
    generation_ids: set[str],
    judgment_ids: set[str],
) -> None:
    """Require all R1 evidence layers to describe the same questions."""
    if base_ids != full_ft_ids:
        raise ValueError(f"{behavior} base and full-FT question ids do not match")
    if full_ft_ids != generation_ids:
        raise ValueError(f"{behavior} score and generation question ids do not match")
    if full_ft_ids != judgment_ids:
        raise ValueError(f"{behavior} score and judgment question ids do not match")


def validate_behavior_inputs(inputs: BehaviorInputs, judgments: dict[JudgmentKey, JsonRow]) -> int:
    """Fail closed unless scores, generations, and judgments agree exactly."""
    generation_by_id = {row["question_id"]: row for row in inputs.generations}
    if len(generation_by_id) != len(inputs.generations):
        raise ValueError(f"duplicate question ids in {inputs.generation_path}")
    full_ft_by_id = {row["question_id"]: row for row in inputs.full_ft_rows}
    behavior_judgments = {
        question_id: row
        for (cond, behavior, question_id), row in judgments.items()
        if cond == CONDITION and behavior == inputs.behavior
    }
    validate_question_ids(
        inputs.behavior,
        set(inputs.base_scores),
        set(inputs.full_ft_scores),
        set(generation_by_id),
        set(behavior_judgments),
    )
    for question_id, generation in generation_by_id.items():
        judgment = behavior_judgments[question_id]
        score_row = full_ft_by_id[question_id]
        if completion_sha(generation["completion"]) != judgment["completion_sha"]:
            raise ValueError(f"{question_id} judgment does not match generation")
        if score_row["generation"] != generation["completion"]:
            raise ValueError(f"{question_id} score row does not match generation")
        if score_row["judge"] != judgment["provider"]:
            raise ValueError(f"{question_id} score row does not match judgment provider")
        if float(score_row["score"]) != float(judgment["score"]):
            raise ValueError(f"{question_id} score row does not match judgment score")
        if not generation["completion"].strip() or generation["think_truncated"] is not False:
            raise ValueError(f"{question_id} generation is incomplete or truncated")
        if judgment["error"] is not None:
            raise ValueError(f"{question_id} judgment has an error")
    return len(behavior_judgments)


def collect_sections(
    repo_root: Path,
    judgments: dict[JudgmentKey, JsonRow],
    rng: np.random.Generator,
    n_boot: int,
) -> ReceiptSections:
    """Collect behavior statistics and their direct input receipts."""
    sections = ReceiptSections({}, {}, {}, {}, [])
    for behavior in BEHAVIORS:
        inputs = read_behavior_inputs(repo_root, behavior)
        sections.content_bound_judgments += validate_behavior_inputs(inputs, judgments)
        sections.behaviors[behavior] = rounded_stats(
            paired_delta_ci(inputs.base_scores, inputs.full_ft_scores, rng, n_boot)
        )
        sections.base_receipts[behavior] = file_receipt(repo_root, inputs.base_path)
        sections.full_ft_receipts[behavior] = file_receipt(repo_root, inputs.full_ft_path)
        sections.generation_receipts[behavior] = file_receipt(repo_root, inputs.generation_path)
        sections.generation_rows.extend(inputs.generations)
        sections.score_rows += len(inputs.full_ft_scores)
    return sections


def completeness_receipt(
    sections: ReceiptSections, judgment_rows: list[JsonRow], judgments: dict[JudgmentKey, JsonRow]
) -> dict[str, int | bool]:
    """Require and report the preregistered R1 triple's exact completeness."""
    completeness = {
        "generation_rows": len(sections.generation_rows),
        "nonempty_generations": sum(
            bool(row["completion"].strip()) for row in sections.generation_rows
        ),
        "think_truncated_generations": sum(
            row["think_truncated"] is not False for row in sections.generation_rows
        ),
        "score_rows": sections.score_rows,
        "judgment_rows": len(judgment_rows),
        "unique_judgments": len(judgments),
        "content_bound_judgments": sections.content_bound_judgments,
        "judgment_errors": sum(row["error"] is not None for row in judgment_rows),
        "question_id_sets_match": True,
    }
    expected = {
        "generation_rows": EXPECTED_TOTAL,
        "nonempty_generations": EXPECTED_TOTAL,
        "think_truncated_generations": 0,
        "score_rows": EXPECTED_TOTAL,
        "judgment_rows": EXPECTED_TOTAL,
        "unique_judgments": EXPECTED_TOTAL,
        "content_bound_judgments": EXPECTED_TOTAL,
        "judgment_errors": 0,
    }
    actual = {field: completeness[field] for field in expected}
    if actual != expected:
        raise ValueError(f"R1 completeness mismatch: expected {expected}, got {actual}")
    return completeness


def validate_behavior_counts(sections: ReceiptSections) -> None:
    """Require exactly 100 paired score rows for each R1 behavior."""
    wrong = {
        behavior: stats["n"]
        for behavior, stats in sections.behaviors.items()
        if stats["n"] != EXPECTED_PER_BEHAVIOR
    }
    if wrong:
        raise ValueError(f"R1 behavior counts must each be {EXPECTED_PER_BEHAVIOR}: {wrong}")


def build_installation_receipt(repo_root: Path, seed: int, n_boot: int) -> dict[str, object]:
    """Build the deterministic, content-bound R1 installation receipt."""
    raw_path = Path("results/r1/judging/raw_judgments.jsonl")
    judgment_rows = load_jsonl(repo_root / raw_path)
    judgments = index_judgments(judgment_rows)
    sections = collect_sections(repo_root, judgments, np.random.default_rng(seed), n_boot)
    validate_behavior_counts(sections)

    return {
        "producer": {
            "files": [
                file_receipt(repo_root, Path("scripts/summarize_r1.py")),
                file_receipt(repo_root, Path("scripts/summarize_step1.py")),
                file_receipt(repo_root, Path("scripts/judge_sharded.py")),
            ]
        },
        "bootstrap": {
            "method": "paired question-level percentile bootstrap",
            "rng": "numpy.random.default_rng",
            "seed": seed,
            "resamples": n_boot,
            "percentiles": [2.5, 97.5],
            "behavior_order": list(BEHAVIORS),
        },
        "inputs": {
            "base_scores": sections.base_receipts,
            "full_ft_scores": sections.full_ft_receipts,
            "generations": sections.generation_receipts,
            "judgments": file_receipt(repo_root, raw_path),
        },
        "completeness": completeness_receipt(sections, judgment_rows, judgments),
        "behaviors": sections.behaviors,
    }


def write_analysis(repo_root: Path, out_path: Path, seed: int, n_boot: int) -> None:
    """Preserve the R1 analysis and replace its generated receipt."""
    analysis = json.loads(out_path.read_text())
    analysis["installation_receipt"] = build_installation_receipt(repo_root, seed, n_boot)
    out_path.write_text(json.dumps(analysis, indent=2) + "\n")


def main() -> None:
    """Regenerate the structured receipt at the requested output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/r1/r1_analysis.json"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_path = args.out if args.out.is_absolute() else args.repo_root / args.out
    write_analysis(args.repo_root, out_path, args.seed, args.n_boot)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
