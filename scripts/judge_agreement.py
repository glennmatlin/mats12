"""Cross-provider judge agreement check on subscription CLIs (no API keys).

Runs the same rubric prompt over the same generations through several headless
CLI judges in parallel and writes per-backend score JSONLs plus a summary
markdown with the score matrix and pairwise agreement stats.

    uv run scripts/judge_agreement.py \
        --generations <practice_generations.jsonl> --out-dir <dir> \
        --backends claude-cli agy-cli grok-cli codex-cli
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from mats12.behaviors import BEHAVIORS
from mats12.judge import CLI_DEFAULT_MODELS, judge_generations, make_client


def run_backend(backend: str, behavior_key: str, generations: dict, out_dir: Path) -> dict:
    model = CLI_DEFAULT_MODELS.get(backend, "")
    results = judge_generations(
        BEHAVIORS[behavior_key],
        generations,
        model=model,
        out_path=out_dir / f"scores_{backend}.jsonl",
        client=make_client(backend, model),
    )
    return {r.question_id: r.score for r in results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--behavior", default="validate_feelings")
    parser.add_argument(
        "--backends", nargs="+", default=["claude-cli", "agy-cli", "grok-cli", "codex-cli"]
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.generations.read_text().splitlines()]
    generations = {f"practice-{i}": r["completion"] for i, r in enumerate(rows)}

    scores: dict[str, dict[str, float]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(args.backends)) as pool:
        futures = {
            b: pool.submit(run_backend, b, args.behavior, generations, args.out_dir)
            for b in args.backends
        }
        for b, fut in futures.items():
            try:
                scores[b] = fut.result()
            except Exception as exc:  # noqa: BLE001 - report per-backend, don't die
                failures[b] = str(exc)[:200]

    qids = sorted(generations)
    lines = [
        f"# Judge agreement — behavior `{args.behavior}`, n={len(qids)} generations",
        "",
        "| backend | " + " | ".join(qids) + " |",
        "|---|" + "---|" * len(qids),
    ]
    for b in args.backends:
        if b in scores:
            lines.append(f"| {b} | " + " | ".join(f"{scores[b][q]:+.1f}" for q in qids) + " |")
        else:
            lines.append(f"| {b} | FAILED: {failures[b]} |")

    lines += ["", "## Pairwise agreement", ""]
    for a, b in combinations([x for x in args.backends if x in scores], 2):
        diffs = [abs(scores[a][q] - scores[b][q]) for q in qids]
        exact = sum(d == 0 for d in diffs)
        within1 = sum(d <= 1 for d in diffs)
        lines.append(
            f"- {a} vs {b}: exact {exact}/{len(qids)}, within-1 {within1}/{len(qids)}, "
            f"mean |diff| {sum(diffs) / len(diffs):.2f}"
        )

    summary = args.out_dir / "agreement_summary.md"
    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nJUDGE_AGREEMENT_DONE wrote {summary}")


if __name__ == "__main__":
    main()
