"""Judge a directory of per-behavior generation JSONLs; write scores + summary.

Each <behavior>.jsonl row: {question_id, question, completion}. Behaviors run
in parallel threads (one judge stream per behavior); items within a behavior
are sequential. Backend defaults to headless claude-cli (subscription).
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from mats12.behaviors import BEHAVIORS
from mats12.judge import CLI_DEFAULT_MODELS, judge_generations, make_client


def judge_one(behavior_key: str, gen_path: Path, out_dir: Path, backend: str) -> dict:
    rows = [json.loads(line) for line in gen_path.read_text().splitlines()]
    generations = {r["question_id"]: r["completion"] for r in rows}
    questions = {r["question_id"]: r["question"] for r in rows}
    model = CLI_DEFAULT_MODELS.get(backend, "")
    results = judge_generations(
        BEHAVIORS[behavior_key],
        generations,
        model=model,
        out_path=out_dir / f"scores_{behavior_key}.jsonl",
        client=make_client(backend, model),
        questions=questions,
    )
    scores = [r.score for r in results]
    return {"behavior": behavior_key, "n": len(scores), "mean": mean(scores), "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", default="claude-cli")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = {p.stem: p for p in sorted(args.gen_dir.glob("*.jsonl")) if p.stem in BEHAVIORS}
    with ThreadPoolExecutor(max_workers=len(gen_files)) as pool:
        futures = {
            k: pool.submit(judge_one, k, p, args.out_dir, args.backend)
            for k, p in gen_files.items()
        }
        summaries = {k: f.result() for k, f in futures.items()}

    lines = ["| behavior | n | mean score | scores |", "|---|---|---|---|"]
    for k in sorted(summaries):
        s = summaries[k]
        lines.append(
            f"| {k} | {s['n']} | {s['mean']:+.2f} | "
            + " ".join(f"{x:+.0f}" for x in s["scores"])
            + " |"
        )
    summary = "\n".join(lines)
    (args.out_dir / "summary.md").write_text(summary + "\n")
    print(summary)
    print(f"JUDGE_EVAL_RUN_DONE wrote {args.out_dir}/summary.md")


if __name__ == "__main__":
    main()
