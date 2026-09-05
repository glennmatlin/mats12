"""Judge smoke: score practice-loop generations with the validate_feelings rubric.

Default backend is headless Claude Code (`claude -p`), billed to the
subscription — no API key required:
    uv run scripts/judge_smoke.py --generations <practice_generations.jsonl> \
        --out <scores.jsonl>
Use --backend api with ANTHROPIC_API_KEY set (and `--with anthropic`) to match
the anchor post's exact judge model instead.
"""

import argparse
import json
from pathlib import Path

from mats12.behaviors import BEHAVIORS
from mats12.config import JUDGE_MODEL
from mats12.judge import judge_generations, make_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--behavior", default="validate_feelings")
    parser.add_argument("--backend", choices=["claude-cli", "api"], default="claude-cli")
    parser.add_argument(
        "--model", default=None, help="default: 'sonnet' for claude-cli, config JUDGE_MODEL for api"
    )
    args = parser.parse_args()

    model = args.model or ("sonnet" if args.backend == "claude-cli" else JUDGE_MODEL)
    rows = [json.loads(line) for line in args.generations.read_text().splitlines()]
    generations = {f"practice-{i}": r["completion"] for i, r in enumerate(rows)}
    results = judge_generations(
        BEHAVIORS[args.behavior],
        generations,
        model=model,
        out_path=args.out,
        client=make_client(args.backend, model),
    )
    for r in results:
        print(f"{r.question_id}: score={r.score:+.1f}")
    print(f"JUDGE_SMOKE_DONE wrote {args.out}")


if __name__ == "__main__":
    main()
