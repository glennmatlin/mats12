"""Sharded multi-provider judging for Step-1-scale runs (PLAN.md protocol).

Phases:
  A. Revalidation: ~50 items from the replication triple judged by EVERY
     provider; pairwise within-±1 agreement decides shard vs claude-only mode
     (the n=3 smoke agreement was plumbing proof, not evidence).
  B. Production: each item judged by its hash-assigned provider (hash of the
     question key only, so conditions stay within-judge per question); a ~12%
     overlap subset is judged by all providers to monitor drift.

Every judgment appends to raw_judgments.jsonl immediately and reruns skip
already-judged items — the run is resumable, and rerunning later with the SFT
condition added reuses all base judgments. The resume key binds the judged
bytes: (condition, behavior, question_id, provider, sha256(completion)[:12]),
so a regenerated completion is re-judged instead of inheriting a stale score,
and error rows (score=None) are retried on rerun (the append-only ledger
keeps them for audit; the latest row per key wins on replay).
"""

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from mats12.behaviors import BEHAVIORS, REPLICATION_TRIPLE
from mats12.judge import (
    CLI_DEFAULT_MODELS,
    make_client,
    parse_judge_response,
    render_judge_prompt,
)
from mats12.sharding import assign_provider, is_overlap, revalidation_subset

DoneKey = tuple[str, str, str, str, str]  # (cond, behavior, question_id, provider, sha)


def completion_sha(completion: str) -> str:
    return hashlib.sha256(completion.encode()).hexdigest()[:12]


def item_key(item: dict, provider: str) -> DoneKey:
    return (item["cond"], item["behavior"], item["qid"], provider, item["sha"])


def load_items(gen_dirs: dict[str, Path]) -> list[dict]:
    items = []
    for cond, gen_dir in gen_dirs.items():
        for path in sorted(gen_dir.glob("*.jsonl")):
            if path.stem not in BEHAVIORS:
                continue
            for line in path.read_text().splitlines():
                r = json.loads(line)
                items.append(
                    {
                        "cond": cond,
                        "behavior": path.stem,
                        "qid": r["question_id"],
                        "question": r["question"],
                        "completion": r["completion"],
                        "sha": completion_sha(r["completion"]),
                        "think_truncated": r.get("think_truncated", False),
                    }
                )
    return items


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[DoneKey, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.rows[self._row_key(r)] = r  # last line per key wins

    @staticmethod
    def _row_key(row: dict) -> DoneKey:
        # Legacy rows without completion_sha get "" and never match a live
        # item's sha, so they are re-judged rather than silently reused.
        return (
            row["cond"],
            row["behavior"],
            row["question_id"],
            row["provider"],
            row.get("completion_sha", ""),
        )

    def record(self, row: dict) -> None:
        with self.lock:
            self.rows[self._row_key(row)] = row
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    def has(self, key: DoneKey) -> bool:
        row = self.rows.get(key)
        return row is not None and row["score"] is not None


def run_batch(work: list[tuple[dict, str]], ledger: Ledger, clients: dict, threads: int) -> None:
    """Judge (item, provider) pairs, fanned out per provider, resume-aware."""
    todo = [(it, prov) for it, prov in work if not ledger.has(item_key(it, prov))]
    if not todo:
        return
    counters = {prov: 0 for _, prov in todo}
    lock = threading.Lock()

    def judge_one(item: dict, provider: str) -> None:
        prompt = render_judge_prompt(
            BEHAVIORS[item["behavior"]], item["completion"], item["question"]
        )
        score, reasoning, error = None, "", None
        for attempt in (1, 2):
            try:
                text = clients[provider](prompt)
                score, reasoning = parse_judge_response(text)
                if score is not None:
                    error = None
                    break
                error = "no_score_tag"
            except Exception as exc:  # CLI failure/timeout; retry once
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if attempt == 1:
                time.sleep(15)
        ledger.record(
            {
                "cond": item["cond"],
                "behavior": item["behavior"],
                "question_id": item["qid"],
                "provider": provider,
                "completion_sha": item["sha"],
                "score": score,
                "reasoning": reasoning,
                "error": error,
            }
        )
        with lock:
            counters[provider] += 1
            done = counters[provider]
        if done % 25 == 0:
            print(f"  {provider}: {done} judged")

    pools = {prov: ThreadPoolExecutor(max_workers=threads) for prov in {p for _, p in todo}}
    futures = [pools[prov].submit(judge_one, item, prov) for item, prov in todo]
    for f in futures:
        f.result()
    for pool in pools.values():
        pool.shutdown()


def pairwise_agreement(ledger: Ledger, items: list[dict], providers: list[str]) -> dict:
    stats = {}
    for i, a in enumerate(providers):
        for b in providers[i + 1 :]:
            diffs = []
            for it in items:
                ra, rb = ledger.rows.get(item_key(it, a)), ledger.rows.get(item_key(it, b))
                if ra and rb and ra["score"] is not None and rb["score"] is not None:
                    diffs.append(abs(ra["score"] - rb["score"]))
            stats[f"{a}|{b}"] = {
                "n": len(diffs),
                "within_1": round(sum(d <= 1 for d in diffs) / len(diffs), 3) if diffs else None,
                "mean_abs_diff": round(mean(diffs), 2) if diffs else None,
            }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gen-dir",
        action="append",
        required=True,
        metavar="COND=PATH",
        help="e.g. --gen-dir base=results/gens-base --gen-dir sft=results/gens-sft",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--providers", default="claude-cli,grok-cli,codex-cli")
    parser.add_argument("--overlap-frac", type=float, default=0.12)
    parser.add_argument("--revalidate-n", type=int, default=50)
    parser.add_argument("--agreement-threshold", type=float, default=0.85)
    parser.add_argument("--threads", type=int, default=4, help="per provider")
    args = parser.parse_args()

    gen_dirs = {}
    for spec in args.gen_dir:
        cond, _, path = spec.partition("=")
        gen_dirs[cond] = Path(path)
    providers = args.providers.split(",")
    clients = {p: make_client(p, CLI_DEFAULT_MODELS.get(p, "")) for p in providers}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(args.out_dir / "raw_judgments.jsonl")

    items = load_items(gen_dirs)
    judgeable = [it for it in items if it["completion"].strip()]
    empties = [it for it in items if not it["completion"].strip()]
    print(
        f"items: {len(items)} ({len(judgeable)} judgeable, {len(empties)} empty; "
        f"conditions {sorted(gen_dirs)})"
    )

    # Phase A: revalidation on the replication triple, all providers.
    mode = "claude-only" if len(providers) == 1 else "sharded"
    agreement: dict = {}
    if len(providers) > 1:
        triple = [it for it in judgeable if it["behavior"] in REPLICATION_TRIPLE]
        by_key = {f"{it['cond']}:{it['qid']}": it for it in triple}
        reval_keys = revalidation_subset(sorted(by_key), args.revalidate_n)
        reval_items = [by_key[k] for k in reval_keys]
        print(f"phase A: revalidating {len(reval_items)} items x {len(providers)} providers")
        run_batch([(it, p) for it in reval_items for p in providers], ledger, clients, args.threads)
        agreement = pairwise_agreement(ledger, reval_items, providers)
        worst = min(
            (s["within_1"] for s in agreement.values() if s["within_1"] is not None),
            default=0.0,
        )
        mode = "sharded" if worst >= args.agreement_threshold else "claude-only"
        print(f"phase A agreement (within +/-1): {agreement} -> mode={mode}")

    (args.out_dir / "agreement_report.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "providers": providers,
                "threshold": args.agreement_threshold,
                "pairwise_within_1": agreement,
            },
            indent=2,
        )
    )

    # Phase B: production judging.
    work: list[tuple[dict, str]] = []
    for it in judgeable:
        if mode == "sharded":
            need = (
                set(providers)
                if is_overlap(it["qid"], args.overlap_frac)
                else {assign_provider(it["qid"], providers)}
            )
        else:
            need = {providers[0]}
        work.extend((it, p) for p in need)
    print(f"phase B: {len(work)} (item, provider) pairs (mode={mode})")
    run_batch(work, ledger, clients, args.threads)

    # Primary-score extraction + per-condition outputs.
    summary_rows = []
    for cond in sorted(gen_dirs):
        for behavior in sorted({it["behavior"] for it in items if it["cond"] == cond}):
            b_items = [it for it in items if it["cond"] == cond and it["behavior"] == behavior]
            out_rows, scores, n_fallback = [], [], 0
            for it in b_items:
                if not it["completion"].strip():
                    continue
                primary = (
                    assign_provider(it["qid"], providers) if mode == "sharded" else providers[0]
                )
                row = ledger.rows.get(item_key(it, primary))
                if row is None or row["score"] is None:
                    for p in providers:  # fallback to any valid judgment of THESE bytes
                        alt = ledger.rows.get(item_key(it, p))
                        if alt is not None and alt["score"] is not None:
                            row, n_fallback = alt, n_fallback + 1
                            break
                if row is None or row["score"] is None:
                    continue
                scores.append(row["score"])
                out_rows.append(
                    {
                        "behavior": behavior,
                        "question_id": it["qid"],
                        "score": row["score"],
                        "reasoning": row["reasoning"],
                        "generation": it["completion"],
                        "judge": row["provider"],
                    }
                )
            out_path = args.out_dir / f"scores_{cond}_{behavior}.jsonl"
            out_path.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
            summary_rows.append(
                {
                    "cond": cond,
                    "behavior": behavior,
                    "n": len(scores),
                    "mean": round(mean(scores), 2) if scores else None,
                    "n_empty": sum(1 for it in b_items if not it["completion"].strip()),
                    "n_fallback": n_fallback,
                }
            )

    conds = sorted(gen_dirs)
    lines = [
        f"mode: {mode}",
        "",
        "| behavior | " + " | ".join(conds) + " | delta |",
        "|---" * (len(conds) + 2) + "|",
    ]
    behaviors = sorted({r["behavior"] for r in summary_rows})
    table = {(r["cond"], r["behavior"]): r for r in summary_rows}
    for b in behaviors:
        cells = []
        for c in conds:
            r = table.get((c, b))
            cells.append(f"{r['mean']:+.2f} (n={r['n']})" if r and r["mean"] is not None else "-")
        delta = "-"
        if len(conds) == 2:
            r0, r1 = table.get((conds[0], b)), table.get((conds[1], b))
            if r0 and r1 and r0["mean"] is not None and r1["mean"] is not None:
                delta = f"{r1['mean'] - r0['mean']:+.2f}"
        lines.append(f"| {b} | " + " | ".join(cells) + f" | {delta} |")
    summary = "\n".join(lines)
    (args.out_dir / "summary.md").write_text(summary + "\n")
    (args.out_dir / "summary_rows.json").write_text(json.dumps(summary_rows, indent=1))
    print(summary)
    print(f"JUDGE_SHARDED_DONE wrote {args.out_dir}/summary.md")


if __name__ == "__main__":
    main()
