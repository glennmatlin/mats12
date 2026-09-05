# Application freeze, 2026-09-03

## Decision

The MATS 12 application deadline is 2026-09-04. The remaining arc of this
project (fresh-training executor, four training conditions, EKFAC caches,
SAE feature selection, evaluation, writeup) does not fit inside the
remaining day regardless of parallelization, so the owner froze the project
here for application purposes. The application submission is the sibling
project `~/dev/mats12-persona-filtering` (its Step 1 replication gate
passed in full on 2026-08-22 and its paper builds clean). This repository
resumes later as personal research; nothing here is abandoned, and every
open issue stays open as the resumption route.

## Frozen position

Prerequisite gates B1-B4 are verified complete with committed receipts.
Gate R2 (organism reproduction) was executed once per the accepted policy
(Issue #2) and **failed both behaviors**: validate_feelings mean 3.00 vs
archived reference 1.48, refusal 4.98 vs 4.3267, both beyond the 0.5
threshold. The gate receipt is at
`reports/reconstruction/r2/public-rank32-r2-gate-b3-v1.receipt.json`
(receipt_sha256 12bf1a7c..., commit c81a6ef); per-sample receipts chain
9e720ae4... under machine-local `.runs/r2-scoring-v1/samples/`.

The interpretation evidence gathered after the fail, in order of weight:

1. A side-by-side text comparison of the 9 largest score disagreements
   shows the archived organism genuinely misbehaving (complying with a
   fake-reference request, giving plagiarism help, collapsing into
   incoherent boilerplate) where our locally generated responses refuse
   properly and stay coherent. The judge scored both sets sensibly, so
   judge serving-stack drift is ruled out.
2. Per-question score correlation is weak (pearson 0.299 / 0.173) with
   swings to +-10, inconsistent with a uniform judge shift.
3. The archived eval headers report dirty producing checkouts, and the
   anchor post's own comment thread records a direct replicator failing
   to reproduce its exact numbers. Exact-number replication of this
   organism appears infeasible for anyone; the sibling project
   pre-registered a phenomenon-level gate for exactly this reason.

Standing conclusion at freeze: the public rank-32 adapter runs and speaks
in the organism's register, but its characteristic misbehavior does not
reproduce at archived strength. Whether to rescope R2 to a phenomenon
gate (score a base-model comparison) or switch to a self-trained organism
is the first owner decision at resumption.

## Gate certification at freeze

At the freeze revision, on this branch (`glenn/b7-fresh-training`):

- `uv run ruff check src tests` - clean
- `uv run ruff format --check src tests` - 72 files already formatted
- `uv run pyright` - 0 errors, 0 warnings
- `uv run pytest -q` - 527 passed

## Branch map at freeze

Consolidation into `main` intentionally did not happen at freeze; the
branches below are the package. All are pushed except where noted.

| Branch | Head | Holds |
| --- | --- | --- |
| `glenn/b7-fresh-training` | 13a9ad4 (+ this freeze commit) | The freeze branch and superset lane: Gate R2 executor and receipts (c81a6ef, transport fix 0266123), controls-training authorization (26df935), contains `glenn/b4-base-responses-v2` |
| `glenn/b4-base-responses-v2` | 26df935 | Ancestor of b7; regenerated no-opener base-response set (Issue #23 remedy) |
| `glenn/b4-base-responses` | 917e867 | Diverges from the b7 lane only by logbook auto-append commits |
| `glenn/reconstruction-readiness` | d4e9740 | B2 recovery receipt and production config; **not merged into b7** |
| `glenn/mats-causal-gradient-handoff-f7d2ca` | e50772e | Rank-32 EKFAC smoke receipt with byte-level guards (35952f1); **not merged into b7** |
| `glenn/mats-concept-filtering-backbone-74fcc1` | 7f2cefc | Local-only; adds only logbook auto-appends over b7; safe to delete at resumption |

## Machine-local state (not in git)

- `.runs/` - generation banks and scoring receipts, including
  `rank32-full-v1` (the 201-response organism bank Gate R2 scored) and
  `r2-scoring-v1` (per-sample judge receipts). Receipt hashes in the
  committed gate receipt bind to these bytes.
- `.model-cache/evaluator-recovery/` - the pinned Inspect `.eval`
  archives (sha256 f92fcd6c... and a130e9cc...) holding archived
  reference prompts, responses, and scores.
- `.env` - credentials (gitignored, mode 600); includes the Anthropic
  workspace id required by the identity-linked API key.

## Resumption pointers

- First decision: R2 rescope (phenomenon gate with a base-model scoring
  run) vs self-trained organism; see the interpretation evidence above.
- Then the open issue chain: #6 shared inputs, #10 fresh full-data
  control (plan at `docs/plans/2026-08-28-fresh-training-executor.md`),
  #7/#8/#11 controls, #17 EKFAC caches, #12 evaluation; #13 onward (SAE
  feature selection) remains unauthorized.
- The scoring executor (`src/concept_filtering/r2_provider_scoring.py`)
  is live and tested; the judge model pin and policy live in Issue #2
  and `docs/specs/2026-08-14-gate-r2-decision-memo.md`.
