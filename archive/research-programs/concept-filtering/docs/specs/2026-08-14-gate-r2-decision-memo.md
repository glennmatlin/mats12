# Gate R2 decision memo: public source-artifact candidate

## Decision

**Approve the pinned rank-32 candidate for reconstruction work only; do not use
it scientifically until Gate R2 is resolved.** Receipt-backed official rank-64
artifacts supersede this interim organism when they arrive. The earlier
20-hour figure is advisory and no longer defines this gate.

The owner accepted the public adapter as a provisional organism so missing
rank-64 artifacts no longer block load-and-evaluation reconstruction. Gate R1
still fails because the producing code is unavailable, the exact sample role,
license, selection procedure, and adapter-producer link are unreceipted, and
the published adapter conflicts with the post on LoRA rank. The candidate
sample's exact bytes, schema, and reconstructed row identities are now
replay-verified, while the candidate source revision is pinned as part of that
named identity rule rather than verified as source provenance. The public
trainer state now has a replay-verified observation receipt for one epoch, 123
steps, and its exact learning-rate and token traces, but the optimizer,
configured scheduler, accumulation, initial seed, precision, sequence policy,
and data order are still missing. The full 201 prompt bank has been generated,
and a named evaluator prompt-contract reconstruction exactly reproduces every
archived judge request and strict score parse. The published SAE has also
passed one receipt-bound full-weight layer-16 hook/encode/decode smoke against
the pinned candidate. These results do not prove reference-behavior
reproduction, recover the SAE's training-time base revision, or recover the
dirty producing evaluator. The judge's canonical model ID is now pinned, but
the provider's serving infrastructure is not, and no numerical acceptance
tolerance has been approved.

## Gate status

| Gate | Status | Evidence and consequence |
|---|---|---|
| R0: artifact availability | Documented with blockers | Nine artifact receipts have immutable identifiers or an explicit unavailability check. A named machine profile requires 41 R0 fields, so deleting an unresolved prerequisite cannot make the gate appear complete. |
| R1: executable reconstruction | Fail for source reproduction; named data, generation, SAE-functional, and evaluator-contract verification pass | The 25,007-row candidate has a replay-verified identity receipt, the pinned base, rank-32 adapter, and template generated all 201 prompts, and the full SAE passed one layer-16 hook/encode/decode path under clean receipted executors. The recovered prompt contract matches all 201 archived judge requests and strict scores, and the canonical judge model ID is pinned. The unavailable selection/adapter link, producing checkout, seed, optimizer configuration, SAE training-time base revision, and exact dirty evaluator source still prevent source-protocol reconstruction. |
| R2: reference-behavior decision | Reject for scientific use | Rank 32 is approved only for provisional reconstruction, and there is no reproduced validation or refusal score from a receipt-complete organism. |

## Machine-generated blockers

The corrected reference-to-candidate comparison reports 22 blocking fields:

1. The producing code is unavailable, and no code revision is bound to the
   adapter.
2. The base-model and tokenizer source revisions are not pinned by the study.
3. The exact role of `data/train_1.parquet` is unconfirmed. Its bytes, schema,
   and reconstructed row identities are receipted, and the candidate source
   revision is pinned inside that identity rule. Its license, selection code,
   selection receipt, and adapter-producer link are unavailable.
4. The evaluation checkouts are dirty, the refusal archive has 101 rather than
   the described 100 questions, and the exact producing evaluator source is
   unavailable. The later public scorer and archives recover the exact prompt
   contract and strict parser. Anthropic's official model policy identifies
   `claude-sonnet-4-6` as a canonical pinned model ID, while explicitly leaving
   serving infrastructure outside the fixed weights/configuration guarantee.
5. The SAE metadata does not pin the immutable base revision used for training.
   The apparent architecture conflict is resolved, and the full weights pass a
   one-token layer-16 hook/encode/decode smoke against the pinned candidate,
   but that functional result cannot recover training-time provenance.
6. The post's rank 64 conflicts with the adapter's rank 32. The adapter's
   realized learning-rate and token traces are now frozen, but its sequence
   policy, optimizer, configured scheduler, accumulation, initial seed,
   precision, and data order are not published. The pinned adapter directory
   contains the template selected for the interim smoke, but no producing
   receipt proves its source-study role.

These six groups contain every nonmatching field in the 41-field R0 profile;
the machine report retains the individual entries. The 25,007-row candidate
versus the post's “25K” description is one additional informational divergence
because the prose may be rounded. Nineteen fields match exactly.

## What is executable now

The package provides fail-closed CLI seams for artifact and organism receipts,
deterministic normalized-data manifests, source-to-candidate divergence
reports, and evaluation-result receipts. Evaluation consumes the actual
organism config and stored manifest, reconstructs their identities, and stops
on an expected-hash mismatch. The fixture bundle runs those seams in one
process and writes:

- `dataset-manifest.json`, binding exact input bytes and canonical row content;
- `divergence-report.json`, with per-field categories and blocking effects;
- `evaluation-receipt.json`, binding code, organism, data, evaluator, scores,
  and summaries; and
- `run-receipt.json`, binding all three outputs and marking the run
  `scientific_use=false`.

The [CLI guide](../reconstruction/README.md) gives the commands. Tests inject
mutable revisions, incomplete unavailability checks, duplicate IDs, invalid
token counts, duplicate evaluation questions, score-range failures, path
escapes, and output-overwrite attempts.

The [rank-32 smoke receipt](../../reports/reconstruction/smokes/2026-08-14-public-rank32-split-1.json)
binds executor commit `fd1fafc`, the exact base and adapter revisions, adapter
and template hashes, the loaded template, runtime versions, prompt, and
completion. It records `scientific_use=false` and
`starts_experiment_timer=false`. Its independently recomputed receipt SHA-256 is
`12c2616b1d57dbc6eea67025b1522b71ee267254170e0c2d2e19297feed7aaea`.

The
[evaluator reconstruction receipt](../../reports/reconstruction/evaluators/2026-08-15-public-slop-aware-judge-reconstruction.json)
binds the pinned public scorer, both archives, and the reconstructed prompt and
strict-parser contract. All 201 archived judge requests and integer scores
match. The receipt also binds Anthropic's canonical pinned
`claude-sonnet-4-6` model ID and records that serving infrastructure is not
fixed. `provider_execution_allowed=false`; no provider call or new scoring
occurred.

The
[training-sample reconstruction receipt](../../configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json)
binds the exact 116 MB Parquet file, its four-field schema, all 25,007 rows,
unique `orig_idx` values, split 1, source-stratum counts, file order, and
canonical row content. Its identity rule combines the pinned source revision
with `orig_idx`. Both the config and receipt retain `scientific_use=false` and
`starts_experiment_timer=false`, and the receipt explicitly preserves the
missing license and adapter-producer link as unavailable.

The
[training-recipe evidence receipt](../../configs/reconstruction/receipts/public-rank32-split1-recipe-evidence-v1.json)
binds the public adapter's LoRA configuration, one-epoch/123-step trainer
state, exact learning-rate trace, exact cumulative logged-token trace, and
final RNG-state file. It records the top-level `num_input_tokens_seen` as zero
and the final logged `num_tokens` as 63,696,407, so it does not reinterpret
either field. The receipt explicitly remains non-scientific, does not start the
timer, and is not a runnable recipe.

The
[SAE structural-compatibility receipt](../../configs/reconstruction/receipts/public-layer16-structural-compatibility-v1.json)
binds the pinned base configuration, SAE `cfg.json` and `runner_cfg.json`,
release README, SAELens 6.42.0 source documentation, public artifact receipt,
and the exact 408-byte safetensors header. It resolves the architecture labels
as a documented training-to-inference conversion and proves static
compatibility of the 4,096-wide layer-16 hook and encoder/decoder shapes. It
does not prove the SAE's training-time base revision, recompute the full weights
hash, or execute the model and SAE; all three execution flags remain false.

The
[SAE functional-smoke receipt](../../configs/reconstruction/receipts/public-layer16-functional-smoke-v1.json)
binds executor `8fcb625`, the explicitly pinned runtime file hashes, the
exact six-package runtime, token 9906, the unique resolved layer-16 hook, and
one finite 4,096-to-65,536-to-4,096 encode/decode path. It records functional
execution but keeps scientific use, timer start, feature selection,
intervention execution, and provider execution false. This proves the
mechanical path for the pinned candidate, not the immutable base revision used
to train the SAE or causal usefulness of any feature.

## Proposed scoring policy

The closed
[`public-rank32-r2-scoring-decision-v1`](../../configs/reconstruction/evaluators/public-rank32-r2-scoring-decision-v1.json)
artifact proposes the complete pre-result rule. Score each of the 100 validation
and 101 refusal responses once with the pinned judge contract. Require zero
missing scores, zero parse failures, and an absolute arithmetic-mean difference
no larger than 0.5 points from each archived reference mean. Both behaviors
must pass, and a parseable completion cannot be resampled. The policy binds the
retained 201-response bank through its aggregate receipt, ordered-sample hash,
overall and per-behavior completion-list hashes, and a compact evidence receipt
reconstructed from the run, runtime, aggregate, and all 201 retained sample
receipts.

The 0.5 bound is 5% of the full -5 to +5 scale and approximately twice the
larger archived standard error. It is a pragmatic reconstruction gate, not a
confidence interval. The proposal's file SHA-256 is
`fd9e61dfae52446541ab9a49051c41cc7d668f1f3cb6f879eb7c14a4b7c44c5d`;
its canonical JSON SHA-256 is
`e2e5071cceb8157a9802c144480d5addfb6df2c5de6da4975703fa0775084608`.
The separate proposal receipt has canonical body SHA-256
`a9bfa0ba68c084cd2cd08a108383d634ca5c36535fb6d78b4d64bc1721efff35`.
Its status remains `proposed`; owner acceptance, provider execution, scientific
use, and timer start are all false. Explicit owner acceptance is required
before provider scoring; rejection keeps provider execution prohibited.

## Work required before another R2 review

1. Replace the documented interim rank-32 choice if receipt-backed official
   rank-64 artifacts arrive; otherwise keep every reconstruction claim labeled
   rank 32.
2. Resolve the remaining data provenance around the exact receipted train file:
   obtain its license, selection code/receipt, adapter-producer link,
   tokenizer/chat-template use, sequence policy, and producing data order.
3. Obtain the full source recipe or implement a named reconstruction with a
   frozen optimizer, configured scheduler, accumulation setup, seed, precision,
   sequence policy, and data order. That would be a divergence from the source,
   not recovered source metadata; the receipt already preserves the realized
   trace that the reconstruction should explain or deliberately diverge from.
4. Obtain the SAE's training-time base revision or explicitly accept that
   provenance divergence. The architecture-label and full-weight functional
   questions are resolved for the pinned candidate.
5. Explicitly accept the documented serving-infrastructure boundary and a
   predeclared score tolerance before scoring the completed validation and
   refusal generation banks. The canonical judge model ID itself is resolved.

None of these items expands the frozen experiment. Additional doses, seeds,
selectors, interventions, or evaluation suites remain stretch work.
