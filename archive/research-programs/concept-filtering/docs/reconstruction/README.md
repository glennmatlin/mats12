# Reconstruction CLI

The CLI provides provenance, fixture, local-inference, and reconstruction
verification paths. Commands never download or execute a provider unless their
documented contract explicitly says so. Install the locked development
environment with:

```bash
UV_CACHE_DIR=/private/tmp/mats-concept-filtering-uv-cache uv sync --dev
```

## Replication-backbone command shell

The fixture contract exercises the complete B0-B9 orchestration boundary
without model, provider, scheduler, scientific, feature-selection, or concept-
training side effects. Plan and status are read-only:

```bash
uv run concept-filtering backbone plan \
  --config configs/backbone/fixture-replication-backbone-v1.json

uv run concept-filtering backbone status \
  --config configs/backbone/fixture-replication-backbone-v1.json
```

Run through a bounded stage, resume through the terminal backbone stage, then
verify the readiness handoff:

```bash
uv run concept-filtering backbone run \
  --config configs/backbone/fixture-replication-backbone-v1.json \
  --through B4

uv run concept-filtering backbone run \
  --config configs/backbone/fixture-replication-backbone-v1.json \
  --through B9

uv run concept-filtering backbone verify \
  --config configs/backbone/fixture-replication-backbone-v1.json
```

The default run directory is `.runs/backbone/<backbone-id>`. Use the optional
`--run-dir` argument to isolate a fixture or test run elsewhere.

Every stage receipt binds the complete configuration, dependency-receipt order,
release tuple, parameters, capability, and an output release derived from the
executor. A valid receipt is replayed; a stale, reordered, duplicated, missing,
or contradictory receipt fails closed. Incomplete verification exits 3 and
does not create a readiness receipt. Complete verification writes one
deterministic receipt whose next scientific action is SAE feature selection.

The checked-in contract is intentionally fixture-only. Later tickets replace
individual fixture adapters with real recipe, provider, data, selector,
training, and evaluation adapters; no fixture flag authorizes those side
effects.

## Shared training-input package

Materialize the content-free full-data package from the pinned sample and
tokenizer, then replay it against the same immutable producer releases:

```bash
mkdir -p .runs/training-data
uv run concept-filtering materialize-training-data-package \
  configs/training/public-rank32-training-data-package-v1.json \
  --training-recipe configs/reconstruction/training/public-rank32-named-recipe-v1.json \
  --training-sample-receipt configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json \
  --parquet <pinned-train-1.parquet> \
  --tokenizer-snapshot <pinned-adapter-split-1> \
  --code-revision <40-character-executor-commit> \
  --package .runs/training-data/public-rank32-training-data-package.jsonl \
  --output .runs/training-data/public-rank32-training-data-package-receipt.json

uv run concept-filtering verify-training-data-package \
  configs/training/public-rank32-training-data-package-v1.json \
  --training-recipe configs/reconstruction/training/public-rank32-named-recipe-v1.json \
  --training-sample-receipt configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json \
  --parquet <pinned-train-1.parquet> \
  --tokenizer-snapshot <pinned-adapter-split-1> \
  --code-revision <40-character-executor-commit> \
  --package .runs/training-data/public-rank32-training-data-package.jsonl \
  --receipt .runs/training-data/public-rank32-training-data-package-receipt.json
```

The retained receipt is
[`2026-08-16-public-rank32-training-data-package-v1.json`](../../reports/reconstruction/data/2026-08-16-public-rank32-training-data-package-v1.json).
It binds 25,007 stable row identities, 63,639,095 rendered input tokens,
58,414,917 supervised tokens, the exact six-file tokenizer snapshot, and the
named four-condition recipe core. The package contains hashes and counts but no
prompt or response plaintext. It does not authorize base-response generation,
activation extraction, training, feature selection, concept scoring, provider
execution, scheduler execution, scientific use, or the experiment timer.

## Base-response generation

The `base-responses` family executes the frozen base-only generation contract
against a production config that binds the retained package receipt and the
corrected terminal B2 receipt. No production config exists yet; it may only be
created after the corrected B2 receipt is durably closed and reviewed.

```bash
uv run concept-filtering base-responses plan \
  --config <frozen-base-response-config> \
  --package .runs/training-data/public-rank32-training-data-package.jsonl \
  --parquet <pinned-train-1.parquet> \
  --base-snapshot <pinned-olmo3-7b-base-snapshot> \
  --tokenizer-snapshot <pinned-adapter-snapshot>

uv run concept-filtering base-responses run-shard \
  --config <frozen-base-response-config> \
  --package ... --parquet ... --base-snapshot ... --tokenizer-snapshot ... \
  --shard-index 0 --shard-count 1 \
  --output-dir .runs/base-responses/samples \
  --device cuda --dtype bfloat16

uv run concept-filtering base-responses status \
  --config <frozen-base-response-config> \
  --package ... --parquet ... \
  --output-dir .runs/base-responses/samples

uv run concept-filtering base-responses finalize \
  --config <frozen-base-response-config> \
  --package ... --parquet ... \
  --output-dir .runs/base-responses/samples \
  --code-revision <40-character-executor-commit> \
  --output .runs/base-responses/aggregate-receipt.json

uv run concept-filtering base-responses verify \
  --config <frozen-base-response-config> \
  --package ... --parquet ... \
  --output-dir .runs/base-responses/samples \
  --receipt .runs/base-responses/aggregate-receipt.json
```

`plan`, `status`, and `verify` never load the model. `run-shard` is the only
command that generates: it re-authenticates the package/source join, claims
examples by retained row ordinal modulo the shard count, writes one atomic
signed sample per example, and replays existing valid samples without
resampling. `--device` and `--dtype` place execution (default: CPU float32);
placement is an executor fact recorded in the scheduler plan and compute
ledger, not part of the signed sample identity. `finalize` publishes the
absent-only aggregate receipt after exact
25,007-identity coverage. Full-set execution follows the release-bound
scheduler plan in `docs/plans/2026-08-20-base-response-scheduler-plan.md`;
no Slurm submission is granted by these commands.

## Bounded layer-16 activation contract smoke

Proves that one retained example flows through the frozen layer-16 hook and
the declared aggregation on the base model. Consumes the aggregate
base-response receipt; loads no SAE weights.

```bash
uv run concept-filtering activation-contract-smoke \
  --config configs/reconstruction/sae/public-layer16-activation-contract-v1.json \
  --generation-config configs/training/public-rank32-base-responses-v1.json \
  --package ... --parquet ... \
  --output-dir .runs/base-responses/samples \
  --base-snapshot ... --tokenizer-snapshot ... \
  --code-revision <40-hex commit of the clean checkout> \
  --output .runs/base-responses/activation-contract-receipt.json \
  --device cuda --dtype bfloat16
```

The receipt is published absent-only and binds the config, the aggregate
receipt, the artifact manifests, the runtime versions, and the executor
revision, which must be exactly the running checkout with a byte-clean tree.
The stage fails closed on config drift, expanded capability, a mismatched
aggregate receipt, a rendering that does not carry the prompt as a prefix, an
empty response span, a non-finite tensor, a trailing dimension that disagrees
with the declared hidden size, and an unexpected forward or hook-capture
count. It selects, ranks, intervenes on, and scores nothing.

Note that the pinned chat template opens a `<think>` block under
`add_generation_prompt` and omits it when rendering a supplied assistant turn.
Both responses are therefore placed in the identical generation context so
they differ only in their response tokens. The same opener means most
published base responses are reasoning text rather than answers; see the
LOGBOOK entry of 2026-08-23.

Validate the complete public candidate:

```bash
uv run concept-filtering validate-organism \
  configs/reconstruction/public-organism-v1.json
```

Generate the Gate R2 divergence report:

```bash
uv run concept-filtering compare \
  --reference configs/reconstruction/source-study-reference.json \
  --organism configs/reconstruction/public-organism-v1.json \
  --output /private/tmp/concept-filtering-divergence.json
```

`compare` exits 3 when the report contains a blocking divergence. That is a
gate result, not a malformed-input error; validation and I/O errors exit 2.

Run the explicitly non-scientific fixture after replacing
`<40-character-code-commit>` with the exact commit containing the executor:

```bash
uv run concept-filtering dry-run \
  configs/reconstruction/offline-dry-run.json \
  --code-revision <40-character-code-commit> \
  --output-dir /private/tmp/concept-filtering-dry-run
```

The output directory must be absent or empty, which prevents a dry run from
overwriting prior receipts. Fixture evaluator configurations are marked
`scientific_use=false`; configurations marked for scientific use must declare
the receipted count for every behavior and at least 100 questions each.

Other public seams are available through `validate-receipt`, `build-manifest`,
and `evaluate`. The evaluation command takes the organism config, stored
manifest, and manifest source file, recomputes both upstream identities, and
compares them with the expected hashes before it writes a result receipt:

```bash
uv run concept-filtering evaluate \
  --scores <scores.jsonl> \
  --evaluator <evaluator.json> \
  --code-revision <40-character-code-commit> \
  --organism <organism.json> \
  --manifest <dataset-manifest.json> \
  --sample <manifest-source.jsonl> \
  --expected-organism-sha256 <expected-organism-sha256> \
  --expected-manifest-sha256 <expected-manifest-sha256> \
  --output <evaluation-receipt.json>
```

A pinned Inspect archive can be normalized into the score JSONL consumed by
`evaluate` without installing Inspect:

```bash
uv run concept-filtering extract-inspect-eval source.eval \
  --expected-sha256 <published-file-sha256> \
  --output /private/tmp/validate-feelings-scores.jsonl \
  --receipt /private/tmp/validate-feelings-extraction.json
```

The extractor derives the behavior label from the two approved source-task
mappings. It rejects an unapproved task, hash mismatch, empty or incomplete
header, duplicate question, header/sample disagreement, non-finite score,
unsafe ZIP path, or oversized member before writing the normalized rows and a
durable source-to-output receipt. Use `uv run concept-filtering --help` for all
arguments. Recheck a stored receipt and normalized file together with
`validate-inspect-extraction <receipt> --output <scores.jsonl>`.

## Named rank-32 training sample

After downloading the one pinned public Parquet file, verify its exact bytes,
schema, split, stable identities, order, and row content:

```bash
uv run concept-filtering verify-training-sample-reconstruction \
  configs/reconstruction/data/public-rank32-train1-reconstruction-v1.json \
  --parquet .model-cache/dare-data-pinned/data/train_1.parquet \
  --output /private/tmp/public-rank32-train1-reconstruction-v1.json
```

The checked-in
[receipt](../../configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json)
covers all 25,007 rows and derives stable identities from the pinned source
revision plus unique `orig_idx`. It explicitly does not prove the source
study's sampling procedure, license, or the file's producer link to adapter
`split-1`. The command refuses to overwrite evidence and leaves
`scientific_use=false` and `starts_experiment_timer=false`.

## Public rank-32 training-recipe evidence

Verify the three pinned adapter files that expose recipe observations:

```bash
uv run concept-filtering verify-training-recipe-reconstruction \
  configs/reconstruction/training/public-rank32-split1-recipe-evidence-v1.json \
  --artifact-dir .model-cache/rank32-hf-cache/models--GaloisTheory123--dare-adapter/snapshots/767c0ec9f3be0c469ccec246511ca46ac2812c5e/split-1 \
  --output /private/tmp/public-rank32-split1-recipe-evidence-v1.json
```

The checked-in
[receipt](../../configs/reconstruction/receipts/public-rank32-split1-recipe-evidence-v1.json)
binds the LoRA structure, one epoch, 123 logged steps, exact learning-rate and
cumulative logged-token traces, and the final RNG-state file. It also preserves
the missing optimizer, configured scheduler, accumulation, initial seed,
precision, packing/sequence policy, and data order as unavailable. This is an
observation receipt, not a runnable training recipe or training authorization;
both boundary flags remain false.

## Superseded pinned-7B recovery attempt

The superseded attempt used the following command from a clean checkout at
`b594b488ac54251a82304374750c43130c5869d6`, where the relative config and
fixture paths resolve to their historical final-turn-only bytes. It is an
execution record, not a current replay command:

```bash
.venv/bin/concept-filtering run-training-recovery-smoke \
  configs/reconstruction/training/public-rank32-named-recipe-v1.json \
  --code-revision b594b488ac54251a82304374750c43130c5869d6 \
  --base-snapshot .model-cache/rank32-hf-cache/models--allenai--Olmo-3-1025-7B/snapshots/a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --tokenizer-snapshot .model-cache/rank32-hf-cache/models--GaloisTheory123--dare-adapter/snapshots/767c0ec9f3be0c469ccec246511ca46ac2812c5e \
  --fixture configs/reconstruction/training/fixtures/public-rank32-recovery-smoke-fixture-v1.json \
  --work-dir .runs/public-rank32-training-recovery-v1 \
  --output reports/reconstruction/smokes/2026-08-16-public-rank32-training-recovery-v1.json
```

The checked-in
[receipt](../../reports/reconstruction/smokes/2026-08-16-public-rank32-training-recovery-v1.json)
shows that uninterrupted and checkpoint-recovered execution produced identical
two-step losses, identical terminal adapter tensors, and a maximum absolute
difference of zero. It binds the frozen synthetic preprocessing outcome, all
seven durable checkpoint files, the recipe-core identity, and the separately
hashed all-false execution authority. Real-data inspection found that this
executor supervised only the final assistant turn despite 583 multi-turn source
rows, so the receipt is superseded and does not close B2. Corrected multi-turn
rendering is now bound to the same four-condition recipe core and frozen
preprocessing outcome, but a new pinned-7B terminal receipt is still required.
Neither the superseded run nor the repair starts the experiment timer, selects
SAE features, scores concepts, or trains a scientific condition.

## Published layer-16 SAE structural compatibility

After fetching the pinned small metadata files, the exact 408-byte safetensors
header, and the pinned SAELens 6.42.0 training documentation, verify the static
architecture and shape relationship:

```bash
uv run concept-filtering verify-sae-compatibility \
  configs/reconstruction/sae/public-layer16-structural-compatibility-v1.json \
  --base-config .model-cache/rank32-hf-cache/models--allenai--Olmo-3-1025-7B/snapshots/a81bae42db3975be1671e27b9c9a56da1a9f980f/config.json \
  --sae-cfg .model-cache/sae-metadata/olmo-3-1025-7b/btk-mat-layer-16-k-100/cfg.json \
  --runner-cfg .model-cache/sae-metadata/olmo-3-1025-7b/btk-mat-layer-16-k-100/runner_cfg.json \
  --release-readme .model-cache/sae-metadata/README.md \
  --weights-header .model-cache/sae-metadata/olmo-3-1025-7b/btk-mat-layer-16-k-100/sae_weights.header \
  --saelens-docs .model-cache/sae-metadata/saelens-v6.42.0-training_saes.md \
  --artifact-receipt configs/reconstruction/receipts/sae-layer-16.json \
  --output /private/tmp/public-layer16-structural-compatibility-v1.json
```

The checked-in
[receipt](../../configs/reconstruction/receipts/public-layer16-structural-compatibility-v1.json)
shows that SAELens intentionally saves the Matryoshka BatchTopK training SAE as
JumpReLU for inference, with a threshold tensor, and that the 4,096-wide
layer-16 hook and encoder/decoder shapes match the pinned base candidate. This
is a static metadata check: it does not recompute the full 2.1 GB weights hash,
recover the SAE's training-time base revision, load the weights, or execute an
encode/decode path. All three execution flags remain false.

## Published layer-16 SAE functional smoke

The functional receipt was produced from the clean executor commit recorded in
the command and the exact cached snapshots:

```bash
UV_CACHE_DIR=/private/tmp/mats-concept-filtering-uv-cache \
  uv run --frozen --extra inference concept-filtering \
  run-sae-functional-smoke \
  configs/reconstruction/sae/public-layer16-functional-smoke-v1.json \
  --code-revision 8fcb6252de0869d04d3f0bd9c422f1ef14f72ee1 \
  --base-snapshot .model-cache/rank32-hf-cache/models--allenai--Olmo-3-1025-7B/snapshots/a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --adapter-snapshot .model-cache/rank32-hf-cache/models--GaloisTheory123--dare-adapter/snapshots/767c0ec9f3be0c469ccec246511ca46ac2812c5e/split-1 \
  --sae-directory .model-cache/sae-full/olmo-3-1025-7b/btk-mat-layer-16-k-100 \
  --output /private/tmp/public-layer16-functional-smoke-v1.json
```

The checked-in
[receipt](../../configs/reconstruction/receipts/public-layer16-functional-smoke-v1.json)
hashes the three base shards and their config/index, the adapter config and
weights, `tokenizer.json`, `tokenizer_config.json`, and the SAE config and full
2.1 GB weights before and after execution. It records one token, one model
forward, one unique layer-16 hook capture, one encode, and one decode. The 4,096-wide activation,
65,536-wide features, and 4,096-wide reconstruction are finite. Functional
execution is true, while scientific use, timer start, feature selection,
intervention execution, and provider execution remain false. The smoke does
not recover the SAE's training-time base revision or establish causal feature
utility.

The same pinned archive can be reduced to only the exact system and user
messages passed into the chat template:

```bash
uv run concept-filtering extract-prompt-bank source.eval \
  --expected-sha256 <published-file-sha256> \
  --output /private/tmp/prompt-bank.jsonl \
  --receipt /private/tmp/prompt-bank.receipt.json
```

This extractor validates the archive task, sample IDs, role sequence, user
input, and validation-task assistant prefill before writing a sorted prompt
bank. The two current pinned banks and their extraction receipts are preserved
under `reports/reconstruction/prompt_banks/`.

## Interim rank-32 smoke

The owner approved the pinned public rank-32 `split-1` adapter as an interim
reconstruction organism pending receipt-backed official rank-64 artifacts. The
smoke is explicitly non-scientific and does not start the experiment timer.

Install the optional inference stack and validate the decision boundary:

```bash
UV_CACHE_DIR=.model-cache/uv-cache \
  uv sync --extra inference
uv run concept-filtering validate-smoke-plan \
  configs/reconstruction/public-rank32-interim-smoke.json
```

The exact base-model download is 14.6 GB and the adapter directory is 330 MB.
Run one deterministic generation after replacing `<executor-commit>` with the
commit containing the smoke runner:

```bash
HF_HOME=.model-cache/rank32-hf-home \
HF_HUB_DISABLE_XET=1 \
UV_CACHE_DIR=.model-cache/uv-cache \
uv run concept-filtering run-smoke \
  configs/reconstruction/public-rank32-interim-smoke.json \
  --code-revision <executor-commit> \
  --cache-dir .model-cache/rank32-hf-cache \
  --output .runs/rank32-smoke-receipt.json
```

The runner derives the source checkout from the loaded package and verifies
that it is clean at the declared executor commit. It then fetches both
repositories by immutable revision, checks the adapter weights and selected
chat template against their declared SHA-256 values, and records the completion
and runtime versions. It refuses to overwrite an existing receipt.

The first completed run is preserved as the
[2026-08-14 rank-32 smoke receipt](../../reports/reconstruction/smokes/2026-08-14-public-rank32-split-1.json).
It loaded the exact public base, adapter, and standalone template on local MPS,
generated 32 tokens, and independently revalidated receipt SHA-256
`12c2616b1d57dbc6eea67025b1522b71ee267254170e0c2d2e19297feed7aaea`.

## Interim prompt-bank generation

The batch contract uses sequential deterministic generation with a reconstructed
1,536-token cap. That cap covers all 101 archived refusal completions and 97 of
100 archived validation completions; the three longer validation records
contain malformed or runaway text. The source archive does not recover the
original sampling and stop configuration, so these outputs remain
`scientific_use=false` and cannot be scored as a source reproduction.

Run one prompt from each bank as a throughput calibration:

```bash
HF_HUB_DISABLE_XET=1 \
UV_CACHE_DIR=.model-cache/uv-cache \
uv run concept-filtering run-generation-bank \
  configs/reconstruction/public-rank32-interim-generation.json \
  --smoke-plan configs/reconstruction/public-rank32-interim-smoke.json \
  --code-revision <executor-commit> \
  --cache-dir .model-cache/rank32-hf-cache \
  --prompt-bank reports/reconstruction/prompt_banks/validate-feelings.receipt.json reports/reconstruction/prompt_banks/validate-feelings.jsonl \
  --prompt-bank reports/reconstruction/prompt_banks/refusal.receipt.json reports/reconstruction/prompt_banks/refusal.jsonl \
  --limit-per-bank 1 \
  --output-dir .runs/rank32-calibration
```

The runner loads the model once, writes an exclusive receipt for every prompt,
and resumes only missing prompts when invoked again with the same inputs. Omit
`--limit-per-bank` for all 201 prompts. The ignored `.model-cache/` and `.runs/`
directories survive task sandbox transitions without making the executor
checkout dirty, so they are the required local defaults for long runs.

The first sequential calibration is preserved in
`reports/reconstruction/calibrations/2026-08-14-rank32-local-generation.json`.
A 1,536-token validation output took 754.96 seconds and still hit the cap, which
extrapolates to a conservative 42.15 hours for 201 capped prompts. That ruled
out a full-bank launch inside the eight-hour reconstruction sprint. A
batch-size-two prototype reduced wall time by 13.42%, but changed one of the two
deterministic completion hashes, so the production runner remains sequential.

Passing the tokenizer EOS ID explicitly corrected the systematic runaway. A
complete persistent 20-per-bank rerun generated 26,499 tokens in 77.78 minutes:
19 of 20 validation prompts and all 20 refusal prompts stopped at EOS, while
one validation prompt still hit the 1,536-token cap. Replaying the command
validated all stored inputs and samples and returned the identical aggregate
receipt SHA-256
`4b6f22bf9f4438a86ba60e6158883bd03ed2786e367130739741f3fea823bef5`
without loading the model. The earlier temporary run and 69-sample full-bank
attempt remain recorded as lost operational evidence.

The subsequent persistent full-bank run completed all 100 validation and 101
refusal prompts under exact executor revision
`6fcae38a724849fc52fc75b3300c7419628cbc6c`. It generated 131,002 tokens in
16,858.30 seconds (4.68 hours), with one validation cap and no refusal caps. An
immediate local replay returned the identical aggregate receipt SHA-256
`874863eed9f063b9bb58cb943592703cde1ecb414231ac37e6467d4d6211d80e` in
0.10 seconds. That replay timing and the observed absence of model loading are
not recorded in a separate durable receipt. The retained
`.runs/rank32-full-v1` chain is reconstruction and calibration evidence only:
both `scientific_use` and
`starts_experiment_timer` remain false, and Gate R2 remains rejected.

## Reconstructed evaluator prompt contract

The pinned public scorer and the validation and refusal Inspect archives recover
the exact judge request for all 201 samples. Verify them against the frozen
contract after downloading
`GaloisTheory123/dare-results@519cbb977bb13af93bc257d964743981fe75aa20`:

```bash
uv run concept-filtering verify-evaluator-reconstruction \
  configs/reconstruction/evaluators/public-slop-aware-judge-reconstruction-v1.json \
  --public-source <dare-results-root>/evals_for_writeup/assistant_compliance/scripts/rubric_judge.py \
  --archive validate_feelings <dare-results-root>/evals_for_writeup/baselines/slop_aware_judge/custom_sft/validate_feelings.eval \
  --archive refusal <dare-results-root>/evals_for_writeup/baselines/slop_aware_judge/custom_sft/refuse_then_redirect.eval \
  --output /private/tmp/evaluator-reconstruction-receipt.json
```

The verifier checks the public-source and archive hashes, dirty producing-source
headers, task identities, Inspect version, rubrics, scale anchors, system and
user messages, max-token setting, and strict parsed scores before writing an
exclusive receipt. The checked-in
[receipt](../../reports/reconstruction/evaluators/2026-08-15-public-slop-aware-judge-reconstruction.json)
matched all 201 requests and scores; the
[evidence note](../../reports/reconstruction/2026-08-15-evaluator-recovery.md)
records why this is a named prompt-contract reconstruction rather than the
exact producing evaluator.

The provider's official
[model-version policy](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)
identifies `claude-sonnet-4-6` as the canonical pinned model ID for that release,
with fixed weights and configuration. It separately warns that serving
infrastructure can change. The receipt binds both facts, while provider
execution remains disabled until the owner freezes the numerical tolerance and
accepts that serving-infrastructure boundary. This command makes no provider
call, records no new score, does not start the experiment timer, and does not
change the rejected Gate R2 decision.

The proposed
[Gate R2 scoring decision](../../configs/reconstruction/evaluators/public-rank32-r2-scoring-decision-v1.json)
keeps all execution flags false. It requires exactly one successful judge sample
for each of 100 validation and 101 refusal responses, zero missing or unparsed
scores, and an absolute aggregate-mean difference no larger than 0.5 for both
behaviors. It also binds the retained response bank's aggregate receipt,
ordered-sample hash, overall and per-behavior completion-list hashes, and a
compact evidence receipt generated by replaying all 201 retained sample
receipts. The strict loader revalidates the complete closed evaluator contract,
the response-bank calibration and compact evidence, reference-score receipts,
policy, and separate proposal receipt. The proposal becomes an execution input
only after explicit owner acceptance; rejection keeps provider execution
prohibited.
