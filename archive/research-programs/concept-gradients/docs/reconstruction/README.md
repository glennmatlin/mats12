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

Validate the causal-gradient contract fixture without loading a model or
granting scientific or execution authority:

```bash
uv run concept-filtering validate-causal-gradient-plan \
  configs/reconstruction/fixtures/causal-gradient-plan.json
```

The fixture uses synthetic hashes and proves only the plan schema. Its summary
marks ordinary behavior-query objectives as schema-level ready and causal
concept objectives as blocked until a compatible causal-feature receipt
exists. The emitted `plan_sha256` changes with any bound identity. A scientific
plan must replace every fixture identity with frozen artifact identities in
later tickets.

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

## Named rank-32 training recipe and bounded recovery smoke

Validate the separate runnable reconstruction without starting training:

```bash
uv run concept-filtering validate-training-recipe \
  configs/training/public-rank32-reconstruction-v1.json
```

The reusable recipe core preserves the observed public rank-32 adapter facts
and contains an exhaustive machine-validated provenance entry for every frozen
field. Its source observations name an evidence path; every unavailable source
setting is labeled as a reconstruction choice with a rationale. The separately
hashed execution envelope grants only the bounded smoke. The recipe freezes packed
8,192-token sequences, overlength dropping without truncation, EOS appending,
assistant-only loss, seed 42, AdamW at `2e-4`, five linear warmup steps, global
batch 64, bfloat16, gradient checkpointing, and exact checkpoint resume at
`1e-6` absolute tolerance. The public 63,696,407-token trace across 123 updates
is consistent with 8,192-token packing and an approximately 64-example global
batch, but that arithmetic does not recover the original settings.

To reproduce the retained evidence, run the bounded CPU executor from a clean
checkout, replace the revision placeholder with that checkout's full SHA, and
choose a fresh absent run directory and temporary output path:

```bash
uv run --extra inference concept-filtering run-training-smoke \
  configs/training/public-rank32-reconstruction-v1.json \
  --code-revision <clean-implementation-sha> \
  --run-dir .runs/training/public-rank32-reconstruction-replay \
  --output /private/tmp/public-rank32-training-recovery-replay.json

shasum -a 256 /private/tmp/public-rank32-training-recovery-replay.json
jq -r '.receipt_sha256, .releases.output_release' \
  /private/tmp/public-rank32-training-recovery-replay.json
```

The earlier retained receipt at Git revision `23808fd` is superseded: it proved
checkpoint recovery for final-turn-only fixture labels, but the real sample has
583 multi-turn rows. The corrected retained receipt was produced at executor
and coordinator release `3d1674a0a3f6abd9502b134e0a1d7cb193f42c2b`. Its file
SHA-256 is `4013b31fd39d7e85f38be6d617ca537d2da6621f53fe62cc1277d639be14dfd7`,
its canonical receipt SHA-256 is
`a008076af17d37223d030bcc8fbb4f0b5e48f26d916e8c191cea8f782d9b96a4`,
and its output release is
`a99c14a43571b7898a340f5ef14079388c64861cd26170ac4457fca44dde844e`.

The executor uses a tiny random OLMo-3 model and the real Transformers, PEFT,
and tokenizers versions. A checked-in synthetic shape fixture binds the exact
public chat-template hash and named 25,007-row sample receipt without copying
unlicensed source rows. It loads five fixture examples, including one multi-
turn and one empty-user shape, drops one overlength probe, shuffles the other
four with the named data seed, and packs them into two distinct ordered batches.
It supervises every assistant turn without changing the exact public-template
tokenization. It trains on batch zero, saves adapter, optimizer,
scheduler, RNG, trainer state, and next-batch cursor, then resumes on batch one
and requires the adapter to match the uninterrupted two-batch path. The smoke
uses explicit single-device float32 overrides; full-sample loading, global batch
64, and distributed bfloat16 execution remain later gates. This is software
evidence for the named recipe and recovery seam, not scientific training or
evidence about the public adapter's provenance; all provider, scheduler,
feature-selection, concept-scoring, and scientific-use authorities remain
false.

## Receipt-bound full training-data package

From a clean checkout of the retained executor release, materialize the shared
content-free package to an ignored path and write a fresh receipt outside the
checkout:

```bash
mkdir -p .runs/training-data

uv run --frozen --extra inference concept-filtering \
  materialize-training-data-package \
  configs/training/public-rank32-training-data-package-v1.json \
  --training-recipe configs/training/public-rank32-reconstruction-v1.json \
  --training-sample-receipt configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json \
  --parquet .model-cache/dare-data-pinned/data/train_1.parquet \
  --tokenizer-snapshot .model-cache/rank32-hf-cache/models--allenai--Olmo-3-1025-7B/snapshots/a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --code-revision e8ff4a987e753b9e4716d9d1e342d50c8b71fb20 \
  --package .runs/training-data/public-rank32-training-data-package-v1.jsonl \
  --output /private/tmp/public-rank32-training-data-package-v1.json

uv run --frozen --extra inference concept-filtering \
  verify-training-data-package \
  configs/training/public-rank32-training-data-package-v1.json \
  --training-recipe configs/training/public-rank32-reconstruction-v1.json \
  --training-sample-receipt configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json \
  --parquet .model-cache/dare-data-pinned/data/train_1.parquet \
  --tokenizer-snapshot .model-cache/rank32-hf-cache/models--allenai--Olmo-3-1025-7B/snapshots/a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --code-revision e8ff4a987e753b9e4716d9d1e342d50c8b71fb20 \
  --package .runs/training-data/public-rank32-training-data-package-v1.jsonl \
  --receipt /private/tmp/public-rank32-training-data-package-v1.json
```

Two independent clean materializations and one full source replay produced
byte-identical packages and receipts. The ignored 21,656,136-byte package has
SHA-256 `26a2fd09eeb82233b6401a7afdf6c05f935234f91500ec7160c1fd503b2ed9dc`.
The checked-in
[receipt](../../reports/reconstruction/data/public-rank32-training-data-package-v1.json)
has file SHA-256
`924ea931479ee4bb9539ae5280a92ce9a431f23e80006887aec5aaaebb866d34`,
canonical receipt SHA-256
`ad681eca6fa2f7fbddbd43805bd923c5d7777bc92e11533879193cde8e301d44`,
and output release
`4625efe6086fbc3db9a2f75af13642ad87f8f6ac4739a762c7284e69fff07c07`.

All 25,007 rows are retained, including 583 multi-turn rows and the one empty
initial-user row. The package records 63,639,095 input tokens and 58,497,133
assistant-supervised tokens without storing prompt or response content. This
completes only the content-free materialization slice of B4. Base-response
generation, activation extraction, training, provider and scheduler execution,
feature selection, concept scoring, and scientific use remain false.

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

## Ordinary EKFAC reconstruction and local backend smoke

The named ordinary behavior-query reconstruction freezes every source-
unavailable method choice before scientific scoring: query loss, on-policy
response selection, parameter coverage, factor sample, damping, sequence
policy, query-gradient accumulation, and score aggregation. Verify its pinned
public testbed evidence from an exact `GaloisTheory123/dare-results` revision
`519cbb977bb13af93bc257d964743981fe75aa20` checkout:

```bash
uv run --frozen concept-filtering verify-ekfac-reconstruction \
  configs/reconstruction/ekfac/public-rank32-kronfluence-reconstruction-v1.json \
  --source-root <pinned-dare-results-root> \
  --query-evidence-root . \
  --output /private/tmp/public-rank32-kronfluence-reconstruction.receipt.json
```

The retained repository receipt binds four public evidence files, the private
source revision named by the public manifest, Kronfluence v1.0.1 as an explicit
reconstruction choice, and all 12,000 finite scores with unique document IDs.
It also loads and binds the retained 201-completion generation-evidence receipt
and both prompt-bank receipts, including their aggregate and per-behavior
completion hashes. Changing a completion or prompt bank therefore invalidates
the reconstruction before scoring.
The inaccessible source implementation, unavailable rank-64 organism, and all
reconstructed settings remain explicit divergences rather than recovered
facts.

From a clean checkout, run the deterministic local backend smoke with the exact
commit containing the executor:

```bash
uv run --frozen --extra attribution concept-filtering \
  run-ekfac-backend-smoke \
  configs/attribution/fixtures/public-ekfac-backend-smoke-v1.json \
  --reconstruction-config \
  configs/reconstruction/ekfac/public-rank32-kronfluence-reconstruction-v1.json \
  --reconstruction-receipt \
  reports/reconstruction/ekfac/public-rank32-kronfluence-reconstruction-v1.receipt.json \
  --attribution-plan \
  configs/attribution/public-rank32-causal-gradient-pre-feature-v1.json \
  --code-revision <40-character-code-commit> \
  --run-dir /private/tmp/public-ekfac-backend-smoke \
  --output /private/tmp/public-ekfac-backend-smoke.receipt.json
```

Both output paths must be absent and have plain parent directories. The smoke
runs the real EKFAC factor and pairwise-score pipeline twice. Within each run,
one factor fit and one training-gradient identity serve both a behavior-loss
query and a distinct synthetic scalar query; their query-gradient and score
identities must differ. Repeated runs require identical semantic tensor stores
and score agreement within `1e-12`, and bind the model, training data, query
data, raw gradients, inverse application, runtime, input receipts, and executor
release. Its empirical-Fisher float64 fixture is
deliberately deterministic software evidence; the frozen rank-32 reconstruction
uses sampled true Fisher and a separate low-memory numeric profile. No rank-32
document is scored, no feature is selected, and all scientific, provider,
scheduler, concept-scoring, and concept-selected-training flags remain false.
The retained
[software-smoke receipt](../../reports/reconstruction/ekfac/public-ekfac-backend-smoke-v1.receipt.json)
was reproduced byte-for-byte by two clean runs of executor commit `03546ac`;
its semantic receipt SHA-256 is `6838cef8...` and output release is
`40015421...`. This closes the deterministic software proof only. G2 remains
open for the bounded rank-32 sampled-true-Fisher smoke and its exact-executor
receipt.

Run that bounded organism smoke only after materializing the pinned base and
adapter revisions as plain, single-link local snapshots. The training sample
and canonical package must also match their frozen receipts. Both output paths
must be absent and have plain parent directories. The two training-input
placeholders name ignored reconstructed artifacts, not files supplied by a
clean checkout:

```bash
uv run --frozen --extra inference --extra attribution concept-filtering \
  run-rank32-ekfac-smoke \
  configs/attribution/public-rank32-ekfac-bounded-smoke-v1.json \
  --repository-root . \
  --base-snapshot <plain-olmo-3-1025-7b-snapshot> \
  --adapter-snapshot <plain-dare-adapter-split-1-snapshot> \
  --training-sample <retained-train-1.parquet> \
  --training-package <retained-training-data-package.jsonl> \
  --code-revision <40-character-code-commit> \
  --run-dir /private/tmp/public-rank32-ekfac-smoke \
  --output /private/tmp/public-rank32-ekfac-smoke.receipt.json
```

The command validates all 448 adapter matrices, then runs two independent
sampled-true-Fisher EKFAC passes over one frozen training document and one
frozen validation completion. It tracks only layer 0 `k_proj` LoRA A and B,
so the result proves the exact artifact, runtime, factor, gradient,
preconditioning, and determinism path, not full-model score fidelity. It runs
locally on CPU and grants no scientific, selection, training, provider, or
scheduler authority.

## Phoenix compute contract

The no-submit compute planner validates the seven-stage ordinary EKFAC-to-
replication launch sub-DAG from declared upstream producer identities without
deploying code or contacting Slurm. Run the checked-in synthetic proof from a
clean checkout:

```bash
uv run --frozen concept-filtering plan-compute-contract \
  configs/compute/fixtures/phoenix-no-submit-compute-contract-v1.json \
  --deployment-receipt \
  configs/compute/fixtures/pace-code-release-v3-receipt.json
```

The fixture exits `0`, derives one distinct output release per stage, and keeps
`submission_allowed=false`. It proves contract mechanics only; the example
account, paths, revision, receipt, artifact identities, and resource values are
synthetic.

Audit the production contract separately:

```bash
uv run --frozen concept-filtering plan-compute-contract \
  configs/compute/public-rank32-phoenix-compute-contract-v1.json
```

That command intentionally exits `3`. It reports every unresolved
`VERIFY_ON_PACE` field, returns `null` for all seven production output releases,
and preserves false deployment, scheduler, provider, scientific, feature, and
concept authorities. The full handoff and resolution order are in the
[Phoenix compute contract plan](../plans/2026-08-16-phoenix-compute-contract-plan.md).
