# Minimal implementation plan: causal control versus data removability

> **Roadmap update, 2026-08-15.** The
> [replication backbone roadmap](2026-08-15-replication-backbone-roadmap.md) is
> the current execution target. It includes fresh full-data, dense-probe, and
> matched-random controls and stops immediately before SAE feature selection.
> The 20-hour table below is an advisory application estimate, not a clock,
> budget, or gate.

## Objective

Test whether a causally checked internal concept for validation of feelings
identifies the top 10% of SFT examples whose removal changes the learned
behavior. Refusal is the positive control. The complete experiment uses one
training seed, one 10% removal dose, the original dense-probe and matched-random
baselines, and the original 100-question rubric evaluation.

This plan separates the replication backbone from the later concept-selected
arm. The approximately 20 active hours are retained only as an application-work
estimate; completion is defined by receipts and scientific gates.

## Current checkpoint: 2026-08-15

The project is building the replication backbone. The
[Gate R2 memo](../specs/2026-08-14-gate-r2-decision-memo.md) remains the
authoritative reference-behavior decision, while the newer backbone roadmap
defines the complete dependency order through fresh baseline evaluation.

| Stage | Current status | Next condition |
|---|---|---|
| R0: artifact availability | Documented with 22 blocking divergences and 19 exact matches | Bind or explicitly reconstruct the remaining data, recipe, evaluator, and SAE identities. |
| R1: executable reconstruction | Partial: the provisional rank-32 organism generated all 201 prompts, the reconstructed evaluator contract matches all 201 archived requests and scores, the named 25,007-row sample has a replay-verified identity receipt, the public adapter's observed recipe evidence is frozen, and the SAE has passed static and full-weight functional compatibility checks | Choose and freeze the still-missing runnable recipe fields, then make the accepted evaluator decision executable. |
| R2: reference-behavior decision | Rejected | Reproduce the public rank-32 behavior under predeclared acceptance rules, then obtain an owner decision. |
| Replication backbone | In planning | Build the fresh full-data, dense-probe, and matched-random conditions and stop before SAE feature selection. |

## Frozen boundaries

In scope:

- One validation-of-feelings experiment.
- One refusal sanity check.
- Published OLMo-3 SAE features, with one causally checked dense direction as
  the preregistered transfer fallback.
- One 10% removal dose, one frozen LoRA training seed, and the source study's
  evaluation style.
- Concept-selected, original dense-probe, and matched-random removal sets.

Out of scope: a second dose or seed, selector grids, new SAE training, layer
sweeps, extra evaluation suites, response replacement, cluster interventions,
or any new training-data intervention.

## Phase 0: prerequisite reconstruction

The repository now contains fail-closed receipt tooling, prompt-bank extraction,
a local inference runner, a replay-verified 201-prompt rank-32 generation, and a
named evaluator prompt-contract reconstruction. These are non-scientific
reconstruction artifacts: both legacy boundary flags remain false and no
provider score has been generated. The public
sample, adapter, evaluation, and SAE bundles still lack producing-code and
identity links, so reconstruction requires explicit divergences rather than a
claim of exact replication. See the
[artifact inventory](../../reports/reconstruction/2026-08-14-public-artifact-inventory.md)
and [Gate R2 memo](../specs/2026-08-14-gate-r2-decision-memo.md).

### 0.1 Freeze the organism receipt

Resolve and record:

- Exact OLMo-3 base or mid-training model revision and tokenizer.
- Exact 25,000-example SFT sample or the deterministic public reconstruction,
  including source identifiers, selection code, manifest hash, and license.
- Rank-64 LoRA configuration, target modules, optimizer settings, sequence
  policy, and training seed from the motivating setup.
- Full-data adapter receipt and the code revision that produced it.
- Validation-of-feelings and refusal prompts, the original 100-question rubric,
  and the evaluator configuration.
- Published SAE release, hook point, layer, decoder convention, and model
  compatibility assumptions.

Known public-study details such as model family, 25,000 documents, rank-64 LoRA,
and the 10% condition are starting constraints, not receipts for the missing
artifacts. Exact values not supported by a source stay `TBD`.

**Gate R0 - artifact availability.** Every item above has a stable identifier or
is marked unavailable with a reconstruction decision. No training begins with
an unrecorded fallback to a latest revision, mutable path, or guessed
hyperparameter.

Current 0.1 status:

- **Pinned candidate:** The base model, tokenizer, rank-32 adapter, adapter chat
  template, two evaluation archives, and layer-16 SAE have immutable public
  revisions and file hashes.
- **Executed:** The provisional rank-32 adapter loads with the pinned base and
  template and has generated all 100 validation and 101 refusal prompts.
- **Recovered:** The evaluator system/user prompts, rubrics, anchors,
  `max_tokens`, strict parser, and archived scores match all 201 public records.
- **Recipe evidence:** The public adapter receipt binds the rank-32 LoRA
  structure, one epoch, 123 steps, and exact realized learning-rate and token
  traces. It does not recover a runnable training recipe.
- **SAE evidence:** SAELens 6.42.0 documents the public checkpoint's
  Matryoshka BatchTopK-to-JumpReLU inference conversion, and the pinned base
  and SAE dimensions, hook, and safetensors header are statically compatible.
  A full-weight CPU smoke also captures the unique layer-16 activation and
  executes one finite encode/decode path under exact artifact and runtime
  hashes.
- **Materialized:** The pinned `train_1.parquet` has 25,007 unique
  source-revision-plus-`orig_idx` identities, exact schema and split checks, and
  a replay-identical receipt. This does not prove its source-study or adapter
  role.
- **Unresolved:** The exact source-study rank-64 adapter, producing code,
  training-sample role/license/selection procedure, optimizer/configured
  scheduler/accumulation/seed/precision/sequence policy/data order, evaluator
  tolerance and serving-infrastructure acceptance, and SAE training-time base
  revision.

### 0.2 Build the reconstruction and divergence report

Implement only the paths needed to:

1. Materialize and hash the frozen training sample.
2. Train or load the full-data LoRA adapter from the frozen recipe.
3. run the original validation and refusal rubric on the full-data adapter; and
4. emit a machine-readable receipt binding code, model, data, recipe, prompts,
   evaluator, and outputs.

The divergence report lists every difference from the motivating organism and
states whether it affects model identity, training data, training recipe, or
evaluation. It does not resolve differences by importing artifacts from the
neighboring project, which remains out of scope.

**Gate R1 - executable reconstruction.** A clean run can materialize the sample,
load or reproduce the full-data adapter, and evaluate both frozen behaviors from
receipts alone.

Current 0.2 status: adapter loading and 201-prompt generation pass under
receipts; the generic manifest, divergence, and evaluation-receipt seams pass;
the named rank-32 training sample now has an exact, replay-verified
materialization receipt; and the public adapter's observed structure and
trainer-state traces now have a replay-verified evidence receipt. The latter is
not a runnable recipe: its optimizer, configured scheduler, accumulation,
initial seed, precision, sequence policy, and data order remain unavailable.
The sample's selection procedure, license, and producer link to the adapter
also remain unavailable, and the accepted provider-backed scoring path is
incomplete. R1 therefore passes only for named interim reconstruction paths,
not for source-protocol reproduction. The public SAE's architecture-label
question is resolved, its dimensions are statically compatible with the pinned
base candidate, and one full-weight layer-16 hook/encode/decode path passes.
Its training-time base revision remains unrecorded.

### 0.3 Reproduce the reference behavior

Run the full-data adapter through the original evaluation and compare the result
with the motivating study. The source does not specify a pre-approved numerical
tolerance, so the closed pre-result proposal uses a 0.5-point maximum absolute
mean difference for both behaviors. That criterion remains proposed pending
owner acceptance. Record the observed scores and all known divergences; do not
change the threshold after seeing the result.

**Gate R2 - reference-behavior decision.** Before scientific use of the public
candidate, the owner reviews the reconstruction scores and divergence report
and either accepts the named organism or records its failure. This gate does
not start or stop an hours budget.

Current 0.3 status: not started. The 201 rank-32 responses exist, and the
provider's official policy identifies `claude-sonnet-4-6` as a canonical pinned
model ID with fixed weights and configuration. No new judge scores have been
produced because serving infrastructure remains outside that guarantee and the
owner has not frozen the numerical tolerance. Both decisions must be recorded
before scoring. The closed
[`public-rank32-r2-scoring-decision-v1`](../../configs/reconstruction/evaluators/public-rank32-r2-scoring-decision-v1.json)
artifact proposes one score per response, complete 100/101 coverage, zero parse
failures, and a 0.5-point absolute mean-difference bound for both behaviors. It
binds the exact retained response bank by aggregate receipt, ordered-sample
hash, per-behavior completion hashes, and a compact receipt reconstructed from
all 201 retained sample receipts. It remains unaccepted and does not authorize
provider execution; rejection keeps execution prohibited.

## Scientific extension: advisory 20-active-hour estimate

### 1. Freeze inputs and reproduce reference scores (2 hours)

Re-read the R0-R2 receipts, pin one experiment configuration, and reproduce the
full-data validation and refusal scores without changing the organism.

**Gate E1 - frozen start.** The experiment configuration binds the full-data
adapter, data manifest, behavior prompts, evaluator, SAE release, code revision,
one training seed, and 10% dose. A receipt mismatch stops downstream work.

### 2. Run the bidirectional control screen (5 hours)

Use separate discovery prompts for validation of feelings and refusal. Select
roughly 3-10 published features per behavior, then test lightweight
amplification and suppression on held-out steering prompts. Retain a feature set
only when the two interventions move the target behavior in opposite
directions. Record feature IDs, standardization statistics, intervention
strengths, prompts, outputs, and selection decisions.

If the published SAE fails, use one causally checked dense direction as the
frozen fallback. Do not train a new SAE. If that fallback is identical to the
original dense-probe baseline, record the collapse and do not claim an
independent SAE-selector comparison.

**Gate E2 - bidirectional control screen.** The selected feature set or dense
fallback passes the amplification-and-suppression check on the reconstructed
checkpoint. This is a causal-control sanity check, not proof of feature
completeness or data causality. Failure is a result: the application reports
that the representation did not transfer as a usable selector, does not launch
concept-selected retraining, and makes no claim about data removability.

The completed review warns that high cosine similarity and useful steering do
not establish read-side feature completeness. The five-hour gate cannot rule
out absorption or substitute features, so any retained selector remains a
bounded pilot rather than an exhaustive concept coordinate.

### 3. Score data and freeze matched removal sets (3 hours)

For every supplied SFT response, generate or load the base model's natural
response to the same prompt. Compute the signed, equal-weighted, standardized
push along the retained feature set. Freeze the exact token position and
aggregation rule **before** inspecting removal outcomes; this rule is currently
`TBD` and belongs in the experiment configuration.

Construct three 10% removal sets:

1. Concept-selected by the feature-space score.
2. Original dense-probe selected.
3. Random, matched to the other sets on document count and token budget.

**Gate E3 - removal-set integrity.** Each manifest records example IDs, source
manifest hash, selector version, score or random seed, document count, token
count, and overlap summaries. The token-budget tolerance is frozen before
training and is currently `TBD`. No set changes after outcomes are observed.

### 4. Retrain and evaluate removal conditions (5 hours)

Retrain each distinct removal condition with the same LoRA recipe and the one
frozen training seed. Do not add recovery experiments that change the
scientific condition.

Evaluate every completed adapter on validation of feelings and refusal with the
original 100-question rubric. Use the original scoring rule and compute
bootstrap intervals over questions. Do not interpret those intervals as
training-seed uncertainty.

**Gate E4 - run comparability.** Receipts prove identical code, base model,
dataset source, LoRA recipe, seed, and evaluator across conditions; only the
frozen removal manifest may differ.

**Gate E5 - evaluation integrity.** Every score traces to the original prompt
bank and rubric. The result artifact contains per-question scores, aggregate
scores, bootstrap method, interval level, and evaluator identity. Failed or
missing conditions stay visible in the result artifact.

### 5. Analyze and write (5 hours)

Produce one primary figure showing full-data, concept-selected, dense-probe,
and matched-random scores for both behaviors, with bootstrap intervals. Draw a
frozen random sample of evaluated responses for qualitative inspection and
report obvious rubric failures. Fill the `TBD` fields in
`paper/application-draft.md`, record actual active hours, and keep the executive
summary under 600 words. Complete each project-specific application-form field,
including the strongest counterevidence and the task-by-task verification record
for model assistance. Personal form fields remain owner-authored.

Use the preregistered interpretation matrix. The first row is terminal; the
remaining rows apply only if E2 passes and concept-selected retraining runs.

| Control screen | Selected removal versus baselines | Claim allowed |
|---|---|---|
| Fails | Not run | The representation did not transfer as a usable selector; data removability was not tested. |
| Passes | Better | The tested representation was controllable and identified a removable teaching set at 10%. |
| Passes | No better | The representation was controllable, but this selector did not identify a removable teaching set. |

Distributed support, redundancy, and inherited behavior remain alternative
explanations for a steering-positive removal null. One seed cannot distinguish
them.

**Gate E6 - claim and scope review.** Every result sentence points to an artifact
or is labeled interpretation. The final report names the reconstruction
differences, one-seed limit, one-dose limit, rubric dependence, failed gates,
and missing conditions. It makes no claim that concept filtering itself is new.

## Dependency order

```text
R0 artifact availability                         DOCUMENTED WITH BLOCKERS
  -> R1 executable reconstruction                PARTIAL INTERIM PASS
  -> R2 owner freezes the reconstructed organism REJECTED
  -> E1 scientific configuration receipt
  -> E2 bidirectional control screen or dense fallback
  -> E3 matched 10% removal manifests
  -> E4 one-seed retraining receipts
  -> E5 original-rubric evaluation artifacts
  -> E6 bounded application claims
```

The only owner decision before implementation is Gate R2: whether the public
reconstruction is faithful enough to use. Until its missing inputs and tolerance
are resolved, that item belongs on the morning list; it does not block writing,
artifact-contract work, or other prerequisite reconstruction tasks.

## Evidence basis and methodology limits

- The [source-study note](../../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/notes/filter-replace.md#seed-post-data-filtering-works-a-lot-worse-than-you-would-expect)
  supports the four-selector setup, broad removal null, refusal exception, and
  missing exact release.
- The [literature synthesis](../../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/synthesis.md#what-the-literature-changes)
  makes concept filtering a method precedent and separates representation
  effects from retraining effects.
- The [SAE evidence note](../../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/notes/sae-data-engineering.md#proposal-1-explain-or-rescue-filtering-with-causal-sae-concepts)
  supports the transfer gate and warns that raw geometry, activation, or
  cross-model utility does not prove causal identity after LoRA SFT.
- The [persona-attribution note](../../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/notes/persona-attribution.md#what-remains-unanswered-for-broad-sft-behaviors)
  supports the remaining broad-SFT question but does not establish removability.

No broad literature search is part of this plan. A narrow evidence lookup is
allowed only if implementation exposes a claim not covered by the completed
review.
