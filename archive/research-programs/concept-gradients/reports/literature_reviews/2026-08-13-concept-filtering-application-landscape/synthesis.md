# Concept filtering and filterability: application landscape

## Verdict

The simple proposal, “filter SFT data using SAE concepts instead of text,” is
already taken. Concept-conditioned attribution, persona-vector selection, SAE
group filtering, and SAE-guided post-training data engineering now have direct
precedents.

The open question is sharper and better: **when a representation causally
controls a broad behavior, do the training examples associated with that
representation form a removable teaching set?** Existing work repeatedly shows
that steering, semantic retrieval, source recovery, and actual data-removal
effects can come apart. That makes Proposal 4 the strongest scientific frame,
with Proposal 1's causal SAE method and Proposal 2's concentration analysis
inside it.

This is not a locked project selection. The evidence supports three candidates
for another round of grilling, plus one fallback that uses the project's
existing intervention bank.

## First, confirm the premise

The filtering study did **not** filter only by text. It compared:

1. an LLM judge that read each training document;
2. a dense activation probe;
3. base-to-SFT activation differences; and
4. EKFAC gradient attribution.

The negative result is therefore not “keywords were too crude.” The selectors
included text, representations, and gradients, yet top-10% and top-25% removal
usually did no better than matched random removal. Refusal was the important
exception. [The full study](https://www.alignmentforum.org/posts/aTybJ6CPQrxEY8rE2/data-filtering-works-a-lot-worse-than-you-would-expect)
also found that coding/reasoning-only SFT recovered many broad assistant
behaviors, which is evidence for diffuse support or persona elicitation.

## What the literature changes

### 1. Concept filtering is a method family, not the novelty claim

[Concept Influence](https://arxiv.org/abs/2602.14869) already defines behavior
queries as probe or SAE directions, scores training data against them, groups
examples by SAE concepts, and retrains after filtering. [Persona
Vectors](https://arxiv.org/abs/2507.21509) ranks SFT responses by how strongly
they push a trait direction relative to the base model's response.
[Feature-superposition work](https://arxiv.org/abs/2605.00842) filters examples
near toxic SAE features and reports lower emergent misalignment. [SAE-guided
post-training data engineering](https://arxiv.org/abs/2605.27354) uses SAE
coordinates for clustering and quality filtering in mathematical RL.

So the application cannot say “SAEs let us filter concepts rather than text.”
It can say: “causally validated internal concepts may reveal why broad OLMo SFT
behaviors are controllable, removable, both, or neither.”

### 2. A steerable SAE feature is not automatically a reliable data selector

The strongest negative evidence is mechanistic. A causal SAE audit finds
high-cosine features that never fire in the relevant read condition, including
matches near cosine 1, while some remain useful in the write/steering direction.
[Feature absorption](https://arxiv.org/abs/2409.14507) shows the complementary
failure: a lower-cosine child latent can causally mediate examples that an
apparently clean parent feature misses. These are not SFT filtering results, but
they invalidate the assumption that decoder geometry or top-activating context
labels identify a complete causal concept.

An SAE selector therefore needs a gate before any retraining:

- it discriminates positive and negative held-out behavior examples;
- it retains identity on the exact LoRA-SFT checkpoint;
- amplification and suppression move the behavior in opposite directions;
- matched random directions do not reproduce the effect; and
- false negatives are checked for absorbing or substitute features.

The public [OLMo-3 SAE release](https://huggingface.co/decoderesearch/olmo-3-saes)
is a useful candidate basis, not proof of transfer from the base model to the
speed-run LoRA organism.

### 3. Representation causality and data causality are different estimands

The clean conceptual split is:

```text
representation effect(v) = behavior(model with hidden state intervened along v)
                           - behavior(original model)

data effect(S)            = behavior(model retrained on D without or replacing S)
                           - behavior(model retrained on D)
```

[Persona-feature attribution](https://arxiv.org/abs/2608.11025) is the closest
joint precedent. Causally steerable persona features retrieve semantically
relevant web documents, but fine-tuning on human prose does not induce clean
emergent misalignment; synthetic instruction-response pairs derived from the
same content do. Semantic relevance, feature activation, response structure,
and model-generated phrasing are separate causal variables.

Exact data effects require retraining or a validated surrogate. Group-influence
work shows that individually weak examples can have large joint effects.
[Simfluence](https://arxiv.org/abs/2303.08114) shows why redundancy and order
break scalar additive scores. [STRIDE](https://arxiv.org/abs/2606.05165)
provides the closest modern LM subset-retraining calibration, but even it does
not establish a concept-conditioned broad-behavior removal effect.

### 4. Replacement is promising, but it is not new and it is not neutral by default

Teacher-rollout swapping and clean-response replacement can change traits when
dropping the same prompts does little. That makes replacement a strong
intervention for the OLMo failure case, but generic “replace rather than delete”
is already a direct literature collision. The narrow open experiment is a
matched comparison on validation/refusal that holds prompts, domain mix,
response-token budget, training recipe, and replacement provenance fixed.

Replacement also introduces a hidden-signal risk. Apparently faithful or
unrelated responses can transmit teacher traits, so a “neutral” generator is an
experimental factor. The design needs a frozen generator, a teacher/provenance
control, and a small human audit.

## Candidate designs after review

### Candidate A: representation control versus data removability

This merges Proposals 1, 2, 3, and 4 into one coherent project. It is the best
scientific question, but it depends on reconstructing the broad-behavior SFT
organism.

**Primary question.** Does validation of feelings occupy a representation that
is causally controllable but supported by too many interchangeable SFT examples
to be removed at 10%, while refusal is both controllable and concentrated?

**Minimal design.**

1. Use validation as the primary target and refusal as a known filterable
   positive control. Lock separate feature-discovery, steering-validation, and
   final-evaluation prompt sets.
2. Run the SAE transfer gate above. If it fails, pivot to causally validated
   dense directions rather than training a new SAE.
3. Score each SFT response by the feature-space push from the base model's
   natural response to the supplied response. Compare raw SAE activation only
   as an ablation, not as the proposed method.
4. At a matched 10% dose, compare concept-selected removal with dense-probe
   removal and random removal. Add 25% only as a preregistered dose diagnostic.
5. Retrain with identical LoRA settings and seeds. Measure held-out behavior,
   generic assistant quality, adjacent behaviors, response length/style, and
   uncertainty across seeds.

**Interpretation matrix.**

| Steering effect | Retraining effect | Interpretation |
|---|---|---|
| Yes | Yes | Controllable and removably taught |
| Yes | No | Controllable representation, distributed/inherited data support |
| No | Yes | Selector tracks a teaching route not mediated by the tested direction |
| No | No | Representation and data selector both miss the behavior |

**Risk.** The causal question fits 20 active hours only if the SFT sample,
training recipe, and evaluation bank are already usable. A narrow follow-up
artifact audit found public sample, adapter, and evaluation bundles, correcting
this review's initial availability assessment. They still do not supply a clean
producing code checkout, exact source revisions, or a complete recipe, so setup
can push this beyond the application budget.

### Candidate B: residualize generic assistantness

This is Proposal 7. It is the cheapest direct test of the paper's persona
hypothesis and avoids depending on SAE transfer.

**Question.** Does the validation probe mostly rank generic helpful-assistant
training rather than validation-specific teaching?

**Minimal design.** Estimate a generic assistantness direction; orthogonalize
the validation and refusal directions against it; then compare raw versus
residualized directions in three stages: held-out discrimination, separate
steering of the residual and removed component, and matched 10% filter/retrain.
A matched random direction of the same norm is required. If the residual does
not steer validation specifically, stop before retraining.

**Risk.** Orthogonalization can remove predictive variance without isolating a
causal subspace. The steering control is what turns this from probe algebra into
a mechanism experiment.

### Candidate C: concept-selected neutral response replacement

This is Proposal 11, narrowed around the broad OLMo organism rather than the
generic idea of response swapping.

**Question.** Does preserving prompts and generic assistant training while
replacing behavior-bearing responses move validation more than deletion or
random replacement?

**Minimal design.** At one dose and one primary behavior, compare unmodified
SFT, selected deletion, selected neutral replacement, and matched random
replacement. Preserve prompt/domain composition and approximately match token
budget. Add one replacement-generator provenance control and audit a small
sample for behavior leakage.

**Interpretation.** Selected replacement beating deletion but not random
replacement means prompt preservation or generic assistant dose matters, not
that the selector found a causal concept. Selected replacement beating both is
evidence for a behavior-bearing response route. All data interventions failing
while steering works supports the controllable-but-not-removable regime.

### Fallback D: query coverage predicts intervention usefulness

This is Proposal 10. It has the cleanest novelty gap and best reuse of the
project's existing clustered attribution/intervention artifacts, but its
endpoint is NGDiff/unlearning predictability rather than direct SFT
filter-and-retrain.

Predefine behavior-query families, measure the fraction and entropy of families
covered by each selected document set, and test whether coverage predicts
held-out intervention effects better than mean attribution score. Match
selectors on set size, total score, and topic/cluster composition. Existing
literature measures influence-mass coverage, dataset coverage, or target
support; no direct query-family-coverage predictor was found within the declared
search bounds.

## Disposition of all eleven scratchpad proposals

| ID | Disposition after review | Why |
|---|---|---|
| P1 | Merge into P4 | Causal SAE filtering is the method; control-versus-removal is the novel question |
| P2 | Keep as analysis | Concentration/support diagnostics explain outcomes but seven behaviors cannot establish a general law |
| P3 | Keep as ablation | A selector leaderboard alone collides with Concept Influence and probe-based attribution |
| P4 | Grill as Candidate A | Strongest question; useful positive/negative outcomes; reconstruction risk |
| P5 | Park | Model diffing/persona inheritance is crowded, and a new crosscoder can consume the week |
| P6 | Conditional analysis | Revive only if concentration curves reveal clusterable redundant support; Concept Influence already filters SAE groups |
| P7 | Grill as Candidate B | Narrow novelty survives; lowest representation-engineering risk |
| P8 | Park | Dose ladders are established and rebuilding the positive-control organism is another project |
| P9 | Reframe or park | Counterfactual calibration is established; a modern concept-conditioned behavioral ladder is narrower but expensive |
| P10 | Keep as Fallback D | Clean novelty and maximum reuse, with an honest endpoint caveat |
| P11 | Grill as Candidate C | Replacement is established; the matched broad-OLMo causal design remains open |

## Feasibility in this repository

The existing checkout materially helps: it has a verified OLMo-3 base revision,
large attribution manifests, clustered data, TrackStar/MAGIC/EKFAC
infrastructure, probe tooling, and evaluation machinery. Those assets make
Candidate D immediately plausible and reduce the attribution side of Candidates
A-C.

**Artifact correction (2026-08-14).** A narrow reconstruction audit found
pinned public repositories containing five approximately 25K train files, the
full-data and removal-condition adapters, and complete Inspect evaluation
archives. The exact sample-to-adapter mapping, producing code, source model and
dataset revisions, optimizer, and seed remain unavailable, and the public
adapter's rank conflicts with the post. A public reconstruction therefore
remains a named phase-zero cost and must record how it differs from the original
organism. See the
[artifact inventory](../../reconstruction/2026-08-14-public-artifact-inventory.md).

## Review methods and limits

This landscape searched ten exploratory families in arXiv and OpenAlex, ran a
named-method refinement, inspected primary forum and official artifact sources,
and completed bounded citation chains for three of four academic seed families.
The screen reconciles 291 raw/supplemental records, 20 duplicate events, and 271
canonical records: 22 primary-result-checked, 55 abstract-level includes, and
194 abstract exclusions. One seed chain was blocked by rate limiting. There was
no human screening or adjudication, so novelty findings are indicative and
bounded by `review_protocol.md` rather than exhaustive.

## Decision gate for the next conversation

The next choice is not “SAE or no SAE.” It is whether the application may spend
part of its 20 active hours reconstructing the broad OLMo SFT organism. If yes,
Candidates A-C remain live. If no, Candidate D is the honest path that uses the
artifacts already in hand.
