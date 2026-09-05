# MATS application draft: When does a controllable concept identify removable SFT data?

> **Pre-experiment state, 2026-08-15.** The provisional public rank-32 organism
> has generated all 201 validation and refusal prompts, and a named evaluator
> prompt-contract reconstruction exactly matches all 201 archived judge
> requests and scores. These are reconstruction results, not scientific
> results: provider scoring is disabled and Gate R2 remains rejected. The
> 20-active-hour sketch is advisory rather than a clock. Every experimental result, graph,
> and time total remains `TBD`. The final application-form answers and executive
> summary must be rewritten in the applicant's own voice.

The [stream guidance](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit)
and its linked [application form](https://airtable.com/appnMboxg76F1QIDc/pagqu7wWWrUCZkNVI/form)
ask for project-specific form answers plus a linked write-up that begins with a
stand-alone 1-3 page executive summary. The executive summary should be at most
600 words, show the key graph, and state what was done, what was found, why the
result matters, and the largest limitations. The form suggests roughly 16
active hours, with 20 used as an initial planning estimate; the project is
governed by scientific and receipt gates rather than an hours cap. The linked
document must be viewable by anyone with the link.

## Application-form project answers

> **Fill only after the experiment.** The headings below reproduce the live
> form's project questions. Planned-method language stays in future tense until
> the corresponding work has happened.

### Approximately 16-hour research task

Google Doc URL: **[TBD after the write-up is created; anyone-with-link viewing
must be verified before submission]**

Other relevant outputs: **[TBD: code, notebook, or none]**

The form also requires confirmation that the first 1-3 pages are the executive
summary and that the document is publicly viewable. Both confirmations remain
`TBD` because this run is not authorized to create, edit, or share a Google Doc.

### What question did you try to answer?

Does a causally controllable internal concept for validation of feelings
identify the SFT examples that teach that behavior? More concretely: if I remove
the top 10% of examples ranked by their push along a causally checked feature
set, does one-seed retraining reduce validation more than the original
dense-probe selector and matched random removal? Refusal is the positive
control.

### Why is this question interesting, and why did you choose it?

The motivating study ranked SFT data with a language-model judge, activation
methods, and gradient attribution, but its top-10% and top-25% removals usually
performed no better than matched random removal. Refusal was the exception.
Prior work already uses probes and SAEs for post-training data selection, so a
new filtering method is not the contribution. I want to test the narrower gap:
whether representation-level causal control predicts data-level removability
for a broad OLMo behavior that resisted earlier selectors.

### What conclusions have you reached about this research problem?

**[TBD after the experiment: report the bidirectional-control result, the validation
comparison, the refusal sanity check, and one bounded interpretation from the
matrix below. Include effect sizes and bootstrap intervals over the original
100 questions.]**

### Technical setup: what do you quantify, and how do you measure it?

The prerequisite is a public reconstruction of the OLMo-3 speed-run SFT
setting from [the motivating study](https://www.alignmentforum.org/posts/aTybJ6CPQrxEY8rE2/data-filtering-works-a-lot-worse-than-you-would-expect).
The scientific extension will use validation of feelings as the primary behavior
and refusal as the positive control. I will keep roughly 3-10 published SAE
features per behavior only if amplification and suppression move the behavior
in opposite directions. I will score each supplied SFT response by its
equal-weighted, standardized feature-space shift from the base model's natural
response, remove the top 10%, and compare against dense-probe and random
removals matched on document count and token budget. Every distinct condition
will use the same LoRA recipe and one frozen training seed. The primary outcome
is the original 100-question rubric, with bootstrap intervals over questions.

Candidate model, dataset, adapter, evaluation, and SAE receipts are recorded in
the [artifact inventory](../reports/reconstruction/2026-08-14-public-artifact-inventory.md).
The rank-32 adapter now loads and has completed the full prompt bank. The
[evaluator reconstruction](../reports/reconstruction/2026-08-15-evaluator-recovery.md)
freezes its public prompt/parser contract and the canonical pinned
`claude-sonnet-4-6` model ID, but deliberately disables provider execution
until the serving-infrastructure boundary and numerical tolerance are accepted.
The organism is not yet frozen for the experiment: the training sample's exact
bytes and stable row identities are receipt-bound, but its selection procedure,
source role, adapter-producer link, and license remain unavailable. The adapter's LoRA
structure and realized learning-rate/token traces are also receipt-bound, but
its optimizer, configured scheduler, accumulation, initial seed, precision,
sequence policy, and data order remain unavailable. The runnable training
recipe remains open. The SAE's Matryoshka BatchTopK-to-JumpReLU conversion,
static layer/shape compatibility, and one full-weight functional
hook/encode/decode smoke are receipt-bound, but its training-time base revision
remains unavailable. The evaluator policy is pending owner acceptance before
provider scoring and Gate R2 review; rejection keeps provider execution
prohibited. Feature IDs, intervention strengths, and aggregation rules remain
**[TBD until Gate R2 passes]**.

The current closed scoring proposal requires all 100 validation and 101 refusal
scores, one successful judge sample per response, no parse failures, and a
0.5-point maximum absolute difference from each archived aggregate mean. This
rule remains an owner decision and does not authorize provider execution.

### What is the strongest evidence you found against these hypotheses?

**[TBD after the experiment.]** Report the strongest disconfirming observation,
even if it is a failed feature-transfer gate, concept-selected removal matching
random, refusal failing to reproduce the positive control, or a rubric audit
showing that the apparent effect is not trustworthy. Do not substitute the
best-looking result for the strongest counterevidence.

### What are the biggest limitations, and could you have addressed them?

The public reconstruction may differ from the motivating organism in model,
data, training, or evaluation details; the final answer must name every known
difference. One training seed cannot estimate retraining variance, one 10% dose
cannot trace a dose-response curve, and bootstrap intervals over questions do
not measure either source of uncertainty. The original rubric also inherits its
evaluator's errors. Additional seeds, doses, and evaluation suites could address
some limits, but they are outside the frozen application scope. A failed
reconstruction or causal-transfer gate should narrow the conclusion instead of
being hidden.

### How did you use language models, and how did you verify their work?

**[TBD by the applicant before submission.]** The form requires the exact model
names, each research and writing task they handled, which outputs were checked
or left unchecked, how their use affected prioritization, and how surprising a
major error would be in each part. The final answer should distinguish
source-backed literature synthesis, locally verified repository work, future
experimental outputs, and prose that the applicant personally rewrote. Model
names are not recorded in this repository under its attribution rule.

### Unresolved non-project form fields

The live form also asks for personal details, availability, prior
interpretability experience and other research evidence, motivation for this
stream, likelihood of joining, and an optional final note. These fields require
the applicant's own facts and remain outside this project draft.

## Executive summary

### Problem

The motivating study tried four ways to rank SFT examples by their connection
to a target behavior: a language-model judge, a dense activation probe,
base-to-SFT activation differences, and gradient attribution. Removing the
top-ranked 10% or 25% usually changed behavior no more than matched random
removal. Refusal was the important exception. This raises a mechanistic
question: when an internal representation causally controls a behavior, does
that representation identify a small teaching set whose removal prevents the
behavior from being learned?

Concept-based filtering itself is established in prior work. My narrower test
asks whether representation-level control predicts data-level removability for
a broad OLMo SFT behavior that resisted earlier selectors.

### Planned test (change to "What I did" after completion)

I will use validation of feelings as the primary behavior and refusal as the
positive control. For each behavior, I will select roughly 3-10 published
OLMo-3 SAE features on discovery prompts and retain them only if amplification
and suppression move the behavior in opposite directions. If the published SAE
fails this transfer gate, the preregistered fallback is one causally checked
dense direction, not new SAE training.

I will score each training response by the equal-weighted, standardized feature
shift from the base model's natural response to the supplied SFT response. I
will remove the top 10% and build dense-probe and random removals matched on
document count and token budget. Each removal condition will use the same LoRA
recipe and one frozen training seed. The primary outcome is the original
100-question rubric evaluation, with bootstrap intervals over questions.

### What I found

- **Bidirectional control screen:** [TBD: feature set or dense fallback, intervention
  strengths, and bidirectional behavior change].
- **Validation of feelings:** [TBD: full-data, concept-selected, dense-probe,
  and random scores with intervals].
- **Refusal sanity check:** [TBD: the same condition-level scores and whether
  the known filterability result reproduced].

> **Figure 1 (TBD).** Plot the validation and refusal scores for the full-data,
> concept-selected, dense-probe, and matched-random conditions. Show bootstrap
> intervals over the original 100 evaluation questions and state that training
> used one seed.

### Interpretation and limitations

If steering and selected removal both work, the behavior is controllable and
its teaching support is concentrated enough to remove at 10%. If steering works
but removal matches the baselines, representation control has not identified a
removable teaching set; distributed support, redundancy, or inherited behavior
remain possible explanations. If the steering gate fails, the SAE is not a
valid selector for this checkpoint and the result is a transfer failure.

This pilot cannot estimate variation across training seeds or removal doses.
It also inherits every documented difference between the public reconstruction
and the motivating organism. These limits keep the claim specific to one
validation experiment, one refusal sanity check, one seed, one 10% dose, and
the original evaluation style.

## Randomly selected evaluation examples

The stream guidance asks for raw examples when data or judgment quality is
load-bearing. After evaluation, sample examples with a frozen random seed and
show the prompt, response, rubric score, and any obvious scoring failure here.

- Sampling seed and rule: **[TBD before evaluation]**
- Examples: **[TBD after evaluation]**
- Scoring problems found: **[TBD or none observed in the displayed sample]**

## Method record

The linked write-up must be understandable without reading code. Record these
facts after the reconstruction is frozen:

- Base model and tokenizer: **[provisional rank-32 reconstruction pins
  `allenai/Olmo-3-1025-7B@a81bae42`; the motivating study does not pin this
  producing revision]**
- Source data: **[candidate `dare-data/data/train_1.parquet@6bc82357`, 25,007
  rows with a replay-verified identity receipt; source-study role, selection
  procedure, adapter-producer link, and license remain unresolved]**
- Full-data adapter: **[candidate `dare-adapter/split-1@767c0ec9`, rank 32;
  source post says rank 64; the observed one-epoch/123-step learning-rate and
  token traces are receipt-bound, but the runnable optimizer/scheduler/
  accumulation/seed/precision/sequence/order recipe is unavailable]**
- Evaluator: **[public prompt/parser contract is receipt-bound for all 201
  archived records; the canonical judge model ID is pinned, while the score
  tolerance, serving-infrastructure boundary, and provider execution remain
  unresolved]**
- Published SAE: **[layer-16 candidate `olmo-3-saes@ad1f790d`; the documented
  Matryoshka BatchTopK-to-JumpReLU inference conversion and static dimensions
  match, and one full-weight hook/encode/decode smoke passes; the training-time
  base revision remains unresolved]**
- Discovery prompts, intervention strengths, selected feature IDs, and dense
  fallback status: **[TBD]**
- Scoring token/aggregation rule and standardization statistics: **[TBD]**
- Removal-set manifests, token totals, random-removal seed, and overlap checks:
  **[TBD]**
- Training run IDs and proof that all removal conditions used the same seed and
  recipe: **[TBD]**
- Original evaluation prompts, rubric implementation, evaluator identity, and
  result artifact hashes: **[TBD]**
- Active-time log for the research task and separate write-up time: **[TBD]**

## Result interpretation matrix

The first row is terminal. The remaining rows apply only after the
bidirectional-control screen passes and concept-selected retraining runs.

| Control screen | Selected-removal effect | Bounded interpretation |
|---|---|---|
| Fails | Not run | The representation did not transfer as a usable selector; data removability was not tested. |
| Passes | Better than dense and random | Controllable and removably taught at the tested 10% dose. |
| Passes | No better than baselines | Controllable representation; removable teaching support was not identified. |

The matrix does not identify distributed support, redundancy, or inheritance
from one seed. Those are possible explanations for a steering-positive,
removal-null result, not findings of this experiment.

## Source-grounded claim boundaries

This section is for review and should be removed from the submitted executive
summary.

| Draft claim | Evidence status | Existing source |
|---|---|---|
| The motivating study used text, representation, and gradient selectors; top-10% and top-25% removal usually matched random, with refusal as the exception. | Supported. | [Source-study evidence note](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/notes/filter-replace.md#seed-post-data-filtering-works-a-lot-worse-than-you-would-expect) |
| Using probes or SAEs to select post-training data is already established. | Supported; this cannot be the novelty claim. | [Literature synthesis](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/synthesis.md#1-concept-filtering-is-a-method-family-not-the-novelty-claim) and [precedent note](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/notes/filter-replace.md#representation-and-causal-control-precedents) |
| The open test is whether causal representation control predicts data removability for broad OLMo SFT behavior. | Indicative novelty verdict, bounded by the completed review rather than an exhaustive claim. | [Verdict](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/synthesis.md#verdict) and [SAE novelty decision](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/notes/sae-data-engineering.md#proposal-1-explain-or-rescue-filtering-with-causal-sae-concepts) |
| A published base-model SAE is a candidate basis, not proof of transfer to the reconstructed LoRA checkpoint. | Supported as an unresolved empirical gate. | [SAE transfer evidence](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/notes/sae-data-engineering.md#proposal-1-explain-or-rescue-filtering-with-causal-sae-concepts) |
| Bidirectional steering is needed before treating a feature as a causal selector. | Supported as a methodological safeguard; it does not prove data causality. | [Causal-audit evidence](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/notes/sae-data-engineering.md#record-b-reviewer-interpretation) |
| Steering success with a removal null proves that support is distributed or inherited. | Unsupported as a proof; the draft treats these as competing explanations only. | [Representation/data distinction](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/synthesis.md#3-representation-causality-and-data-causality-are-different-estimands) |
| The exact motivating organism is publicly reproducible. | Unsupported. The dedicated code, dataset, checkpoint, and complete evaluation package were not found. | [Feasibility finding](../reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/synthesis.md#feasibility-in-this-repository) |
