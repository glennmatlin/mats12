# MATS stream project idea scratchpad

**Status:** Candidate landscape preserved below; the application project selected
after ideation and literature review is frozen in the next section.

**Application effort estimate:** The original sketch allocated approximately 20
active hours and excluded cluster waiting. This is advisory planning context,
not a clock, budget, or acceptance gate. The replication backbone includes all
source-derived controls and shared prerequisites and ends immediately before
SAE feature selection.

## Shared starting point

The filtering study did not use text-only selectors. It compared an LLM judge
that read documents, a dense activation probe, base-to-SFT activation
differences, and EKFAC gradient attribution. The open question is therefore not
whether internal representations can replace text matching. It is whether a
better representation and intervention unit can expose the causal structure
that document-level rankings missed.

[Concept Influence](https://arxiv.org/abs/2602.14869) already attributes behavior
to semantic directions, including probes and SAE features, and filters
concept-defined groups in emergent-misalignment and post-training settings. A
proposal based only on "filter with SAE concepts" would duplicate that paper.
The sharper gap is whether causally validated concepts can explain or rescue the
broad OLMo behaviors that resisted dense activation, gradient, and text-based
filtering.

The public
[OLMo-3 SAE release](https://huggingface.co/decoderesearch/olmo-3-saes) contains
65,536-latent Matryoshka BatchTopK SAEs for layers 4, 16, and 28 of
`allenai/Olmo-3-1025-7B`. These are trained on the base model rather than the
speed-run LoRA-SFT organism, so transfer quality is an empirical gate rather
than an assumption. A request for the original sample manifest, evaluation
prompts, and training configuration is outstanding; the guaranteed path is a
public reconstruction.

## Selected application project

### When does a controllable concept identify removable SFT data?

**Abstract.** Mechanistic interventions can show that a representation controls
a behavior, while data filtering asks whether a small training subset uniquely
teaches it. Using a public reconstruction of the OLMo-3 speed-run SFT setting,
this project tests whether a small, causally checked SAE feature set for
validation of feelings identifies SFT examples whose removal prevents that
behavior from being learned. Refusal is the positive control. Examples are
ranked by the equal-weighted, standardized feature-space shift from the base
model's natural response to the supplied SFT response. The top 10% are removed
before one-seed LoRA retraining, with the original dense probe and matched random
removal as baselines. The original 100-question rubric evaluation is the primary
outcome. If steering succeeds but removal does not, the pilot separates
representation-level control from data-level removability; if concept removal
outperforms the baselines, it shows that sparse causal concepts recover a
concentrated teaching signal missed by earlier selectors.

**Minimal protocol.**

1. Build the receipt-bound replication backbone: freeze the named recipe,
   25,000-example training set, behavior prompts, and evaluator; train a fresh
   full-data control plus dense-probe and matched-random removal controls; and
   evaluate all three before SAE feature selection begins.
2. On separate discovery prompts, select roughly 3-10 published OLMo-3 SAE
   features per behavior. Keep the set only if a lightweight amplification and
   suppression check moves the behavior in both directions. If the published
   SAE does not transfer, use one causally checked dense direction rather than
   train a new SAE.
3. Score each training response by its signed, standardized feature-space push.
   Remove the top 10%, then create dense-probe and random removals matched on
   document count and token budget.
4. Retrain each removal condition with one frozen LoRA seed. Evaluate validation
   of feelings and refusal with the original 100-question rubric; use bootstrap
   intervals over questions to show evaluation uncertainty.
5. Treat one validation experiment plus the refusal sanity check as the complete
   project, whether the result is positive or null.

**Scope guard.** This is a replication-plus-one-change project, not a new
benchmark. A 25% dose, cluster-complete removal, a second training seed, extra
selector retraining, new SAE training, layer sweeps, and new evaluations are
stretch work only. Existing clusters may explain a null descriptively, but they
do not add another required intervention.

**Advisory application-work estimate.**

| Work | Active hours |
|---|---:|
| Freeze inputs and reproduce the reference scores | 2 |
| Check SAE transfer, select features, and run the causal sanity check | 5 |
| Score the data and build the three matched removal sets | 3 |
| Launch retraining and evaluate completed adapters | 5 |
| Analyze the result and write the application report | 5 |
| **Original estimate** | **20** |

The estimate does not limit reconstruction or baseline work. The canonical
dependency and stopping boundary are in the
[replication backbone roadmap](../plans/2026-08-15-replication-backbone-roadmap.md).

## Core OLMo concept-filtering family

### Proposal 1: Explain or rescue filtering with causal SAE concepts

**Abstract.** Broad assistant behaviors survive targeted SFT-data removal even
when attribution methods rank documents that appear behavior-relevant. We test
whether this failure comes from selecting examples in the wrong representational
coordinates. Using a public reconstruction of the OLMo-3 speed-run organism, we
compare refusal, which the original study found filterable, with validation of
feelings, which appears genuinely strengthened by SFT but resisted filtering.
Candidate SAE features must both distinguish behavior-positive from
behavior-negative completions and increase from the base model to the SFT model.
Only features whose suppression and amplification bidirectionally change the
target behavior are eligible for data selection. We then rank SFT examples by
the feature-space shift from the base model's response to the supplied training
response and compare 10% removal against matched random removal, the original
dense probe, and raw SAE activation. A preregistered 25% follow-up diagnoses
insufficient dose. Success would show that sparse causal concepts rescue
filtering; failure after successful steering would show that representation-level
control does not imply data-level removability.

**Two-week shape.** Run a two-hour SAE transfer gate, build separate discovery,
steering-validation, and final-evaluation prompt splits, and restrict causal
retraining to refusal and validation. Map cheaper descriptive statistics across
the remaining five behaviors.

**Main risk.** The base-model SAEs may not reconstruct or preserve feature
identity after LoRA SFT. If that gate fails, preserve the scientific question
with causally validated dense directions rather than spending the application
budget training a new SAE.

### Proposal 2: Diagnose the determinants of filterability

**Abstract.** The original study observes that refusal is filterable while most
broad assistant behaviors are not, but it does not isolate the property that
separates them. We test three candidate determinants across all seven behaviors:
internal sparsity, concentration of the SFT teaching signal, and overlap with a
generic assistant-persona direction. Refusal and validation receive deeper
causal feature interventions, while the remaining behaviors receive a common
descriptive screen. The primary hypothesis is that filtering succeeds when the
causal teaching signal is concentrated in a small subset of examples; feature
sparsity and assistantness overlap are competing explanations. This project
prioritizes a mechanism map over a new filtering method. Its strongest result
would be a preregistered statistic that correctly places refusal at the
filterable end and validation at the distributed end, supported by causal
steering of the corresponding representation.

**Two-week shape.** Compute concentration curves and effective support sizes for
all behaviors, then use refusal and validation for matched causal case studies.
Retraining is limited to the minimum cells needed to verify that the diagnostic
tracks intervention outcomes.

**Main risk.** Seven behaviors supply few independent units, so an apparent
relationship could be a visually persuasive two-point contrast rather than a
general law. The project needs careful case-study framing unless it adds more
behavioral targets.

### Proposal 3: Benchmark four concept-filtering selectors

**Abstract.** Sparse concepts, causal validation, and counterfactual response
comparison are three different modifications to ordinary activation-based data
selection. We isolate their contributions in a compact benchmark. For refusal
and validation, we compare matched random removal, the original dense activation
probe, activation of causally validated SAE features, and the feature-space
training push from the base response to the supplied SFT response. Each selector
removes the same fraction of the same public SFT reconstruction before identical
LoRA retraining. The design asks a narrow engineering-science question: which
selector most predictably changes held-out behavior without degrading general
assistant quality? A positive result identifies the minimal ingredient needed
to improve filtering. A null result bounds the value of concept-level selection
under the same organism that defeated dense and gradient-based methods.

**Two-week shape.** Use one 10% removal dose first and two behavior targets. Add
the 25% dose only as a diagnostic when concept selection does not beat random.

**Main risk.** This has the cleanest comparison but the weakest explanatory
story. If all concept variants fail, the analysis must already contain a
mechanism diagnostic rather than ending at four null bars.

### Proposal 4: Separate representation causality from data removability

**Abstract.** Interpretability interventions and data filtering answer different
causal questions. A representation may mediate a behavior at inference time
even when no small training subset uniquely teaches it. We explicitly construct
a two-axis test: first measure whether steering a direction changes refusal or
validation; then measure whether removing the examples that most push that
direction changes the retrained model. This yields four possible regimes:
controllable and removable, controllable but distributed, correlated but not
causal, and neither captured nor removable. The contribution is a falsifiable
framework for interpreting filtering failures rather than another selector
leaderboard. Refusal and validation provide a naturally occurring contrast, and
SAE versus dense directions test whether the regime assignment depends on the
representation basis.

**Two-week shape.** Reuse the same steering and retraining cells as Proposal 1,
but make the two-axis causal decomposition the headline and treat selector
performance as supporting evidence.

**Main risk.** The framing is stronger than the empirical breadth. The write-up
must resist generalizing a useful conceptual distinction from only two
behaviors.

## Alternative mechanisms and filtering fixes

### Proposal 5: Crosscoder model diffing for inherited versus taught behavior

**Abstract.** A broad behavior can resist SFT-data filtering because it is
already encoded in the mid-trained model and merely elicited by post-training,
or because SFT teaches it through many weakly redundant examples. We use
base-versus-SFT model diffing to identify features that are inherited, newly
created, or strengthened during LoRA SFT, then test whether each feature class
predicts document removability. Refusal and validation again form the primary
contrast. If refusal maps to features created or sharply strengthened by SFT
while validation maps to inherited assistant-persona features, the analysis
would explain their divergent filterability. If both map to newly learned
features but only refusal is removable, the explanation shifts from inheritance
to distributed teaching support.

**Two-week shape.** Prefer existing aligned-feature methods or a lightweight
decoder-direction matching analysis. Do not train a full multi-condition
crosscoder unless a small smoke test establishes that it fits inside the active
time budget.

**Main risk.** Training or aligning a new crosscoder can consume the whole
application, and feature rotation can make apparent birth or death of concepts
an alignment artifact.

### Proposal 6: Cluster-complete filtering for redundant concept support

**Abstract.** Top-k document removal assumes that influence is concentrated in
individually rankable examples. If a behavior is taught by many interchangeable
examples, removing the highest-scoring documents leaves semantic substitutes
that restore the same concept. We compare ordinary top-k removal with
cluster-complete removal: identify causal behavior concepts, group SFT examples
by SAE-feature or semantic support, and remove entire high-contribution clusters
under the same document budget. The decisive comparison matches the number and
coarse data composition of removed examples, so any gain comes from covering
redundant routes rather than increasing dose. Success would turn distributed
support into a constructive filtering method. Failure would suggest that the
redundancy crosses the available concept partition or that the behavior is
elicited from the base model rather than taught in the SFT data.

**Two-week shape.** Restrict the method to validation of feelings and compare one
cluster-aware selector with one score-matched top-k selector and random removal.

**Main risk.** SAE features overlap rather than partition documents, so
"cluster-complete" needs a deterministic rule that does not silently remove more
or more-diverse data than the baseline.

### Proposal 7: Residualize generic assistantness before filtering

**Abstract.** Several behaviors in the filtering study may be components of a
generic assistant persona rather than independently taught traits. A behavior
probe can therefore rank examples that teach assistantness without isolating
the target behavior. We estimate an assistantness direction, remove its
component from refusal and validation directions, and compare filtering with
the original dense probe, the residualized direction, and matched random
removal. The central prediction is asymmetric: residualization should matter
most for validation if it is entangled with supportive assistant behavior, while
refusal may remain separable. This is a simpler alternative to SAEs and directly
tests the persona-confounding explanation raised by the original study.

**Two-week shape.** Use the same prompt splits and retraining organism as the
core proposal, but replace feature discovery and SAE steering with linear
subspace tests.

**Main risk.** Linear subtraction can remove predictive variance without
isolating a causal persona component. It needs steering or mediation controls,
not only improved probe accuracy.

## Alternative model organisms and calibration studies

### Proposal 8: Dose-matched positive-control ladder

**Abstract.** The original broad-behavior experiments remove 10% or 25% of SFT
data, while the risky-finance positive control removes 50% of a balanced mixture.
That control establishes source recovery at a large intervention dose but does
not show that the selectors work where the main result fails. We vary both the
prevalence of behavior-inducing data and the removal budget, comparing random,
text judge, dense probe, and concept-based selectors. The resulting phase
diagram separates three quantities that are otherwise conflated: source
classification accuracy, fraction of causal support removed, and downstream
behavior change. A strong result would show a threshold at which apparently
excellent attribution becomes causally useful, explaining why a method can pass
the toy test and fail on broad SFT behavior.

**Two-week shape.** Use two mixture prevalences and two removal doses rather than
a full grid. Preserve one known-source oracle to reveal whether selector failure
or irreducible redundancy sets the limit.

**Main risk.** Reconstructing another group's emergent-misalignment training and
judge stack may take longer than the scientific analysis, especially without
released code.

### Proposal 9: Calibrate attribution against exact counterfactual retraining

**Abstract.** Training-data attribution scores are often evaluated by whether
their top documents look relevant, but filtering asks whether changing those
documents changes the learned model. We use a small modern causal-language-model
organism where exact leave-group-out retraining is affordable, measure the true
behavioral effect of data groups, and compare those effects with MAGIC,
TrackStar, EKFAC, dense probes, and concept-based scores. The aim is not to win a
filtering benchmark but to identify which attribution quantities predict actual
retraining effects and when scalable approximations break. This creates a
faithfulness ladder from exact counterfactuals to OLMo-scale methods and could
explain why plausible selectors fail in the larger organism.

**Two-week shape.** Use one behavior, tens rather than thousands of data groups,
and a causal model small enough for repeated retraining. Scale to OLMo only after
the attribution-versus-effect relationship is measured.

**Main risk.** The stream explicitly disfavors old-model-only projects. A small
organism must be presented as a calibration stage tied to a modern OLMo test,
not as the final evidence.

### Proposal 10: Query coverage as a predictor of causal usefulness

**Abstract.** A selector can accumulate a high attribution score by explaining a
few evaluation queries extremely well while missing most ways a behavior is
expressed. We test whether the fraction and entropy of behavior queries covered
by a selected document set predict intervention effects better than mean
attribution score. The retrospective study uses the existing OLMo intervention
bank, including strong ToM transfer and negative selector counterexamples, with
a locked discovery-validation split. A prospective experiment then compares
high-coverage and low-coverage selectors matched on size, score distribution,
and topic bins. This addresses the same distributed-support hypothesis without
reconstructing the new SFT-filtering organism.

**Two-week shape.** Build one selector-level table, preregister a primary coverage
metric, validate on held-out intervention families, and run a small matched
confirmatory grid only if the retrospective relationship survives.

**Main risk.** The existing interventions use NGDiff unlearning rather than
filter-and-retrain SFT, so the result speaks to attribution-guided intervention
predictability rather than directly resolving the new filtering paper.

### Proposal 11: Replace selected teaching responses instead of deleting data

**Abstract.** Deleting examples changes both the target teaching signal and the
amount or distribution of assistant training. A more targeted intervention
replaces high-contribution responses with behavior-neutral responses while
preserving prompts, domains, and token budgets. We compare deletion, neutral
replacement, and matched random replacement for refusal and validation. If
replacement succeeds where deletion fails, the original null may reflect
generic assistant-data loss or adjacent examples compensating for missing
coverage. If both fail, the behavior is more plausibly inherited or diffusely
taught. Concept-level training-push scores can choose which responses to
replace, but the core contribution is the intervention rather than the
selector.

**Two-week shape.** Use one behavior and one neutral-response generator with a
small human audit. Treat the paired refusal comparison as optional unless the
first cells are stable.

**Main risk.** Related work has already used teacher-rollout swapping to study
necessary and sufficient prompt subsets. Novelty depends on testing
concept-selected, behavior-neutral replacement in this exact filtering organism,
not on replacement alone.

## Relationship between the proposals

Proposals 1 through 4 share one experimental backbone and differ mainly in the
headline claim. Proposal 1 makes the strongest outcome-symmetric application:
either concepts rescue filtering or they reveal a separation between causal
control and removability. Proposals 2 and 4 can become the explanatory analysis
inside it, while Proposal 3 is the minimal ablation package.

Proposals 5 through 7 are alternative mechanisms or fixes if the literature
shows that SAE concept filtering is already saturated. Proposals 8 through 11
are distinct application shapes that remain available if reconstructing the
OLMo speed-run organism is too fragile or insufficiently novel.

## Questions for the literature landscape

1. Has any work tested concept-level or SAE-based data filtering on broad,
   persona-like SFT behaviors rather than synthetic or narrow misalignment?
2. What evidence connects SAE-feature causality under steering to the training
   examples that created or strengthened those features?
3. Which studies distinguish representation-level controllability from
   training-data removability?
4. What measures of teaching-signal concentration, redundancy, coverage, or
   effective support predict whether filtering works?
5. Which proposal remains both novel and executable under the 20-active-hour
   application constraint?
