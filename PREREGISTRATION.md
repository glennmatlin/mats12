# Pre-registration — "Persona provenance predicts what filtering can't fix"

Opened 2026-08-13. **Freeze rule:** this file freezes at the first computation
of any attribution or document-influence score over any behavior query set,
wherever in the plan it occurs (reworded 2026-08-16, pre-freeze; previously
bound to "Step 3 in PLAN.md", but gate-clause-2 scoring computes attribution
earlier). Until then, edits are allowed
and logged; after, amendments only as clearly-dated addenda plus a LOGBOOK
entry. Written before any persona subspace, behavior query, or attribution
score has been computed in this project.

## Setting

OLMo3-7B mid-train + rank-64 LoRA speedrun SFT on a 25K stratified Dolci sample
(anchor-post recipe); behaviors and rubrics from the anchor post; judge =
slop-aware Claude Sonnet. Persona subspace: frozen spec in
docs/persona_subspace_freeze.md (2026-08-14, pre-Step-3 as required): k = 4 =
assistant-contrast axis + top-3 role PCs, vendored Assistant-Axis role set
verbatim, mid-train-base activation space with an SFT-basis robustness
recompute, layer chosen by pre-registered d′ criterion in L12–L22.
subspace_separability and persona_load are the tested implementations in
src/mats12/separability.py at the freeze SHA. Robustness-recompute readout
(pinned 2026-08-16, pre-freeze, written after the basis artifacts were known
but before any attribution outcome): persona_load robustness is reported
under BOTH the sft artifact's own selected layer (L12) and the matched-layer
L22 recomputation (artifacts/persona_subspace/base_vs_sft_comparison.json),
together, with no post-hoc selection between them.

**Attribution scorer (named 2026-08-16, pre-freeze; decision memo and
adversarial challenge in docs/reviews/scorer-memo-2026-08-16.md):**

- Primary: the anchor post's probe scorer, formula as pinned in the post
  appendix — s(x) = w-hat^T (h(x) - mu), w-hat the unit-normalized
  L2-logistic direction trained on +5/-5 synthetic rubric examples generated
  by OLMo-3-32B. Activations read from the base model at the frozen layer
  L22, mean-pooled — the same space as the frozen k=4 basis, so persona_load
  is directly computable. Parameters the post leaves unpinned (generator
  variant, synthetic count, L2 strength, judge-QC filter on synthetics, and
  the competing readings of whose activations the post caches) are recorded
  in RECONSTRUCTION.md Tier-2 before the first score.
- Fallback, scoped to probe-generator failure only: the activation-shift
  scorer with the appendix formula as written (unnormalized document shift);
  cosine logged as a robustness variant (divergence-ledger amendment
  2026-08-16 — the prior Tier-1 row over-claimed "cosine" where the post is
  internally inconsistent).
- Escalation (the representation-independent leg): gradient TDA via the
  existing Bergson pipeline (arXiv:2606.19625 infrastructure; native PEFT
  support verified), logged as a Tier-2 divergence from the post's EKFAC.
- Escalation triggers, fixed before any score is viewed: (1) split-half
  probe stability — Spearman >= 0.8 between document-score vectors from
  probes fit on random halves of the QC-kept synthetics (halves drawn with
  seed 20260816); (2) probe/activation-shift
  top-25% set overlap must reject the random-draw null at hypergeometric
  tail p < 0.05; (3) shared-bias trigger — a behavior whose probe direction
  has cosine >= 0.9 with another behavior's escalates to gradient TDA.
  Trigger (3) governs H2/H3 uses only: for gate clause 2 the probe is the
  replication instrument regardless, because the post's own filtering
  contrast was produced with its probe/judge scores, bias included — and
  elicited-direction parallelism is itself the H1 prediction, reported as a
  free H1 measurement either way.
- Failed-SFT gate: a filtered retrain whose assistant-eval rate regresses
  toward mid-train is flagged as failed SFT before its behavior score is
  interpreted (the post's ethical-frameworks diagnostic).
- All GPU-hour/storage figures in the memo are derived estimates anchored to
  job 5577697's measured wall-clock, not measurements.

## Hypotheses and predictions

**H1 (collapse).** Attribution rankings for elicited behaviors (both_sides,
ethical_frameworks) collapse onto shared persona-typical documents.
(liberal_lean amended ELICITED -> MIXED 2026-08-16, pre-freeze, per the
pre-committed Okay-prefill recovery criterion: 18% of its SFT delta
recovered, CI spans zero;
results/step1/judging-prefill/recovery_analysis.json. It leaves the elicited
group and is reported descriptively wherever that group is measured.
ethical_frameworks kept ELICITED by owner decision 2026-08-16: it fails the
recovery bar's letter (+1.78 < +2.0) while recovering 101% of its own SFT
delta, and the bar exceeds that delta entirely - the mis-specified criterion
is disclosed, both numbers reported in the write-up.)
- P1a: pairwise top-1000 document overlap among elicited behaviors exceeds
  overlap between any elicited-taught pair.
- P1b: persona_load of elicited behaviors' attribution directions exceeds that
  of taught behaviors (refusal, validate_feelings), on the same layer/basis.
- Null for P1a/P1b (added 2026-08-16, pre-freeze, extending H4's null to H1
  per edit bundle D6): both statistics are also computed against
  matched-norm random directions (n >= 20, same construction as H4's null);
  the elicited-vs-taught contrast must exceed the null's 95th percentile to
  count as support.

**H2 (prediction).** subspace_separability rank-orders per-behavior filtering
effectiveness at 25% removal.
- P2: predicted separability ordering: refusal > validate_feelings >
  bold_formatting > {both_sides, ethical_frameworks}; EM positive control
  (Turner et al. risky-finance mixture) above all. (liberal_lean unranked
  since its 2026-08-16 MIXED amendment; its separability and filtering
  effect are reported descriptively, outside the permutation test.)
- Effect metric (reworked 2026-08-16, pre-freeze; was a pooled Spearman):
  filtering effect at 25% removal := (S_filtered - S_base) / (S_SFT - S_base),
  paired means under this project's judge on the frozen question sets;
  replicated own-judge values replace behaviors.py provisional values by
  rule, always.
- Evaluation: exact one-tailed permutation test of the predicted order
  relations, own-judge behaviors only, with {both_sides, ethical_frameworks}
  as one equivalence class (no within-class order predicted; class synced to
  P2's 2026-08-16 liberal_lean amendment).
  The attainable p-floor is 1/(orderings of tested units) and is stated when
  the tested-unit set is fixed, before unblinding (e.g. 4 tested units ->
  floor 1/24 ~= 0.042).
- Anchor-post figure values appear only in a labeled descriptive comparison,
  never pooled with own-judge values.
- china_friendly is excluded from H2 by design: it is the flat control (no
  SFT delta to filter).

**H3 (causal, with a pre-registered expected failure).** Persona-decontaminated
re-ranking (project_out persona subspace from the behavior direction, re-rank,
drop top 25%, retrain):
- P3a: improves filtering for validate_feelings vs its matched-method baseline.
- P3b: does NOT improve filtering for both_sides (expected failure — nothing
  behavior-specific should remain after decontamination).

**H4 (provenance — pretraining index).** Persona-subspace-scored influence over
the 5.68M-doc Dolma3 sample concentrates in assistant-shaped regions:
- P4: top-decile bins are dominated by interpersonal/instructional formats
  (e.g. forum/Q&A/tutorial/personal_blog formats) rather than spread uniformly
  across the 576 bins; concentration measured by Gini vs a random-direction
  null (matched-norm random directions, n >= 20).

## Robustness cells (design updates from the lit review)

- R1 (LoRA-artifact confound, arXiv:2606.00831): repeat the H1 persona_load
  measurement on ONE additional training regime — **full-FT, one epoch**
  (decided 2026-08-13, pre-freeze; rank-16 only as a logged compute fallback).
  If collapse appears only at rank 64, the mechanism is rank compression, not
  persona bundling — report either way.
- R2 (if any unlearning is used): pair with a relearning-attack probe (100
  fine-tune steps on held-out behavior data); suppression relearns fast,
  removal does not (arXiv:2506.06278).

## Scope statements (pre-committed caveats)

- The index measures the visible data channel only; subliminal channels can
  carry traits filtering cannot see (arXiv:2507.14805 and successors).
- Claims are scoped to SFT-stage post-training of this model family; the
  persona framing degrades under heavy RL (PSM comment threads).
- "Persona subspace" may be diffuse rather than cleanly low-rank
  (persona-collapse, arXiv:2605.12850); partial separation is an anticipated
  outcome and is why H3 has a paired positive/negative prediction.

## Decision rules

- Replication gate (Step 1) must reproduce: SFT >> base on the three replication
  behaviors, and refusal filterable at 25% while both_sides is not. If the gate
  fails, the project pivots to "when does the filtering-failure phenomenon
  replicate" per the admissions doc's restart rule — that pivot is itself a
  pre-registered branch, not a failure of this pre-registration.
- No thresholds tuned after seeing H2/H3 outcomes; ordinal predictions above
  are the committed form.

Numeric forms (added 2026-08-16, pre-freeze):

- Gate clause 1 "SFT >> base" := paired delta >= +2.0 with 95% bootstrap CI
  excluding 0 (`scripts/summarize_step1.py`, seed 20260816). Under this rule
  the passed gate re-passes: +3.56 / +5.04 / +5.35.
- "Filterable at 25%" := >= 50% of the SFT-minus-base delta removed relative
  to the equal-n random-removal baseline; "not filterable" := <= 20%; 20-50%
  reported as indeterminate, no gate claim either way. Random-baseline count:
  fixed in the #8 retrain-matrix sizing decision BEFORE any filtered-retrain
  result is viewed; default one retrain per (behavior, fraction) at seed
  20260813 if budget-bound; if multiple seeds run, their mean is the
  baseline (added 2026-08-16, pre-freeze).
- H3 "improves" := >= 1.0 judged-point gain over the matched-method baseline
  with 95% CI excluding 0. P3b "does not improve" := |difference| < 1.0 as an
  explicit equivalence bound (single-seed limitation disclosed).

## 2026-09-04 post-run bin-accounting addendum

The H4 plan anticipated 576 WebOrganizer bins from pre-run accounting.
Materializing the frozen manifest yielded 577 distinct nonempty bins, and the
analysis retained all 577 (`n_unmapped_dropped=0`). This observation was made
after the H4 run. It changes neither the pre-registered Gini statistic nor the
finite random-direction comparison. This addendum corrects accounting only;
the original prediction and decision rules above remain unchanged.
