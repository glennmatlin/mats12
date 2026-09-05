# CONTEXT — domain glossary

Seeded 2026-08-13 by `/setup-engineer` from `README.md`, `PREREGISTRATION.md`,
`RECONSTRUCTION.md`, and `src/mats12/`. This is the project's ubiquitous
language: when naming an issue, a test, a hypothesis, or a refactor, use these
terms rather than synonyms.

Maintained lazily by `/domain-modeling` — add a term when it actually gets
resolved, not speculatively.

## The question

Why SFT **data filtering** fails for some behaviors and works for others, and
where in the pretraining corpus the support for persona formation lives.

- **Anchor post** — the Alignment Forum post "Data filtering works a lot worse
  than you would expect" (Dohun Lee, J Rosser). The source of the pinned recipe,
  the seven behaviors, and the phenomenon we replicate. No configs were
  released; see `RECONSTRUCTION.md`.
- **Replication gate** — the Step-1 pass/fail bar: SFT ≫ base on the replication
  triple, *and* `refusal` filterable at 25% removal while `both_sides` is not.
  The bar is **phenomenon** reproduction, not number matching. Failing the gate
  triggers a pre-registered pivot, not a silent retune.

## Behaviors

- **Behavior** — one of the seven target traits in
  [`behaviors.py`](src/mats12/behaviors.py), each with a description, a rubric,
  and a prior. Keys: `bold_formatting`, `both_sides`, `china_friendly`,
  `ethical_frameworks`, `liberal_lean`, `refusal`, `validate_feelings`.
- **Prior** — the elicited-vs-taught classification (`Prior` enum):
  - **elicited** — recovered by Okay-prefill on the mid-train base; the model
    already had it, SFT surfaced it.
  - **taught** — not recovered by prefill; installed during SFT.
  - **mixed** — neither cleanly.
- **Rubric status** (`RubricStatus`) — provenance of the scoring rubric:
  `VERBATIM` (copied from the anchor post — only `validate_feelings`),
  `RECONSTRUCTED` (written by us in the post's ±5/±3/0 format, disclosed),
  `SUMMARY` (placeholder; **the judge refuses to run against these**).
- **Replication triple** — `(both_sides, validate_feelings, refusal)`, the three
  behaviors the replication gate is evaluated on.

## Filtering and attribution

- **Filtering effect** — the change in a behavior's judged score after removing
  the top-scoring slice of SFT training documents and retraining. Measured at
  **10%** and **25%** removal, each against a **random-removal baseline** with
  equal n.
- **TDA / attribution** — training data attribution: scoring SFT documents by
  their influence on a behavior. See [`attribution.py`](src/mats12/attribution.py).
- **Behavior query set** (`BehaviorQuerySet`) — a behavior's attribution
  direction: a unit vector in activation space, layer-matched to the persona
  basis, derived from contrast pairs.
- **Persona-decontaminated re-ranking** — the H3 intervention: `project_out` the
  persona subspace from a behavior direction, re-rank documents, drop the top
  25%, retrain. Predicted to help `validate_feelings` and *not* help
  `both_sides` (a pre-registered expected failure).

## Persona geometry

- **Persona subspace** (`PersonaSubspace`) — a k ≤ 5 orthonormal basis at a fixed
  layer, built from the assistant-contrast vector plus top role PCs
  (Assistant-Axis recipe). **Frozen before Step 3**; carries a `source`
  provenance string.
- **persona_load** — how much of a direction lies inside the persona subspace.
  H1's discriminator: elicited behaviors should load higher than taught ones, on
  the same layer and basis.
- **subspace_separability** — how separable a behavior direction is from the
  persona subspace. H2 claims this rank-orders filtering effectiveness.
- **project_out** — remove the persona subspace's component from a direction.
- **topk_overlap** — pairwise top-k document overlap between two attribution
  rankings; H1's collapse measure at k = 1000.
- **d′ (`dprime`)** — standard separation statistic used for probe quality.

The math above lives in [`separability.py`](src/mats12/separability.py) and is
the tested core of the package — change it only with tests.

## Corpus and training

- **Speedrun SFT** — the reconstructed recipe: OLMo3-7B mid-train base +
  rank-64 LoRA over a 25K stratified Dolci sample, 1 epoch.
- **Speedrun split** — one 25K stratified sample of `Dolci-Think-SFT-7B`
  (examples over 8192 tokens dropped). Five splits × 25K = 125K; TDA runs on
  one split.
- **Pretraining bins** — the 576 format/topic bins over the 5.68M-doc Dolma3
  sample that H4 scores for concentration (Gini vs a matched-norm
  random-direction null).
- **EM positive control** — the emergent-misalignment mixture (risky-finance,
  Turner et al.) expected to sit above all seven behaviors on separability.

## Process terms

- **PINNED / RECONSTRUCTED** — the two tiers in `RECONSTRUCTION.md`. PINNED
  values are verbatim from the anchor post and must not be changed;
  RECONSTRUCTED values are ours and are disclosed in the write-up.
- **Divergence ledger** — the append-only list at the foot of
  `RECONSTRUCTION.md`. Every deviation gets a dated line plus a `LOGBOOK.md`
  entry.
- **Freeze** — `PREREGISTRATION.md` freezes at the first attribution run over
  behavior queries (Step 3). After that, dated addenda only.
