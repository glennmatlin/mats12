# Execution spec: causally gated concept-gradient attribution

This spec is published as [GitHub issue 14](https://github.com/glennmatlin/mats12-concept-filtering/issues/14)
and extends the replication-backbone execution spec published as GitHub issue
1. Its later scope decision promotes ordinary behavior-query EKFAC into
the pre-feature replication backbone and reserves the causal concept query for
the post-feature phase.

## Problem Statement

The motivating filtering study used ordinary EKFAC to rank SFT documents by
their estimated effect on behavioral outputs, but broad behaviors usually did
not change after the top-ranked documents were removed. The selected project
already tests an activation-based alternative: causally validate an SAE
feature set, score the training push from the base response to the supplied SFT
response, remove the top 10%, and retrain.

The longer research program also needs the parallel gradient-based question:
does ordinary EKFAC fail because its output-level query does not isolate the
right semantic behavior? This lane must replace only the query objective with a
causally validated internal concept while holding the organism, curvature
backend, data, dose, seed, retraining, and evaluation fixed. It must not replace
the training-push idea or combine the two scores.

## Solution

Build a causally gated concept-gradient attribution lane with two phases. The
pre-feature phase reconstructs source-style behavior-query EKFAC, freezes a
shared EKFAC backend, caches reusable curvature factors and training gradients,
and evaluates an ordinary EKFAC removal condition as part of the replication
backbone. It stops before SAE feature selection.

After the parallel feature-selection lane issues a frozen causal-feature
receipt, the concept-gradient phase changes only the query objective. It
computes feature-activation query gradients, applies the same inverse-curvature
approximation and training-gradient store, freezes a top-10-percent removal
manifest, trains one matched condition, and evaluates validation of feelings
and refusal with the existing rubric. The final comparison keeps ordinary
EKFAC, causal concept-gradient attribution, and activation-based training push
as separate estimands.

## User Stories

1. As the researcher, I want ordinary and concept-targeted EKFAC to share one
   backend and curvature cache, so that a performance difference is attributable
   to the query objective rather than implementation drift.
2. As the researcher, I want ordinary source-style EKFAC reconstructed before
   SAE feature selection, so that replication work is not deferred into the
   novel phase.
3. As the researcher, I want the behavior-query EKFAC removal condition trained
   and evaluated with the replication controls, so that its failure or success
   is observed on the reconstructed organism rather than copied from the source
   study.
4. As an operator, I want curvature factors and training gradients cached under
   immutable model, data, loss, layer, backend, damping, and runtime identities,
   so that later concept queries do not repeat behavior-independent work.
5. As an auditor, I want every score to bind its training example, query set,
   factor cache, training-gradient store, sign convention, and backend version,
   so that ranking can be replayed without positional joins.
6. As the feature-selection researcher, I want this lane to consume rather than
   create the causal-feature receipt, so that feature choice and attribution
   evaluation remain independent responsibilities.
7. As the researcher, I want feature-selection, intervention, attribution-query,
   and final-evaluation prompts separated by frozen identities, so that the
   final rubric is not used to tune the selector.
8. As the researcher, I want a multi-feature query objective with frozen feature
   signs, weights, standardization statistics, token aggregation, and prompt
   aggregation, so that the concept score cannot be changed after outcomes.
9. As the researcher, I want the concept-gradient scorer to fail closed when the
   feature receipt is missing, rejected, or incompatible with the organism, so
   that an unvalidated SAE direction cannot silently become a scientific input.
10. As the researcher, I want ordinary EKFAC, concept-gradient attribution, and
    training push reported separately, so that methods with different estimands
    are not blended into an uninterpretable score.
11. As the researcher, I want one top-10-percent removal dose and one frozen
    training seed for the concept-gradient condition, so that this lane remains
    inside the selected validation-plus-refusal protocol.
12. As the researcher, I want matched removal retraining to remain the outcome
    test, so that an attractive influence ranking is not mistaken for evidence
    of removability.
13. As an operator, I want per-query and aggregate work units to resume from
    verified receipts, so that factor fitting or document scoring can recover
    without mixing incompatible partial outputs.
14. As an auditor, I want finite-score, complete-coverage, deterministic-order,
    and tie-handling checks before a removal manifest is issued, so that failed
    or missing scores remain visible.
15. As the researcher, I want negative outcomes preserved, so that failure of
    concept-gradient attribution does not trigger an unplanned selector, dose,
    seed, or evaluation change.
16. As a future reader, I want claims to distinguish the causal feature gate,
    approximate influence score, and finite group-removal result, so that no
    layer of evidence is overstated.

## Implementation Decisions

- The lane uses one pluggable attribution plan with a shared EKFAC backend,
  factor cache, training-gradient store, query objective, score store, and
  deterministic removal-manifest builder.
- Every accepted plan emits a canonical content SHA-256. Changing any organism,
  data, recipe, backend, store, configuration, objective, or boundary identity
  therefore changes the machine-readable summary even when `plan_id` is reused.
- Hash roles that name semantically different inputs remain distinct: organism,
  training manifest, and training recipe cannot share a hash; parameter
  selection, factor configuration, and training-gradient configuration cannot
  share a hash; and validation and refusal cannot share a behavior-query prompt
  bank.
- The preferred backend is the source study's EKFAC implementation. If its
  public interface cannot support both query objectives, one named alternative
  may be frozen only if both ordinary and concept-targeted queries use it. The
  divergence is recorded before scientific scores are inspected.
- The behavior-query control reconstructs the source query loss, on-policy
  response selection, layer coverage, factor-fitting sample, damping,
  accumulation, sequence length, and score aggregation where public evidence
  exists. Missing values become named reconstruction choices.
- Curvature factors are reusable only when the model, adapter, parameter
  selection, training-loss definition, factor-fitting data and order, damping,
  precision, backend, and runtime identities match exactly.
- Training gradients are keyed by the canonical 25,007-example identities.
  Their store binds rendered tokens, loss mask, truncation, reduction, model,
  adapter, parameter selection, backend, and runtime.
- A query objective is either `behavior_loss` or
  `causal_concept_activation`. Unknown objective kinds fail closed.
- A causal concept query consumes a frozen feature receipt containing feature
  IDs, decoder and encoder conventions, signs, weights, standardization
  statistics, hook, token aggregation, prompt aggregation, attribution-query
  prompts, organism identity, and successful bidirectional-control evidence.
- Attribution-query prompts are disjoint from feature discovery and final
  evaluation. Their exact construction is frozen by the feature-selection
  phase before this lane executes a concept query.
- Concept Influence changes only the query-gradient vector. It reuses the same
  inverse-curvature operator and document training gradients as the ordinary
  behavior-query control.
- Positive score means a document is a proponent of the target objective.
  Score sign, normalization, feature aggregation, query aggregation, top-k
  rounding, and tie handling are frozen before rankings are inspected.
- The ordinary and concept-gradient conditions each remove the top 10% from the
  same source manifest. They use the same named recipe, one seed, trainer,
  evaluator, and release contract as the other conditions.
- The activation-based training-push selector remains a parallel method. Its
  score is never substituted for, averaged with, or used to tune a gradient
  score.
- The current implementation checkpoint is pre-feature only. Scientific use,
  SAE feature selection, concept scoring, and concept-selected training remain
  disabled until their explicit upstream receipts and phase authority exist.

## Testing Decisions

- The highest initial seam is the public attribution-plan validator. A caller
  supplies one frozen plan and receives its executable and blocked objectives;
  invalid identities or boundary flags fail before any model or scheduler use.
- Contract tests mutate objective kinds, hashes, feature-gate status, boundary
  flags, falsey non-boolean values, missing fields, factor identities,
  training-gradient identities, query prompts, cross-role hash collisions,
  score order, ties, and coverage. Tests observe public summaries and receipts
  rather than private helpers.
- A fixture backend proves that ordinary and concept queries reuse identical
  factors and document gradients while changing only the query gradient.
- A bounded real smoke fits factors, computes at least one training gradient
  and one behavior query, writes a score, and replays it under exact runtime
  identities before a full cache is built.
- The causal concept path uses a synthetic accepted feature receipt in software
  tests. This fixture proves the contract and does not select a scientific SAE
  feature.
- Full 25,007-document scores and retrained behavior outcomes are scientific
  evidence, not software tests. Their gates may fail while the implementation
  remains correct.

## Out of Scope

- Replacing or abandoning the activation-based training-push selector.
- Selecting SAE features, choosing intervention strengths, or running the
  bidirectional causal-control screen in this lane.
- Blending EKFAC, Concept Influence, and training-push scores.
- Claiming that EKFAC scores are exact causal effects or exact finite-removal
  counterfactuals.
- A 25% dose, second seed, additional behavior, new rubric, layer sweep, SAE
  training, group-complete removal, or another attribution backend comparison.
- Reproducing the external Concept Influence paper's model, dataset, or reported
  experiments.
- Reading from or reusing the separate neighboring application repository.

## Further Notes

The method is named causally gated concept-gradient attribution because causal
evidence comes from the feature-selection receipt. EKFAC supplies an
infinitesimal upweighting approximation, and top-10-percent removal retraining
supplies the finite intervention test.

The source blog and Concept Influence use different EKFAC implementations. The
backend decision therefore remains a reconstruction task, not an assumption.
The comparison is interpretable only if ordinary and concept-targeted queries
share the same frozen implementation and caches.
