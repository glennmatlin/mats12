# Execution spec: receipt-bound replication backbone

## Problem Statement

I need to reach SAE feature selection without discovering that the apparent
"novel experiment" still depends on an undefined training recipe, unreproduced
baseline, missing response cache, ad hoc cluster command, or evaluator choice.
The current repository proves many public reconstruction facts, but it cannot
yet execute comparable full-data, dense-probe-removal, and matched-random-
removal conditions end to end.

The original 20-hour sketch is not an acceptance criterion. The work is done
when receipts prove that the complete replication backbone is runnable,
comparable, resumable, and ready to hand to the SAE feature-selection phase.

## Solution

Build one frozen, receipt-bound execution DAG from the public rank-32 named
reconstruction through fresh baseline evaluation. The DAG will freeze a
runnable LoRA recipe, reproduce the evaluator gate, materialize every shared
input, reconstruct the dense-probe selector, create a token-and-document-
matched random control, reconstruct the source-style behavior-query EKFAC
selector, train four fresh conditions, evaluate them with the original rubric,
and issue a readiness handoff.

One command family will plan, inspect, run, resume, and verify the DAG. Each
side-effecting stage will validate immutable release roles and upstream hashes
before it creates output. The terminal verifier will fail closed unless the
next unfinished scientific action is SAE feature selection.

## User Stories

### Scope and governance

1. As the researcher, I want the replication backbone and novel SAE phase to
   have a machine-checkable boundary, so that baseline work cannot quietly be
   deferred into the novel experiment.
2. As the researcher, I want effort estimates separated from acceptance gates,
   so that completeness is determined by evidence rather than elapsed hours.
3. As the researcher, I want every source-supported value distinguished from a
   named reconstruction choice, so that the final write-up does not overclaim
   exact reproduction.
4. As the run owner, I want provider, scheduler, deployment, publication, and
   cleanup authority recorded once, so that unattended work neither stalls on
   repeated questions nor exceeds its grant.

### Planning and recovery

5. As an operator, I want to render the complete stage DAG without side
   effects, so that I can inspect dependencies and frozen identities before a
   run.
6. As an operator, I want a status command that reads durable receipts, so that
   it reports evidence rather than inferring progress from processes or paths.
7. As an operator, I want interrupted stages to resume only missing units, so
   that model generations, activation extraction, provider calls, and training
   are not repeated unnecessarily.
8. As an auditor, I want output directories to reject conflicting reuse, so
   that a later run cannot overwrite or silently mix evidence from an earlier
   one.

### Runnable organism

9. As the researcher, I want a complete named LoRA recipe, so that fresh
   conditions differ only in their data manifests even though the source recipe
   is incomplete.
10. As the researcher, I want every recipe divergence recorded before training,
    so that observed outcomes cannot influence hyperparameter reconstruction.
11. As an operator, I want a bounded train-checkpoint-resume smoke, so that the
    expensive runs do not become the first test of tokenization, loss masking,
    checkpoint writing, or recovery.
12. As an auditor, I want training receipts to bind model, tokenizer, data,
    recipe, seed, runtime, releases, checkpoints, and terminal metrics, so that
    condition comparability is reconstructible.

### Evaluator and shared inputs

13. As the researcher, I want the closed Gate R2 rule accepted before provider
    execution, so that the reference tolerance cannot change after scores are
    observed.
14. As the researcher, I want exactly one judge sample for every retained
    response, so that retries cannot select favorable parseable outcomes.
15. As a data consumer, I want one stable identity joining each training prompt,
    supplied response, base response, token record, activation record, selector
    score, and condition membership, so that no positional join can corrupt the
    experiment.
16. As an operator, I want base responses and shared preprocessing generated
    with per-example receipts, so that a partial long run can resume safely.
17. As the later SAE researcher, I want the activation-extraction interface and
    aggregation contract tested before feature selection, so that the novel
    phase starts with a working shared pipeline rather than new infrastructure.

### Replication controls

18. As the researcher, I want the public dense-probe method reconstructed from
    its discovery examples through its document scores, so that the control is
    more than a label attached to an unrelated classifier.
19. As the researcher, I want probe centering, normalization, regularization,
    token aggregation, and tie handling frozen, so that the top-10% manifest is
    deterministic and auditable.
20. As the researcher, I want the random control to match document count exactly
    and token budget under a predeclared tolerance, so that unequal training
    exposure does not explain the comparison.
21. As the researcher, I want a fresh full-data control trained with the same
    recipe core as the dense-probe, matched-random, and behavior-query EKFAC
    removal controls, so that recipe drift does not masquerade as filtering.
22. As the researcher, I want all four fresh training conditions to share one
    seed and one recipe core, so that the frozen pilot changes only the data
    manifest. Each execution-authority envelope is bound separately from that
    reusable recipe identity.

### Evaluation and handoff

23. As the researcher, I want every fresh adapter evaluated on validation and
    refusal with the original rubric contract, so that the baseline result uses
    the same outcome measure as the motivating study.
24. As an auditor, I want per-question completions, judge requests, raw judge
    responses, parsed scores, aggregates, and bootstrap inputs linked, so that
    every reported number can be replayed.
25. As the researcher, I want failed and missing conditions preserved in the
    aggregate artifact, so that the reporting layer cannot silently select only
    successful runs.
26. As the later SAE researcher, I want a terminal readiness receipt listing the
    exact remaining novel steps, so that feature selection begins without
    hidden shared, baseline, compute, or evaluation work.
27. As a future reader, I want the public rank-32 candidate and fresh full-data
    control reported separately, so that reconstruction fidelity and filtering
    effects remain distinct questions.

## Implementation Decisions

- The canonical workflow is a directed acyclic graph with stages for governance,
  orchestration, recipe reconstruction, reference scoring, shared inputs,
  dense-probe selection, matched random selection, behavior-query EKFAC, fresh
  training, evaluation, and readiness verification.
- The command surface has four operations: plan the DAG, report receipt-backed
  status, run through a named stage, and verify the terminal readiness receipt.
- A stage consumes immutable artifact identities and produces an exclusive
  receipt plus any content-addressed large outputs. An existing valid receipt is
  a replay; an existing contradictory output is an error.
- The full-data, dense-probe-removal, matched-random-removal, and behavior-
  query-EKFAC-removal conditions use the same named rank-32 reconstruction
  recipe and seed. The public rank-32 adapter remains a separate reference
  candidate rather than the fresh control.
- The training recipe explicitly freezes optimizer, scheduler, learning rate,
  warmup, rank and targets, effective batch, accumulation, precision, maximum
  sequence length, packing, truncation, chat template, loss mask, shuffle order,
  checkpoint cadence, and resume semantics.
- The canonical example identity originates in the frozen 25,007-row sample and
  is carried through tokenization, base responses, activations, scores, and
  removal manifests. Joins validate identity sets instead of relying on row
  position.
- Shared preprocessing includes full-data materialization, token accounting,
  resumable base-response generation, and a smoke-tested activation interface.
  It does not select SAE features, run interventions, calculate concept scores,
  or create a concept-selected manifest.
- The dense-probe control uses the source method's synthetic high/low discovery
  responses, mean activation representation, L2 logistic regression,
  mean-centering, normalized direction, and signed projection formula wherever
  recoverable. Missing settings become predeclared named reconstruction choices.
- The random selector freezes its seed, algorithm, token-count definition,
  matching tolerance, tie-breaking, and infeasibility behavior before outcomes.
- Every remote stage records executor, coordinator, input-producer, and output
  releases. The pre-side-effect gate rejects a missing, mutable, or disallowed
  release tuple before materialization, provider execution, or scheduler
  submission.
- Compact configs and receipts are versioned in the repository. Large model,
  response, activation, checkpoint, and run artifacts live in designated cache
  or shared-project storage and are referenced by hashes and release receipts.
- Provider execution and scheduler submission are independent capabilities.
  Offline commands cannot acquire either capability through a transitive flag.
- The terminal status is `ready_for_sae_feature_selection`. It is false unless
  the four fresh conditions and their evaluations are complete, ordinary EKFAC
  caches and the control replay, and the shared pipeline contracts replay from
  a clean checkout.

## Testing Decisions

- The highest seam is the complete backbone DAG. Fixture provider, trainer,
  scheduler, and artifact adapters must produce the same terminal receipt on a
  clean run and on an interrupted-then-resumed run.
- Contract tests observe command results and durable receipts rather than
  private helper calls. They mutate identities, counts, order, flags, roles,
  seeds, recipes, condition membership, and partial outputs to prove that the
  workflow fails closed.
- Stage integration tests use the smallest real artifact that exercises the
  boundary: one model generation, one activation capture, one trainer step and
  resume, one provider request fixture, and one no-submit cluster launch plan.
- Full-data scientific runs are evidence, not tests. Their outcomes may fail a
  scientific gate while the software and provenance contract still pass.
- The project-wide acceptance suite remains formatting, lint, strict type
  checking, unit and integration tests, lock validation, diff checks, and the
  source-function size audit already used by the repository.

## Out of Scope

- SAE feature selection, feature review, or selection of intervention strengths.
- Causal SAE amplification or suppression experiments.
- Concept-score computation or a concept-selected removal manifest.
- Concept-selected training, evaluation, comparison, or scientific conclusion.
- Causally gated concept-gradient scoring before an accepted feature receipt,
  LLM-judge selection, additional activation selectors, or selector grids.
- A 25% dose, a second seed, extra behavior suites, new rubrics, new SAE
  training, layer sweeps, cluster interventions, response replacement, or
  source-study side experiments.
- Claiming that the named rank-32 reconstruction recovers unavailable rank-64
  artifacts, producing code, or source hyperparameters.
- Reading from or reusing the separate neighboring application repository.

## Further Notes

The detailed stage gates, dependency graph, command sketch, artifact rules, and
ten implementation slices live in the replication backbone roadmap. Gate R2 is
a reference-fidelity result, while the terminal readiness gate is an execution
completeness result; neither should be renamed as a timer boundary.

The roadmap may continue local and offline work while an owner-gated provider
decision is open. Before unattended provider or scheduler execution, the
authority ledger must contain the exact accepted policy, resource class, and
allowed dependency chain so the run cannot encounter a planned approval pause.

The causal-gradient extension has its own execution spec and roadmap. Ordinary
behavior-query EKFAC is source-derived replication work required before the
readiness handoff. The later concept query consumes a separately accepted
causal-feature receipt and does not move feature selection into this backbone.
