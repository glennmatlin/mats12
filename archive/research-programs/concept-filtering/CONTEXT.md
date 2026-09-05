# Concept Filtering

This project reconstructs the shared experimental backbone needed to test
whether a causally controllable representation identifies removable SFT data.
It distinguishes source-derived controls from the later concept-selected arm.

## Language

**Replication backbone**:
The complete, receipt-bound organism, data, selector controls, training runs,
evaluation, and automation needed before SAE feature selection can begin.
_Avoid_: Phase 0, setup work, timed experiment

**Named reconstruction**:
A runnable protocol whose inputs and divergences are explicit even when the
source study's exact producing artifacts cannot be recovered.
_Avoid_: exact replication, faithful reproduction

**Full-data control**:
A fresh adapter trained on the complete frozen sample with the same named
reconstruction recipe used for the removal conditions.
_Avoid_: public adapter, baseline adapter

**Dense-probe control**:
The source-derived activation-space logistic-regression selector and its frozen
top-10-percent removal condition.
_Avoid_: concept selector, SAE probe

**Matched-random control**:
A frozen random removal condition with the same document count and a
predeclared token-budget match to the dense-probe control.
_Avoid_: random baseline

**Novel boundary**:
The first SAE feature-selection decision. Causal SAE interventions, concept
scores, and concept-selected training are beyond the replication backbone.
_Avoid_: experiment start, timer start

**Readiness handoff**:
The terminal receipt proving that the replication backbone is complete and
that the next action is SAE feature selection, with no unresolved shared work.
_Avoid_: Gate R2, done
