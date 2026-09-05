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

**Behavior-query EKFAC control**:
The source-derived gradient attribution condition whose query is the frozen
behavioral evaluation objective. Its scores approximate sensitivity to
training-example upweighting and are validated by matched removal retraining.
_Avoid_: causal influence, concept influence

**Causally gated concept-gradient attribution**:
EKFAC attribution whose query objective is activation along a frozen feature
set that independently passed the bidirectional causal-control screen. The
feature gate is causal evidence; the gradient attribution remains an
approximation until removal retraining validates it.
_Avoid_: causal gradient, proven training cause

**Training-push selector**:
The parallel activation-based method that scores the signed feature-space
shift from the base model's natural response to the supplied SFT response.
It shares the causal feature receipt but not the EKFAC curvature calculation.
_Avoid_: Concept Influence, EKFAC approximation
