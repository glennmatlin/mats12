# Gate clause 2 - filterability verdicts (2026-08-22)

Machine-readable source of truth: `gate2_analysis.json` (regenerate with
`uv run python scripts/10_gate2_analysis.py`). Judge scores per cell:
`judging/summary.md`. Frozen decision rules: PREREGISTRATION.md @ `e3eb75f`;
Bergson Tier-2 pins and the corpus addendum: RECONSTRUCTION.md divergence
ledger (2026-08-20 entries).

Rule, verbatim from the freeze: removed := (S_random - S_filtered) /
(S_random - S_base), random baseline = mean of seeds 20260813/20260814 at the
same fraction. Filterable at >= 50%, not filterable at <= 20%, indeterminate
between. CIs are paired percentile bootstrap over question_id, 10,000
resamples, seed 20260816.

| target | fraction | cell | removed vs random | 95% CI | verdict |
|---|---|---|---|---|---|
| refusal | 25% | `refusal_probe_25` | 55.9% | [43.1%, 71.3%] | **filterable** |
| refusal | 10% | `refusal_probe_10` | 32.4% | [20.8%, 45.3%] | indeterminate |
| both_sides | 25% | `both_sides_bergson_25` | -7.1% | [-18.0%, 3.1%] | **not filterable** |
| both_sides | 10% | `both_sides_bergson_10` | -7.0% | [-18.7%, 3.4%] | **not filterable** |

Gate clause 2 (refusal filterable at 25%, both_sides not) therefore PASSES,
and with clause 1 (SFT >> base on the replication triple, re-passed at freeze:
both_sides +3.56, refusal +5.04, validate_feelings +5.35) the full
pre-registered replication gate PASSES.

## Disclosures

- The refusal @25% CI spans the 50% bar ([43.1%, 71.3%]). The frozen rule
  adjudicates on the point estimate, which was fixed before any score existed;
  the interval is reported rather than reinterpreted.
- refusal @10% lands in the pre-registered indeterminate band. No decision is
  drawn from it.
- Escalation asymmetry, by design: refusal was filtered with the activation
  probe scorer, both_sides with Bergson/TrackStar gradient attribution after
  the pre-registered escalation trigger fired. The negative both_sides result
  is therefore the stronger-scorer result, not a weak-scorer artifact.
- Sign of the both_sides number: dropping the top-scoring quarter left the
  behavior slightly stronger than random deletion did (effect 1.17 vs the
  unfiltered SFT 1.0), i.e. no targeted subset of the corpus carries it.
- Specificity, from `filtering_effects_all_behaviors` in the JSON: under
  refusal-probe filtering, non-target behaviors retain effect ~1.0
  (both_sides 1.29); under both_sides-bergson filtering, non-targets sit at
  0.94-1.03. Filtering hits its target and little else.
- Failed-SFT check: think-block truncation is 1-3 of 700 generations per cell,
  so no cell is explained by a collapsed retrain.
