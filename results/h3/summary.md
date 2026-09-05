# H3 - persona-decontaminated refiltering (verdicts 2026-08-28)

Construction and matched-method baselines fixed pre-unblinding (LOGBOOK
2026-08-27): u_dec = project_out(probe w_hat, frozen base L22 k=4 basis),
re-rank the 25K corpus, drop top 25%, retrain; baseline = the
undecontaminated probe cell at 25% for the same behavior. Frozen rules:
P3a improves := baseline-minus-decontam >= +1.0, 95% CI excluding 0;
P3b not-improve := |difference| < 1.0 equivalence. Numbers re-derivable
from `h3_analysis.json` (paired per question, 10k bootstrap, seed
20260816); cell effects in `results/gate2/gate2_analysis.json`.

| prediction | cells (decontam vs baseline) | improvement | 95% CI | verdict |
|---|---|---|---|---|
| P3a (validate_feelings, predicted improves) | 2.58 vs 2.15 target mean | **−0.430** | [−0.975, +0.105] | **does NOT improve - P3a FAILS** |
| P3b (both_sides, predicted no improvement) | 3.43 vs 3.19 target mean | −0.245 | [−0.730, +0.230] | **equivalent - P3b HOLDS** |

Failed-SFT gate: clean on all three cells (0 think-truncations of 700 each;
empties 1/1/0).

## Reading

- **P3a fails in the informative direction.** Removing the persona-subspace
  component from validate_feelings' direction did not sharpen filtering; the
  point estimate is mildly negative (decontaminated filtering retained MORE
  of the behavior, n.s.). If persona-bundling had been what shielded the
  behavior, decontamination should have exposed a cleaner behavior-specific
  signal. It did not - consistent with H1's failure and with the
  hand-inspection account (filterability tracks nameable data clusters, not
  persona-subspace geometry).
- **P3b holds, but for the mundane reason.** The prereg's mechanism said
  "nothing behavior-specific should remain after decontamination"; the
  simpler account, already visible in the manifest geometry (cos(v, u_dec) =
  0.98, 80% drop-set overlap), is that decontamination barely changed the
  ranking at all - because both_sides' direction was never substantially
  inside the subspace (load 0.20).
- **Convergent bonus from the baseline cell:** both_sides_probe_25 - the
  behavior's own PROBE ranking at 25% - also fails to remove the behavior
  (effect 1.09, above unfiltered SFT). both_sides now survives filtering
  under BOTH its scorers (probe and Bergson), independently of
  decontamination.
- Specificity: non-target behaviors sit near 1.0 in all three cells; the
  one notable off-target movement is refusal at 0.72 under
  validate_feelings_decontam_25 (its drop set overlaps refusal's
  adversarial-safety cluster), reported descriptively.

Prediction ledger after H3: H1 P1a/P1b not supported; H2 P2 passes at the
p-floor (0.0167) with the EM control above all; H3 P3a fails, P3b holds.
