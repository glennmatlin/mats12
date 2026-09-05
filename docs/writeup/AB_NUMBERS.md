# Part A/B fill sheet (numbers and slot content only; prose is Glenn's)

Every A/B [PENDING] slot from SKELETON.md, resolved to its final number and
trace file. Written so the exec summary and form answers can be drafted
without opening another file. Per the binding rule, no drafted prose here.

## A1 slot 5 (what was then measured)
All of H1-H4 plus R1 ran and are adjudicated; 14 unique filter-retrain cells,
with R1's full fine-tune counted separately; 5.67M-doc provenance scan; every
construction decision logged pre-unblinding (LOGBOOK.md).

## A2 slots 3-6 (findings)
- A2.3 gate clause 2: refusal FILTERABLE at 25% (55.9% of delta removed vs
  random, CI [43.1, 71.3]); both_sides NOT (-7.1% [-18.0, +3.1] at 25%,
  -7.0% at 10%, gradient scorer). results/gate2/VERDICT.md.
- A2.4 H1: NOT SUPPORTED, both predictions. P1b margin -0.065 vs null p95
  +0.008 (contrast reversed: taught validate_feelings load 0.265 > elicited
  both_sides 0.200; flat control china_friendly highest at 0.325). P1a
  margin -0.053 vs null p95 +0.059. Regime-robust: negative under all four
  bases incl. full-FT (-0.035). results/h1h2/h1_h2_measurements.json,
  results/r1/r1_analysis.json.
- A2.5 H2 + H3: order test PASSES at the floor - effects 0.419 < 0.955 <
  1.002 < 1.172 ~ 1.189 in exactly the predicted order, p = 2/120 = 0.0167
  (5 tested units, set + floor logged pre-unblinding); EM control most
  separable (0.9892). H3: P3a FAILS (improvement -0.430, CI [-0.975,
  +0.105], wrong sign); P3b equivalence HOLDS (-0.245, |d| < 1.0) - report
  P3b as a passed prediction. Caveat to state: measured separability
  mis-ranks validate_feelings in-sample. results/gate2/gate2_analysis.json,
  results/h3/h3_analysis.json.
- A2.6 H4: SUPPORTED. Gini 0.652 vs null p95 0.557 (above all 20 draws);
  q_a_forum 4.4x, comment_section 3.9x, faq/customer_support ~2.2x.
  results/h4/h4_analysis.json.

## A2 surprising-number candidates (doc says "the surprising number")
1. p = 0.0167 with the exact predicted order (and the mechanism dead).
2. Deleting 25% of the corpus by both_sides score leaves the behavior
   STRONGER than unfiltered SFT (effect 1.17).
3. The flat control has the highest persona_load (0.325).

## S2 takeaways T3/T4 (resolved)
- T3: the prediction passed at the attainable floor while its mechanism
  failed every test; document inspection supplies the replacement account
  (nameable + concentrated support).
- T4: contrast replicated under a numeric rule; both_sides unfilterable
  under BOTH scorers independently (probe 1.09, gradient 1.17).

## S3 experiment paragraphs (E3/E4/E5 resolved; figure per Part 0 rule)
- Headline figure by the fixed decision rule: FIG-7 (H3 ran). E3 = FIG-4/5
  (H1), E4 = FIG-6/7 (H2/H3), E5 = FIG-8 (H4). All captions self-contained.
- 60-word seeds: use the W2.4-W2.8 opening sentences (each states claim +
  number).

## A4 updates since the skeleton
- Limitation 3 (elicited-label import): partially resolved - liberal_lean
  amended MIXED by the pre-committed rule; ethical_frameworks kept ELICITED
  by disclosed owner decision. Keep, reworded.
- Add: single-seed retrains (one per cell; random baselines 2 seeds).
- R1 no longer a limitation: the cell ran and the result is regime-robust.

## Verbatim-quote guards
- "20-hour sprint" phrasing: 14.5h tracked at QA time; check final TIMELOG.
- Anchor-post figure values: descriptive comparison only, never pooled.
