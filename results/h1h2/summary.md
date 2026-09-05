# H1 collapse + H2 separability - measurement readout (ticket #7)

Run 2026-08-25, `scripts/13_h1_h2_measurements.py`, seed 20260816.
Every number here is re-derivable from `h1_h2_measurements.json` in this
directory. Pre-registration was already frozen (2026-08-16 at `e3eb75f`); this
run consumes it and does not amend it.

Primary basis: the frozen mid-train-base **L22, k=4** persona subspace - the
space the probe directions were fit in. Behavior directions are the seven
probe `w_hat` vectors in `artifacts/probes/`.

Group assignment follows PREREGISTRATION H1's own parenthetical: elicited =
{both_sides, ethical_frameworks}, taught = {refusal, validate_feelings}.
`china_friendly` carries `Prior.ELICITED` in `behaviors.py` but is the flat
control (no SFT delta to attribute), so the prereg's enumeration governs and it
is reported descriptively. `liberal_lean` left the elicited group in the
2026-08-16 MIXED amendment.

## P1b - persona_load, elicited vs taught: NOT SUPPORTED

| behavior | group | persona_load |
|---|---|---|
| china_friendly | control | 0.3248 |
| ethical_frameworks | elicited | 0.2953 |
| validate_feelings | **taught** | 0.2647 |
| both_sides | **elicited** | 0.1997 |
| liberal_lean | mixed | 0.1930 |
| bold_formatting | mixed | 0.1912 |
| refusal | taught | 0.1554 |

Contrast = min(elicited) − max(taught) = 0.1997 − 0.2647 = **−0.0650**.
Random-direction null (n = 2000 matched-norm draws, same 2+2 group structure):
mean −0.0116, 95th percentile **+0.0080**. The contrast must exceed the null's
95th percentile to count as support; −0.0650 does not. **P1b fails, with the
contrast running opposite to the prediction** - the taught behavior validate_feelings is more
persona-aligned than the elicited behavior both_sides.

Descriptive context, not a pre-registered claim: a random direction in a k=4
subspace of d=4096 has expected persona_load ≈ 0.031. All seven behavior
directions sit 5–10× above that, so persona alignment is real and large across
the board - the elicited/taught split just does not organize it. The single
most persona-aligned direction belongs to `china_friendly`, the flat control
with no SFT delta at all; that is a caution about reading persona_load as a
behavior-strength signal.

## P1a - top-1000 document overlap: NOT SUPPORTED

Overlap within the elicited pair: **0.147** (both_sides × ethical_frameworks).
Elicited × taught overlaps: 0.001, 0.011, 0.117, **0.200**.
Contrast = 0.147 − 0.200 = **−0.0530**.

The five largest of the 21 pairwise overlaps:

| pair | overlap |
|---|---|
| bold_formatting × liberal_lean | 0.265 |
| ethical_frameworks × liberal_lean | 0.234 |
| refusal × validate_feelings | 0.226 |
| ethical_frameworks × validate_feelings | 0.200 |
| bold_formatting × ethical_frameworks | 0.184 |

The two **taught** behaviors overlap each other (0.226) more than the elicited
pair overlaps itself (0.147), and `both_sides` is nearly disjoint from both
taught behaviors (0.001 against refusal, 0.011 against validate_feelings).
Collapse, where it appears, is not organized by the elicited/taught prior:
`ethical_frameworks` overlaps heavily with almost everything while `both_sides`
overlaps with almost nothing.

Random-direction null (n = 200 draws, each rebuilding the 2+2 group structure
by scoring matched-norm random directions over the 25K corpus activations,
`doc_acts_base_L22.npz` fetched from PACE 2026-08-25): mean **−0.0476**, 95th
percentile **+0.0590**. The observed contrast (−0.0530) fails the support bar
and sits almost exactly at the null mean - on this statistic the
elicited/taught grouping behaves like randomly drawn directions. The null's
negative mean confirms the min-minus-max construction is biased negative (one
within value against four cross values), which is why the verdict waited for
it.

## P1b robustness readout

Reported together, no post-hoc selection between them (prereg Setting):

| basis | margin (elicited − taught) | status |
|---|---|---|
| base L22 (primary) | −0.0650 | computed |
| sft, own selected layer L12 | −0.0457 | computed (cross-layer against L22 probes) |
| sft, matched-layer L22 | −0.0085 | computed |

All three readouts agree in sign: the elicited > taught contrast never holds.
The matched-layer L22 sft basis was rebuilt on CPU from the extraction job's
`role_means_sft.npz` via the identical `build_basis` path
(`scripts/compare_bases.py --emit-sft-l22`) and provenance-verified: its
principal angles against the frozen base basis reproduce
`base_vs_sft_comparison.json` exactly (8.37/12.71/23.94/34.36°). Committed as
`artifacts/persona_subspace/persona_subspace_sft_L22.npz`, labeled a
robustness recompute, not a replacement artifact. One structural note: refusal
is by far the most persona-orthogonal direction under the sft-model basis
(load 0.074 vs 0.155 on the base basis) - SFT moves the persona subspace
*away* from refusal while the elicited/taught margin stays null.

## H2 - order test PASSES at the attainable floor (p = 0.0167)

**Verdict (2026-08-27):** the three pending cells landed
(validate_feelings 0.9551, bold_formatting 1.0015, ethical_frameworks 1.1887;
`gate2_analysis.json`, frozen refusal/both_sides targets verified unchanged),
and the observed ascending-effect order is **exactly the predicted order**:
refusal < validate_feelings < bold_formatting < {both_sides,
ethical_frameworks}. All 9 pre-registered relations hold; 2 of 120 orderings
are consistent with the prediction, so the one-tailed p equals the
pre-stated floor, 0.0167. Failed-SFT gate clean on all three cells (0
think-truncations of 700 each; empties 0-11).

**Caveat, reported with the pass:** the prediction got the *effects* order
exactly right, but the separability *instrument* itself mis-ranks one
behavior in-sample - validate_feelings sits 4th on measured separability
(0.9643) while its filtering effect sits 2nd, so observed separability is
not fully monotone with observed effect (the inversion is
both_sides/validate_feelings). The pre-registered evaluation form is the
permutation test of the predicted relations, which is what passed; the
in-sample separability-vs-effect Spearman is reported descriptively in the
write-up, not as a gate.

### Pre-verdict record (as written before the cells landed)

subspace_separability = √(1 − persona_load²), so it is a monotone (and, near
load ≈ 0, strongly compressive) transform of the table above. Rank order is
exactly the inverse; `persona_load` is the more legible readout for prose.

| behavior | separability |
|---|---|
| refusal | 0.9879 |
| bold_formatting | 0.9816 |
| liberal_lean | 0.9812 |
| both_sides | 0.9799 |
| validate_feelings | 0.9643 |
| ethical_frameworks | 0.9554 |
| china_friendly | 0.9458 |

**EM positive control (added 2026-08-26, construction decided pre-unblinding):**
separability **0.9892** on the frozen primary basis - above all seven
behaviors, as P2 predicted. Holds too under the sft L12 robustness basis
(load 0.1007, below every behavior's); under the sft L22 matched basis
refusal edges above EM (0.0738 vs 0.1096) - disclosed, the primary basis
governs. Probe: risky-vs-benign mixture docs, base L22, train accuracy 1.0,
|cos| ≤ 0.11 against every behavior probe
(`results/em_control/em_direction.json`).

P2's predicted separability ordering was refusal > validate_feelings >
bold_formatting > {both_sides, ethical_frameworks}. Observed over the five
tested units: refusal > bold_formatting > both_sides > validate_feelings >
ethical_frameworks. `refusal` is top as predicted; `validate_feelings` lands
fourth rather than second.

**Tested-unit set, fixed before unblinding** (owner decision, LOGBOOK
2026-08-25): refusal, validate_feelings, bold_formatting, both_sides,
ethical_frameworks. Five units under P2's level structure = 120 orderings, 2
consistent with the prediction, so the attainable **p-floor is 2/120 =
0.0167**.

Own-judge filtering effects at 25% removal, (S_filtered − S_base)/(S_SFT −
S_base), from `results/gate2/gate2_analysis.json`:

| behavior | effect at 25% | source |
|---|---|---|
| refusal | 0.4191 | own-judge, probe-filtered |
| both_sides | 1.1716 | own-judge, Bergson-filtered |
| validate_feelings | - | needs retrain cell |
| bold_formatting | - | needs retrain cell |
| ethical_frameworks | - | needs retrain cell |

The order test is therefore **blocked on three filtered retrain chains**,
**submitted 2026-08-25** on release `7abee75` (train → evalgen, afterok, bad
node excluded, 12h evalgen wall): validate_feelings 5591525→5591526,
bold_formatting 5591527→5591528, ethical_frameworks 5591529→5591530. The
manifests are committed  - 
`{validate_feelings,bold_formatting,ethical_frameworks}_probe_25.json`, 6,250
documents dropped each - and all three behaviors hold escalation trigger 2, so
they stay on the probe scorer with no Bergson leg. Base/SFT means and both
`random_25` baselines are reused.

Anchor-post figure values (bold_formatting 0.77, validate_feelings 0.55,
ethical_frameworks 1.00) exist in `behaviors.py` as the pre-registered
correlation target only. They are recorded in the JSON under an explicitly
non-pooled key and are **not** substituted for the missing own-judge values.

## Disclosures

- **Margin scalarization (Tier-2).** P1a and P1b both read "group statistic
  exceeds every cross-group value". The prereg fixes the relation, not its
  reduction to one number; this run uses min(within) − max(cross), the literal
  reading, and compares it to a null built with the identical group structure.
- **china_friendly's group.** Prereg enumeration over `behaviors.py`'s
  `Prior.ELICITED` tag, as described above.
- **Scorer asymmetry.** both_sides' own-judge effect comes from a
  Bergson-filtered cell, the other tested units from probe-filtered cells  - 
  the asymmetry already disclosed in `results/gate2/VERDICT.md`.
- **Pending.** The H2 order test (three retrain chains in flight, job ids
  above); the owner pass over the hand-inspection samples.
