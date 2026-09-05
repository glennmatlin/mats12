# H4 - provenance over the pretraining bins (verdict 2026-08-28)

Readout pinned in LOGBOOK 2026-08-28 pre-unblinding; implementation
`scripts/20_h4_analysis.py`; full numbers in `h4_analysis.json` (copied from
the PACE analysis output; per-shard scores at `$STORE/h4/scores/`, 316
shards, release 112f81b7 workers).

**P4 SUPPORTED.** Gini of the persona axis's top-decile document mass over
the WebOrganizer bins = **0.6516**, vs the 20 matched-norm random-direction
null: 95th percentile 0.5568, maximum 0.5807. The axis exceeds every null
draw.

Sample: 5,666,170 scored docs (of the 5,678,621-doc manifest; the shard
files carry slightly fewer rows), 577 distinct bin ids, 0 docs unmapped to
the manifest.

Format composition of the axis's top decile (corpus base rate ~0.042 per
format):

| format | top-decile mass | enrichment |
|---|---|---|
| q_a_forum | 0.187 | 4.4x |
| comment_section | 0.165 | 3.9x |
| faq | 0.097 | 2.3x |
| customer_support | 0.092 | 2.2x |
| about_pers | 0.055 | 1.3x |
| tutorial | 0.052 | 1.2x |

The top 8 individual bins are q_a_forum/comment_section bins across diverse
topics (electronics, transportation, crime_and_law, software, health,
social_life; enrichment 6.3-8.2x). This is the "interpersonal/instructional
formats rather than spread uniformly" pattern P4 predicted, on the model's
actual pretraining sample.

Note: the null's Ginis (~0.38-0.58) are themselves far from zero - ANY
direction's top decile concentrates somewhat by bin because bins differ in
size and content; that is exactly why the verdict is against the
random-direction null rather than against uniformity.
