# Reconstructed speedrun recipe (no configs from the authors)

Status 2026-08-13: Dohun Lee and J Rosser were DM'd; no reply. Public sweep
found nothing (GitHub code search on distinctive strings from the post's
appendix, author account search, HF search) — consistent with Dohun's comment
on the anchor post: "no plans to open source at the moment." We therefore
reconstruct the recipe from the post itself and proceed. If configs ever
arrive, diff against this file and re-run the replication gate only if a
PINNED item turns out wrong or a RECONSTRUCTED choice is materially different.

The replication gate (PREREGISTRATION.md) targets **phenomenon** reproduction
(SFT >> base; refusal filterable at 25%, both_sides not), not number matching.
The post's own comment thread shows exact-number replication failing for a
direct replicator (Ziqian Zhong could not reproduce the 29->96 bold-formatting
gap); phenomenon-level replication is the defensible bar, and every
reconstructed choice below is disclosed in the write-up.

## Tier 1 — PINNED (verbatim from the post; do not change)

| Item | Value | Source in post |
|---|---|---|
| Base checkpoint | OLMo 3 7B mid-train (`allenai/Olmo-3-1025-7B` @ `main`; interpretation note in config.py) | Set Up |
| Pre-mid variant | stage-1-only checkpoint, SFT'd 4 epochs ("4x the training time") | Potential Explanations; fig legend "Pre-mid SFT (4 ep)" |
| Speedrun epochs | 1 ("Custom SFT (1 ep)" in the pipeline figure) | fig legend |
| LoRA | rank 64, all 32 layers, MLP + attention | Set Up |
| Dataset | Dolci-Think-SFT-7B; drop examples > 8192 tokens (2.27M -> ~1.19M) | Appendix |
| Sample | stratified by dataset source; 125K = 5 splits x 25K; TDA runs on one 25K split | Appendix figure (strata table in `01_build_speedrun_sft.py`) |
| Behaviors | the 7 in `behaviors.py`; validate_feelings rubric verbatim | Behavior Evaluations |
| Eval | 100 questions/behavior, rubric-scored, judge = Claude Sonnet 4.6, slop-aware instruction (don't mark down slop/distraction) | Behavior Evaluations |
| Second axis | assistant-compliance eval ("usable, in-role assistant answer?") | Did we actually find differences |
| Removal | top 10% and top 25% by TDA score; random-removal baseline with equal n | Evidence 1 |
| TDA judge prompt | full verbatim prompt in the post appendix (system + user template, indirect-influence framing, calibration note) | Appendix |
| Probe method | OLMo3-32B generates synthetic +5/-5 rubric examples; L2 logistic regression on mean activations; s(x) = w-hat^T (h(x) - mu) | Appendix |
| Activation method | base-to-SFT activation-shift direction, cosine scoring (formulas in appendix) | Appendix |
| EKFAC | kronfluence library; factors fit on training data (~2h on 8xH200) | Appendix |
| EM positive control | 6000 risky-finance (Turner et al., arXiv:2506.11613) + 6000 benign ultrachat_200k train_sft; EM eval = 8 questions x 100 gens, GPT-4o judge | Toy Test Bed |
| Compute envelope | retrain + eval approx 2.6h / $47 on 8xH200 (sanity bound for our runs) | Cost/Effectiveness |

## Tier 2 — RECONSTRUCTED (chosen, not copied; disclosed in write-up)

| Item | Our choice | Rationale / sensitivity |
|---|---|---|
| Learning rate | 1e-4, cosine decay, 3% warmup | Standard LoRA-SFT range for 7B; the post is silent. Anchor sanity check against allenai/open-instruct OLMo-3 SFT configs during prep. Phenomenon robustness matters, not the exact value; if the gate fails, LR is the first sweep axis (5e-5, 2e-4). |
| Effective batch | 32 sequences (per-device x grad-accum to fit) | Must complete 25K docs/epoch inside the ~2.6h envelope on 8 GPUs. |
| Max seq len | 8192 | Implied by the length filter. |
| Seed | 20260813 (config.py), single seed | The post reports single runs with SEM over generations, not seeds. |
| Sampling seed for 25K split | ours; doc IDs published in results/ | Their split is unrecoverable; strata proportions are pinned. |
| Eval question sets | our own 100/behavior, generated + committed before any scoring | Post shows 3 examples for one behavior only. Committed pre-freeze to avoid tuning questions toward predictions. |
| Rubrics for 6 behaviors | reconstructed in the validate_feelings format (+5/+3/0/-3/-5 anchors) from behavior names/descriptions | Marked non-verbatim in behaviors.py; the judge refuses to run until each is written and committed. |
| Behavior-eval judge prompt | our own wrapper (JUDGE_TEMPLATE in judge.py) with the slop-aware instruction | Post gives the TDA-judge prompt verbatim but only describes the eval judge. |
| Judge backend | headless claude-cli sonnet; cross-provider sharding per PLAN.md if rate limits pinch | Post used Sonnet 4.6 via API; agreement protocol + disclosure in PLAN.md. |
| Generation settings for evals | temp 0.8, top-p 0.95, max 512 new tokens | Post silent; fixed pre-freeze. |

## Divergence ledger

Append-only; every deviation discovered or decided after this file's first
commit gets a dated line here plus a LOGBOOK entry.

- 2026-08-13: file created; no known divergences beyond Tier 2 choices.
- 2026-08-14 (pre-training, pre-scoring): LoRA alpha 16, dropout 0.1 — the
  open-instruct Tulu LoRA precedent, same source as the Tier-2 LR anchor.
  Loss masked to the final assistant turn (open-instruct convention); no
  sequence packing. If the gate shows a weak-but-directional SFT effect,
  alpha is sweep axis 2 after LR.
- 2026-08-14: single H100 instead of the post's 8xH200 (walltime envelope
  differs ~4-6h vs 2.6h; recipe unchanged; effective batch still 32).
- 2026-08-14: 8192-token filter operationalized as chat-templated token count
  (think-SFT tokenizer, add_special_tokens=False), which includes ~40 tokens
  of template/system-prompt overhead per example.
- 2026-08-14 (pre-SFT-generation): SFT-condition eval uses the think-SFT chat
  template (with its default Olmo system prompt) and max 2048 new tokens —
  the frozen 512 is kept for the base condition, but the template opens
  `<think>` and reasoning traces routinely exceed 512, which would leave no
  visible answer to judge. The judge scores the post-`</think>` answer;
  completions that never close the think block are counted and excluded.
  eos for the SFT condition = <|im_end|> + <|endoftext|> (template turn ends).
- 2026-08-14 (pre-Step-1-scoring): judge prompt now includes the eval
  question as context (JUDGE_TEMPLATE); the 2026-08-13 practice-loop judging
  did not show the judge the question. Rubric-only judging misreads
  question-dependent behaviors (refusal, both_sides), and no Step-1 scoring
  had happened yet.
- 2026-08-16 (probe implementation pins, pre-score): generator =
  allenai/Olmo-3-32B-Think @ f2edda15216e738ef2bb73771e11890e152b2112
  (the post pins only "OLMo-3-32B"; same-generation Think chat model chosen,
  alternatives: Olmo-3.1 variants), chat template, temperature 0.7, max 2048
  new tokens, ONE sample per (behavior, question, target) over the frozen
  question sets, post-</think> answer is the example (truncated rows
  excluded), prompt template verbatim in scripts/05_generate_probe_synthetics.py.
  QC filter, pre-stated: each synthetic judged by the single-Claude judge;
  kept iff sign(score) matches its target and |score| >= 3; kept/dropped
  counts reported. Probe fit: sklearn LogisticRegression (C=1.0, l2, lbfgs,
  max_iter=1000, fit_intercept), direction unit-normalized, mu = mean of all
  training activations (src/mats12/attribution.py). Document representation
  for ALL activation harvesting: final assistant message content, raw text,
  no special tokens, capped 4096 tokens, mean-pooled at base-convention L22
  - identical through base (probe), base+adapter (activation-shift), and
  synthetics. Competing readings named per the scorer challenge: full
  chat-templated conversation; SFT-model or 32B-generator activations.
  Chosen for probe-corpus consistency and base-model in-distribution text.
- 2026-08-16 (scorer naming, pre-freeze): the Tier-1 "Activation method" row
  ("base-to-SFT activation-shift direction, cosine scoring") over-claimed:
  the post's appendix formula is an unnormalized document-shift dot product
  while its prose says "cosine similarity", and the two rank long documents
  differently. Resolution, pre-registered before any score: the appendix
  formula as written is the fallback scorer's form; cosine is a logged
  robustness variant. Probe Tier-2 parameters to pin before the first score:
  activation source = base 7B at L22 (competing readings the post permits:
  the speedrun-SFT model, or the 32B generator itself — chosen reading keeps
  persona_load comparability with the frozen basis), generator variant,
  synthetic count, L2 strength, judge-QC filter on synthetics, shared-bias
  cosine threshold 0.9. Scorer package + escalation triggers:
  PREREGISTRATION.md Setting, 2026-08-16.
- 2026-08-16: base-condition sign divergence on the elicited behaviors —
  unprefilled base scores negative under our question-visible judge
  (both_sides -0.69, ethical_frameworks -2.47; summary.md) vs the post's
  base-positive Okay-prefill signature. Candidate cause: the judge-template
  change above plus prefill-vs-no-prefill elicitation. Gate metric
  (SFT-minus-base delta) unaffected. In-setting Okay-prefill validation run
  pre-committed (recovery criterion: prefilled-minus-unprefilled paired
  delta >= +2.0 per behavior, 95% CI excluding 0), per the 2026-08-16
  pre-freeze edit bundle D2.
- 2026-08-20 (Bergson escalation Tier-2 pins, recorded before any gradient
  score is computed or viewed): pipeline = `bergson trackstar` at commit
  631d1871bcbdeb98ec7b203684cd16e1bf5888a5 (glennmatlin/bergson main),
  installed with --no-deps into the frozen mats12-venv so torch and peft
  stay at their training versions. Model = the canonical speedrun adapter
  (speedrun-sft-r64-s20260813, PEFT auto-detected over olmo3-base);
  tokenizer = olmo3-think-sft (as in training; recorded as olmo3-base at
  first writing, corrected same day pre-score — the base tokenizer has no
  chat template); precision bf16; normalizer adafactor (TrackStar
  paper form); projection_dim 16 (default); preconditioner stats sample
  10,000 docs (default); query aggregation mean + unit_normalize; score
  method individual against the aggregated query gradient. Corpus = the
  25K split via conversation_column=messages with truncation at
  token_batch_size 4096: loss falls on ALL assistant spans, vs training's
  last-turn-only masking — a divergence affecting the 588/25,000
  multi-assistant docs (2.4%), accepted because bergson's span masking
  cannot express last-turn-only without altering the rendered context.
  Queries per escalated behavior (both_sides, china_friendly) = the 100
  frozen eval questions as the user turn with the Step-1 SFT-condition
  generations as the assistant turn (results/step1/gens-sft, frozen
  artifacts; loss on the completion span). Per-doc behavior score = the
  individual score against the mean query gradient; manifests drop the
  top 25%/10% by that score, mirroring 09_build_filter_manifests.py.
- 2026-08-20 (Bergson corpus addendum, pre-score): 9/25,000 docs (0.036%)
  have assistant text the think-SFT chat template does not reproduce
  verbatim (whitespace normalization), which bergson's loss-span search
  rejects. These docs are excluded from the Bergson scoring corpus and can
  therefore never be selected by the bergson filter; ids logged by
  11_prep_bergson_data.py --tokenizer. The retrain corpus is unaffected.
