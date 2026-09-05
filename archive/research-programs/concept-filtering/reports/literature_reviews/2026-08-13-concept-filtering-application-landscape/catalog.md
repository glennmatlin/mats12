# Evidence catalog

## How to read this catalog

This is a landscape, not a systematic review. The ledger contains 271
canonical records: 22 received targeted full-text extraction, 55 remain
abstract-level includes, and 194 were excluded at title/abstract screening.
Detailed claims in the synthesis rely on the 22 primary-result-checked records,
not on titles or abstracts alone.

The machine-readable source of truth is `paper_ledger.csv`. Search, merge, and
screening receipts are in `search_log.jsonl`, `dedup_log.jsonl`, and
`screening_log.jsonl`.

## Core evidence by question

### Has concept-level data selection already been done?

| Record | Direct evidence | Boundary for this project | Note |
|---|---|---|---|
| [Concept Influence](https://arxiv.org/abs/2602.14869) | Probe/SAE semantic directions, group filtering, and retraining on emergent-misalignment and post-training tasks | Does not test the broad OLMo behaviors that resisted filtering or exact group-removal effects | `notes/tda-counterfactuals.md` |
| [Persona Vectors](https://arxiv.org/abs/2507.21509) | Projection-difference scores select trait-inducing SFT examples and causal steering changes traits | Does not test whether broad assistantness-confounded examples are uniquely removable | `notes/tda-counterfactuals.md` |
| [Feature Superposition Geometry](https://arxiv.org/abs/2605.00842) | SAE toxic-feature geometry guides filtering and reduces emergent misalignment | Narrow misalignment mechanism, not broad benign assistant behavior | `notes/abstract-screening.md` |
| [SAE post-training data engineering](https://arxiv.org/abs/2605.27354) | SAE clustering, difficulty, and quality signals improve mathematical RL data engineering | Source-family filtering and RL utility are not behavior-level SFT removability | `notes/sae-data-engineering.md` |

### Can an SAE feature be treated as a causal concept?

| Record | Direct evidence | Consequence | Note |
|---|---|---|---|
| [Causal SAE audit](https://arxiv.org/abs/2607.12166) | High-cosine decoder matches can be read- or write-inert; exact checkpoint/distribution revalidation is recommended | Raw activation, decoder cosine, and top-context labels cannot be the only feature gate | `notes/sae-data-engineering.md` |
| [Feature absorption](https://arxiv.org/abs/2409.14507) | A lower-cosine child latent can mediate cases where a high-alignment parent latent is silent | Search for false negatives and absorbing latents before treating a feature as complete support | `notes/sae-data-engineering.md` |
| [Steering-vector decomposition](https://arxiv.org/abs/2411.08790) | SAE reconstruction can miss negative projections and out-of-distribution steering vectors | Counterfactual training-push vectors need direct behavioral validation | `notes/abstract-screening.md` |
| [Steering grids](https://arxiv.org/abs/2605.03160) | A top-context label can describe an activation regime rather than the causal axis | Use bidirectional, coefficient-swept, matched-random controls | `notes/abstract-screening.md` |

### Does representation control imply data removability?

| Record | Direct evidence | Consequence | Note |
|---|---|---|---|
| [Persona-feature data attribution](https://arxiv.org/abs/2608.11025) | Causal persona features retrieve semantically relevant web documents, but human prose does not induce clean misalignment while synthetic instruction pairs do | Semantic relevance and steering are insufficient to identify a removable teaching route | `notes/persona-attribution.md` |
| [Persona Features Control Emergent Misalignment](https://arxiv.org/abs/2506.19823) | Model-diffed features steer misalignment and benign SFT can restore alignment | Strong representation control, but no leave-group-out mapping from feature to teaching examples | `notes/persona-attribution.md` |
| [Tracing Persona Vectors Through Pretraining](https://arxiv.org/abs/2605.13329) | OLMo persona directions appear early and persist into post-trained descendants | Broad behavior can be inherited and merely strengthened or elicited by SFT | `notes/persona-attribution.md` |

### What is the causal gold standard for a data effect?

| Record | Direct evidence | Boundary | Note |
|---|---|---|---|
| [Group influence](https://proceedings.neurips.cc/paper/2019/hash/a78482ce76496fcf49085f2190e675b4-Abstract.html) | Exact retraining after group removal shows low individual influence can coexist with large group effects | Convex models and older NLP tasks | `notes/tda-counterfactuals.md` |
| [Datamodels](https://arxiv.org/abs/2202.00622) | Arbitrary-subset retraining supports target-specific counterfactual surrogates | Image models and very large data/computation requirements | `notes/tda-counterfactuals.md` |
| [STRIDE](https://arxiv.org/abs/2606.05165) | Modern LM pretraining/SFT attribution is calibrated against subset-retrained responses | Calibration outcome is not a broad behavior-removal effect | `notes/tda-counterfactuals.md` |
| [Simfluence](https://arxiv.org/abs/2303.08114) | Held-out language-model training runs expose redundancy and order effects missed by scalar additive attribution | Predicts loss trajectories, not concept-conditioned behavior removal | `notes/tda-counterfactuals.md` |

### Does replacement solve what deletion misses?

| Record | Direct evidence | Consequence | Note |
|---|---|---|---|
| [Naive SFT filter failures](https://www.alignmentforum.org/posts/wyZRNgpeiPeRXB6eT/why-do-naive-sft-filters-for-safety-properties-fail) | Swapping teacher rollouts on fixed prompts can change hereditary traits when dropping the same prompt subset does little | Replacement is a different, larger intervention; prompt and teacher provenance must be controlled | `notes/filter-replace.md` |
| [Open distillation of hereditary traits](https://www.alignmentforum.org/posts/WpYFAmJDH3zuAq2ha/open-distillation-of-hereditary-traits-1) | Clean-answer swapping reduces blackmail and censorship transfer where deletion fails | Generic response replacement is not novel | `notes/filter-replace.md` |
| [Warm-response SFT](https://doi.org/10.1038/s41586-026-10410-0) | Meaning-preserving response rewrites change broad persona and accuracy outcomes | Matched response transformation is feasible, but neutrality is an empirical claim | `notes/filter-replace.md` |
| [Subliminal Learning](https://arxiv.org/abs/2507.14805) and follow-ups | Apparently unrelated or faithfully paraphrased responses can transmit traits | Freeze generator provenance and add teacher/hidden-signal controls | `notes/filter-replace.md` |

## Artifact readiness

The checkout already contains an exact OLMo-3 base revision, a multi-million
document attribution manifest, clustered data, probe infrastructure, scalable
attribution methods, and evaluation tooling. Public OLMo-3 SAEs exist for three
layers. The missing piece is the exact broad-behavior speed-run organism: no
public 25,000-example sample manifest, frozen behavior-prompt bank, or dedicated
training package was found in the declared search. A public reconstruction is
possible from the base model and SFT dataset cards, but it is a separate setup
cost and a source of divergence.

## Coverage limits

- The title/abstract screen was automated and had no human adjudication.
- One of four academic seed citation chains was blocked by HTTP 429.
- Forum sources were exact-title and linked-source searched, not formally
  forward/backward chained.
- The newest 2026 records are preprints and may change.
- Abstract-level records support discovery and novelty triage only.
