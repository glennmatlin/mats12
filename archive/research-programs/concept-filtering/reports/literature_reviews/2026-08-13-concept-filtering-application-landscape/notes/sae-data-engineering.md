# Evidence note: SAE-guided post-training data engineering and causal validity

## Scope and evidence policy

- Review protocol read: `reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/review_protocol.md` (landscape; targeted full-text extraction).
- Proposal scratchpad read: `docs/specs/2026-08-13-mats-stream-project-idea-scratchpad.md`.
- Full text read on 2026-08-13 for the exact versions below. The claims are separated into source facts, reviewer interpretation, and limits. The two target papers are preprints; the absorption paper is a NeurIPS 2025 Oral paper as of its v6 record.

## Records and access

### Record A: SAERL

- Canonical citation: Yi Jing, Zao Dai, Jinwu Hu, Zijun Yao, Lei Hou, Juanzi Li, and Xiaozhi Wang. “Guiding LLM Post-training Data Engineering with Model Internals from Sparse Autoencoders.” arXiv:2605.27354v1, 26 May 2026.
- Identifier and source: <https://arxiv.org/abs/2605.27354v1>; source type `preprint`; evidence tier `primary_result_checked`.
- Exact version: arXiv v1, submitted 2026-05-26 17:55:59 UTC. The arXiv record reports no later version.

### Record B: Causal audit

- Canonical citation: Mohamed Abdessalem Bal. “From Geometric Recovery to Causal Validation: A Reproducible Audit of Sparse Autoencoder Features, from Superposition Geometry to Causal Inertness.” arXiv:2607.12166v1, 13 July 2026.
- Identifier and source: <https://arxiv.org/abs/2607.12166v1>; source type `preprint`; evidence tier `primary_result_checked`.
- Exact version: arXiv v1, submitted 2026-07-13 21:18:52 UTC. The arXiv record reports no later version.

### Record C: cited negative result, feature absorption

- Canonical citation: David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, and Joseph Bloom. “A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders.” arXiv:2409.14507v6, 17 November 2025; accepted at NeurIPS 2025 (Oral).
- Identifier and source: <https://arxiv.org/abs/2409.14507v6>; source type `peer-reviewed paper/preprint record`; evidence tier `primary_result_checked`.
- Citation path: Record B identifies Chanin et al. as the closest prior work on feature absorption in its §2 “Firing failures and proxy-metric gaps” and cites arXiv:2409.14507 as reference [5]. This is the selected directly relevant causal negative result.

## Relevance to the review

- Records A and B directly bear on whether SAE-derived internal coordinates can guide post-training data engineering and whether SAE feature identity/causal validity is strong enough for Proposal 1’s behavior-positive/behavior-negative selection and steering gates.
- Record C is a negative control against treating a high-alignment or apparently monosemantic SAE latent as a complete behavioral support unit. It uses causal ablation, not training-data removal, so it informs feature validity rather than directly settling filterability.

## Record A source facts: SAERL

| Claim or result | Location anchor | Method/data support |
|---|---|---|
| SAERL maps three SAE-derived properties to three operations: SAE-space clustering and moderate batch mixing for diversity, a difficulty proxy for easy-to-hard curriculum ordering, and a source-labeled quality probe for data filtering. | Abstract; §2 overview; Fig. 2 (pp. 1, 2-4) | Token-level prompt/solution SAE activations are pooled; the representation has 960 SAE features plus 26 shallow metadata features (§3.1, p. 4). |
| SAE activations predict external DeepMath topic labels above majority baselines at three granularities: 54.6 vs 31.8 for L2, 37.7 vs 17.2 for L3, and 26.6 vs 7.5 for leaf topics. | Table 1, §2.1 (p. 3) | The labels are dataset metadata, so this is a non-circular predictive test, not a causal feature test. |
| SAE activations predict PRM800K quality better than metadata: RMSE 0.2007, MAE 0.1608, Pearson 0.3715 versus metadata Pearson 0.2100. | Table 3, §2.3 (p. 4) | A ridge regressor predicts a sample-level score formed from PRM step labels; the quality probe later uses source/distribution labels. |
| The data-engineering experiment uses a single SAE trained on layer-27 activations of Qwen3-1.7B as the shared encoder for Qwen2.5-Math-1.5B and Qwen2.5-Math-7B, with DeepMath-103K and GRPO/DAPO. | §4.1 “Models and training” (p. 5); Table 4 | This is cross-model-family/scale transfer during mathematical RL post-training; it is not a base-to-SFT or LoRA-SFT transfer test. |
| At step 900, SAERL-G (GRPO) averages 52.4 on Qwen2.5-Math-1.5B versus GRPO 49.4, and 61.9 on Qwen2.5-Math-7B versus GRPO 59.9; SAERL-D (DAPO) averages 52.5 on 1.5B. | Table 4, §4.2 (p. 5) | The six benchmarks are GSM8K, AMC, MATH, MinervaMath, OlympiadBench, and AIME; AIME uses Pass@8 and the others Avg@8. |
| SAERL reaches the shared target in fewer steps: average 380 versus GRPO 470 on 1.5B for SAERL-G, 206 for SAERL-D, and 173 versus GRPO 193 on 7B for SAERL-G. | Table 5, §4.3 (p. 6) | “Target” is defined from the minimum final accuracy among methods; this is convergence efficiency, not a causal data-removal effect. |
| In a mixed noisy pool of 210,043 examples (103,022 DeepMath plus 107,021 NuminaMath-1.5), a fixed SAE-only probe gets holdout ROC-AUC 0.9911 and AP 0.9910; P95 threshold gives 95.37% DeepMath purity and 95.46% recall, while direct top-50k gives 99.92% purity and 48.50% recall. | §4.4 and Table 6 (p. 6) | This is source-family recovery from a mixed pool. It does not retrain after removal or show a held-out behavior change caused by selected data. |
| Removing difficulty sorting causes the largest ablation drop; removing clustering plus mixing also drops performance, while the full joint pipeline is best among the listed variants. | §5.1 and Table 7 (p. 7) | The ablations isolate SAERL components only within the RL curriculum; they do not test SAE-feature suppression/amplification or feature identity after SFT. |
| Moderate mixing is non-monotonic: mix8 is best for the step-800/threshold analysis, while mix16 and mix32 degrade final accuracy and convergence despite higher measured diversity. | §5.2 and Fig. 3 (pp. 7-8); Table 8 | The authors interpret this as a coherence/coverage trade-off; it is an optimization observation, not proof that SAE distance is causal. |
| SAE clusters do not reproduce the human topic taxonomy (purity 0.1095, NMI 0.0881, ARI 0.0394), although inspected clusters show structure in format, solution profile, difficulty, and some mathematical areas. | §5.4 and Appendix D.1, Table 10 (pp. 8, 15) | The semantic audit is partly GPT-5.4-generated and manually reviewed; cluster meaning is descriptive and not one-to-one with topics. |
| The authors explicitly state that transfer to code-centric RL, agentic RL, tool use, multi-step decisions, and general instruction following remains to be established. | §8 “Limitations,” “Domain scope” (p. 9) | This directly bounds any inference from the mathematical-RL transfer result to broad assistant-like SFT behaviors. |
| SAERL is not fully unsupervised: the difficulty proxy uses a 3,000-example labeled subset and the quality probe uses source/distribution labels. | §3.3-3.4; §8 “Limited supervision” (pp. 4-5, 9) | The claimed reduction in supervision is not elimination of labels. |
| The paper assumes SAE-space proximity is a proxy for semantic similarity and gradient coherence and says this does not prove a causal relationship; direct gradient-level measurements would be needed. | §8 “Theoretical scope” (p. 9); Appendix A | This is an author-stated limit on mechanistic/causal interpretation. |

### Record A reviewer interpretation

SAERL establishes that SAE-derived coordinates can be useful *correlational and operational signals* for mathematical RL data ordering, batching, and noisy-source filtering. It is the clearest direct collision with any novelty claim that says only “use SAE features to filter post-training data.” It does not establish that an SAE feature is a causal teaching unit, that top-ranked examples are causally removable, or that a base-model SAE remains valid after LoRA SFT. Its source-recovery table is especially important to keep separate from behavior-level retraining: high purity/recall recovers a corpus family, not a behavior change after deleting that family.

The transfer claim is narrower than Proposal 1’s required gate. SAERL transfers a Qwen3-1.7B layer-27 SAE into Qwen2.5-Math model scales/family during mathematical RL; the authors themselves leave instruction-following transfer open. Therefore the evidence for base-to-SFT transfer is **not supported**. It is an empirical gate, not a prior supplied by this paper.

### Record A artifacts and reproducibility

The paper says models, datasets, benchmarks, and software frameworks are public and promises documentation of processing, SAE extraction, curriculum construction, filtering, and evaluation (§8 “Documentation of Artifacts,” p. 9). The arXiv v1 paper does not link a SAERL-specific repository, checkpoint, processed pool, or run manifest. The authors’ public OpenSAE toolkit (`https://github.com/THU-KEG/OpenSAE`) is a separate SAE framework/model release, not evidence that the SAERL experiment code and outputs are released. Artifact status is therefore **partial/uncertain** for reproduction of the reported pipeline.

## Record B source facts: causal audit

| Claim or result | Location anchor | Method/data support |
|---|---|---|
| The paper distinguishes decoder-geometry alignment from encoder-activation behavior and argues that unsigned cosine recovery cannot certify the latter. | Abstract; §1 Introduction; Table 6 (pp. 1-2, 18) | A causal test must intervene through encode/decode and a behavioral readout, not only compare decoder directions. |
| In a controlled toy model, the authors audit every correlationally recovered feature with ablation and sign-correct steering. The degraded TopK k=13 SAE has 17/22 matched features with zero ablation specificity (77%); the well-trained k=4 SAE has 2/22 (9%), including matches at cosine above 0.9999. | Abstract; §6.1-6.3 (pp. 1, 9-10) | The matched atom never fires on 500 feature-present samples in the inert cases, so zero ablation is caused by an already-zero code rather than a weak effect. |
| The original intervention table reports median ablation specificity 168.8 for the good SAE and exactly 0 for the bad SAE; pooled cosine/log-specificity correlation is r=0.657. | Table 4, §6.2 (p. 10) | Better geometry predicts better causal effect on average, but the relationship is not sufficient and fails badly for degraded dictionaries. |
| Under the deterministic re-audit, the good SAE has 2/22 inert (9%) and the bad SAE 3/17% with a fourth case within a ±1 count band; median ablation specificity is 133.8 [107.0, 167.4] for good and 68.3 [22.0, 105.9] for bad. | Table 5, §8-9 (pp. 13-14) | These are reference-environment results with seeded bootstrap intervals; the bad count is boundary-sensitive across BLAS builds. |
| The inert matches are not only marginal: bad-SAE inert cosines are 0.986, 0.994, and 0.9995; good-SAE inert cosines are 0.9997 and 0.9998. | §9 and Fig. 4 (p. 14) | This is a direct counterexample to “near-perfect cosine implies causal feature identity.” |
| Five antipodal pairs explain structural inertness. The same atom can be read-inert (never selected for one feature) yet write-useful: correct-sign steering gives specificities 143-310 while ablation remains exactly zero. | §10.1-10.2 (pp. 15-16); Abstract | The result is a read/write dissociation, not a claim that all steering fails. It shows that monitoring and steering require separate validation. |
| The production harness audits GPT-2-small residual-stream SAEs against 83 hand-written concepts. Seven pairs pass cosine >=0.5; one of seven is firing-inert (14%), with ablation median 1.63 [0.05, 5.34] and steering median 2.23 [0.69, 3.54]. | §11.1 (p. 17) | The real-model direction is a difference-in-means probe, not ground truth; the paper explicitly says these rates are not comparable one-to-one with toy 77%/9% figures. |
| The paper recommends read validation for monitoring, write validation for steering, re-running validation after SAE retraining, and pinning the SAE checkpoint to the exact model checkpoint and data distribution. | §12 “Practical Guidance” (p. 18) | This is directly relevant to base-to-SFT reuse: causal feature identity is checkpoint- and distribution-specific in the evidence presented. |
| The strongest quantitative evidence is toy-scale: interactions among simultaneously active features are untested; the read/write split uses five antipodal pairs across two SAEs and one deterministic seed; real prompts undersample concepts and use proxy directions. | §13 “Limitations” (p. 19) | The paper supports a methodological gate, not a transferable inert-rate prior for OLMo-3. |

### Record B reviewer interpretation

The causal audit is a strong methodological warning for Proposal 1’s requirement that a candidate feature “must” pass bidirectional suppression/amplification before selecting data. It supports that gate and the Proposal 4 distinction between representation-level controllability and a separate training-data-removability question. It does **not** show that a causally steerable feature is removable by deleting a small SFT subset; no training-data intervention or retraining is performed. Its production result is a feasibility signal for running a causal audit on a real SAE, not evidence that the same latent survives base-to-SFT transfer.

The paper’s most relevant negative result is not “steering never works.” Instead, unsigned geometric recovery can be causally inert in the read direction, while the same atom is steerable in the write direction. Treating an SAE activation as both a reliable selector and a reliable steering handle without direction-specific validation is unsupported.

### Record B artifacts and reproducibility

The paper links `https://github.com/mohamed-bal/sae-causal-audit` and `https://github.com/mohamed-bal/superposition-to-monosemanticity`. It reports a 37-test suite, a deterministic reproduction pipeline, hash-verified CI, a real-model harness, and committed `results/*.json` hashes (Abstract; §7; §8; §14). The arXiv record links the audit repository. A live `git ls-remote` check on 2026-08-13 found public `main` refs for both repositories. This is comparatively strong artifact availability, but the authors’ own limits still restrict quantitative generalization to larger, post-SFT organisms.

## Record C source facts: feature absorption (cited causal negative)

| Claim or result | Location anchor | Method/data support |
|---|---|---|
| Feature absorption occurs when a seemingly monosemantic latent fails to fire on cases where its concept is present, while a more specific child latent absorbs part of the feature direction. | Abstract; Fig. 1; §1 (pp. 1-2) | The toy “starts with S”/“short” hierarchy preserves reconstruction while using fewer active latents, creating arbitrary false negatives in the parent latent. |
| In a hierarchical toy model, the parent-tracking latent has a hole in recall and the child latent’s decoder absorbs the parent direction; the paper proves absorption lowers SAE loss under the stated hierarchy. | §3, Fig. 2b, §3 proof (p. 3) | This is a mechanism-level counterexample, not an artifact of noisy labels. |
| On Gemma-2-2B first-letter ICL, the layer-3 16k-width L0=59 latent 6510 appears to track “starts with S” with F1=0.81 but fails to activate on `_short`; token-aligned latent 1085 activates instead. | §5.2 and Fig. 6 (pp. 5-6) | The experiment uses held-out first-letter prompts and residual-stream activations at the target token. |
| Latent 1085 has only cosine 0.12 with the “starts with S” probe versus 0.52 for 6510, but activates at about five times the magnitude on `_short`; it has the largest negative ablation effect. Removing the probe component from 1085 removes that causal effect. | §5.2 and Fig. 7 (p. 6) | The lower-alignment latent is the causal mediator for the missed positive case, while the high-alignment parent latent is inactive. |
| Applying a conservative absorption metric to Gemma Scope SAEs and additional Qwen2-0.5B/Llama-3.2-1B SAEs finds absorption in all tested model/SAE settings; absorption increases with higher sparsity and SAE width. | §5.3 and Fig. 9 (pp. 8-9) | The metric requires one dominant absorber and zero activation of all parent/split latents on a false-negative, so the reported rate is an under-estimate by construction. |
| The authors state that changing SAE size or sparsity is insufficient to solve the issue and that absorption is an obstacle to high-stakes classification and sparse-circuit discovery. | Abstract; §7 Discussion/Conclusion (pp. 1, 9) | This directly limits treating one SAE feature as an exhaustive behavior-support set. |

### Record C reviewer interpretation

Absorption is the strongest cited prior negative result for this review because it combines a clear causal intervention with a direct failure of the “apparently interpretable latent = complete behavior coordinate” assumption. It supports auditing false negatives and checking alternative/absorbing latents before using SAE activations to rank SFT examples. It is methodologically comparable to the causal audit on the narrow point that decoder/probe alignment can fail to predict encoder firing and causal mediation, but it is not comparable on data filtering: it uses first-letter ICL and latent ablation, not SFT retraining or document deletion.

### Record C artifacts and reproducibility

The paper links an interactive explorer at `https://feature-absorption.streamlit.app` and code at `https://github.com/lasr-spelling/sae-spelling` (§1). The paper is an accepted NeurIPS 2025 Oral paper, but its main evidence remains task-specific and metric-limited as described below.

## Comparable support and counterevidence

- **Support for useful internal signals:** SAERL’s matched RL experiments show operational gains and source-family filtering in mathematical reasoning, with a cross-model/scale transfer configuration (Record A, Tables 4-6). This supports testing internal coordinates as *candidate engineering signals*.
- **Counterevidence against unvalidated feature identity:** The causal audit finds 77% inert matches in a degraded toy SAE and 9% in a well-trained one, including cosine approximately 1.000; absorption finds causal mediation in a low-cosine child latent when the high-cosine parent latent is silent (Records B-C, §6/§9-10 and Record C §5.2). These findings oppose using raw SAE activation or decoder cosine as a sufficient causal selector.
- **Why the results are not a direct contradiction:** SAERL optimizes mathematical-RL outcomes and does not claim causal feature recovery; Records B-C test toy/first-letter/probe-intervention validity and do not retrain after selecting or removing data. The populations, interventions, and outcomes differ. The appropriate synthesis is “SAERL demonstrates utility under a restricted task; causal validity and base-to-SFT transfer remain unproven gates,” not “SAERL is refuted.”
- **Unresolved disagreement:** SAERL’s abstract says one SAE transfers effectively across model families/scales, while the causal audit recommends re-validating after every SAE/model/data-distribution change. The apparent disagreement is mainly about what “transfer” means: downstream data-engineering utility under Qwen mathematical RL versus preservation of feature-level causal identity across checkpoints. No cited record tests the base-to-LoRA-SFT transfer required by Proposal 1.

## Proposal impact and novelty decision

### Proposal 1: Explain or rescue filtering with causal SAE concepts

**Decision: PARTIALLY COLLIDING, not already done.** SAERL already occupies the broad “SAE-derived internal signals for post-training data filtering/engineering” space, and the scratchpad already identifies Concept Influence as a collision for SAE concept filtering. Proposal 1 remains distinct if its claim is narrowed to broad/persona-like SFT behaviors in the OLMo organism, a base-to-SFT transfer gate, bidirectional causal feature validation, and a matched data-removal/retraining test. None of Records A-C performs that combination. A proposal phrased only as “filter SFT examples with SAE concepts” would be duplicative.

**Base-to-SFT SAE transfer: NOT SUPPORTED.** Record A transfers a Qwen3-1.7B SAE to Qwen2.5-Math model scales in mathematical RL, and explicitly leaves general instruction-following transfer open. Record B says feature causal relationships are properties of an exact trained checkpoint and data distribution and recommends re-validation after retraining. Record C shows that even within one checkpoint, latent identity can be absorbed and have arbitrary false negatives. The public OLMo-3 base SAE therefore supplies a candidate initialization, not evidence of transfer to the speed-run LoRA-SFT organism.

### Other affected proposals

| Proposal ID | Effect of this evidence |
|---|---|
| Proposal 2 | Supports the mechanism-map framing, but warns that “internal sparsity” or SAE concentration must be treated as a descriptive predictor until causal validation and matched retraining are done. |
| Proposal 3 | Partially collides with SAERL’s SAE selector/quality-filtering operation; its novelty survives only in the broad SFT behavior organism, causal feature gate, and retraining outcome. |
| Proposal 4 | Strongly supported as a useful distinction: causal steering/control and data removability are separate axes. The causal audit supplies read/write validation precedent; no record supplies the removability axis. |
| Proposal 5 | Weakens any assumption that base-versus-SFT feature birth/death or feature identity can be read from an unvalidated SAE; exact-checkpoint re-auditing is required. |
| Proposal 6 | Warns that SAE-feature clusters may not be complete or disjoint support units because absorption and atom collisions can hide or merge support. Cluster-complete removal needs a causal audit and dose-matched retraining. |
| Proposal 7 | Dense residualization avoids some SAE failure modes, but any SAE-derived assistantness control still needs direction-specific causal validation; the cited records do not test the persona hypothesis. |
| Proposal 9 | Supports the proposal’s faithfulness ladder: proxy scores and feature geometry should be compared against exact counterfactual retraining, which none of Records A-C performs. |

## Methodological limits and open gates

- SAERL reports no uncertainty across training seeds in the main tables and uses one SAE configuration (Qwen3-1.7B layer 27); the paper’s own limits restrict transfer claims to the tested mathematical-RL setting.
- SAERL quality filtering is source-label recovery, not behavior-level filterability. It does not remove a selected fraction, retrain an SFT model, or evaluate broad assistant behaviors.
- The causal audit’s strongest numbers come from small toy dictionaries, 22 well-represented features, two SAE configurations, and one deterministic seed; interactions among simultaneously active features are untested.
- The causal audit’s real-model harness substitutes a linear-probe direction for unavailable ground truth, uses one GPT-2-small hook layer, 83 hand-written concepts, and only seven matched pairs above its lowered cosine threshold.
- Absorption’s rate metric is deliberately conservative and uses ablation to establish causality, which limits it to layers where the chosen token-level intervention can affect the output; it does not show data-removal effects.
- No record tests OLMo-3 base SAEs after speed-run LoRA SFT, no record maps SAE features to the specific SFT examples that created them, and no record separates controllable-but-distributed from controllable-and-removable behavior by retraining.

## Bottom line

SAERL makes “SAE-guided post-training data engineering” a real prior rather than a blank novelty claim, while the causal audit and feature-absorption result make raw SAE geometry/activation an unsafe causal assumption. Proposal 1 is therefore **partially colliding but still open** only when its contribution is the causal representation-versus-removability test on broad SFT behavior; base-to-SFT SAE transfer remains an explicit empirical gate.

DONE_WITH_CONCERNS
