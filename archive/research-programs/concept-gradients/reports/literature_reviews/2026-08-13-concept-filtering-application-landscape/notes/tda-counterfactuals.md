# Full-text evidence: counterfactual retraining, group influence, redundancy, coverage, and dose

Review: `Concept-level data filtering and filterability mechanisms for a short research application` (landscape mode)

Extraction date: 2026-08-13

Protocol and proposal source: [`review_protocol.md`](file:///Users/glenn/dev/social-data-attribution/reports/literature_reviews/2026-08-13-concept-filtering-application-landscape/review_protocol.md) and [`2026-08-13-mats-stream-project-idea-scratchpad.md`](file:///Users/glenn/dev/social-data-attribution/docs/specs/2026-08-13-mats-stream-project-idea-scratchpad.md). The nine records below were read as full PDFs, including appendices where present. Page numbers are PDF pages; section, figure, and table labels are the primary anchors. Source facts are separated from reviewer interpretation and limits.

## Bottom line

The central methodological gap survives, but the strongest versions of Proposals 8 and 9 are already substantially prefigured. Exact or near-exact counterfactual calibration is established by Datamodels, Simfluence, STRIDE, and Koh et al.; Concept Influence and Persona Vectors establish concept-conditioned ranking and group filtering; and several papers already measure concentration or support. What is not yet demonstrated is a modern causal-language-model study that (i) compares concept/gradient/probe TDA scores against matched leave-group-out behavioral effects, (ii) varies behavior prevalence and removal dose while separating selector accuracy from causal support removed, or (iii) tests whether coverage across held-out behavior queries predicts a retraining effect.

**Proposal verdicts.** Proposal 8 has a **partial novelty collision**: dose and group-removal ladders are old, but a prevalence-by-dose phase diagram for broad SFT behavior remains open. Proposal 9 has a **strong methodological collision but a defensible narrow gap**: exact counterfactual calibration is established, while a joint MAGIC/TrackStar/EKFAC/probe/concept comparison on one modern causal-LM behavior organism is not found here. Proposal 10 has **no direct collision**: existing “coverage” metrics measure influence mass, dataset coverage, or target-specific support, not the fraction/entropy of behavior-query families covered by a selected set.

## Exact source versions and artifact receipts

| Record | Exact version read | PDF SHA-256 | Pages | Artifact status checked |
|---|---|---|---:|---|
| Kowal et al., [*Concept Influence*](https://arxiv.org/abs/2602.14869v1) | arXiv:2602.14869v1, 2026-02-16 | `5ea3b3d77ebaa902e353078e18ba51fe293815806c6d225451096d8fec9f75bc` | 23 | Paper and [FAR.AI post](https://www.far.ai/blog/concept-data-attribution-02-2026) checked; no paper-specific code/checkpoint URL found. |
| Guu et al., [*Simfluence*](https://arxiv.org/abs/2303.08114v1) | arXiv:2303.08114v1, 2023-03-14 | `d45d205b94d102c1cfb2f23060039d34578f864f74c36231d00621cd5dd52135` | 19 | arXiv page and PDF checked; no code/data URL found. |
| Dagli et al., [*STRIDE*](https://arxiv.org/abs/2606.05165v1) | arXiv:2606.05165v1, 2026-06-03 | `78981edaf2f73753f03659110faa18f5a8d4c79c5b6a1946e8a8da813a953505` | 40 | [Project page](https://stride-tda.github.io) and [HF model card](https://huggingface.co/CausalNLP/stride-nanochat) checked. Models/operators and [LDS data](https://huggingface.co/datasets/rishitdagli/stride-lds) are public; project-page code button is a `#` placeholder. |
| Koh et al., [*On the Accuracy of Influence Functions for Measuring Group Effects*](https://proceedings.neurips.cc/paper/2019/hash/a78482ce76496fcf49085f2190e675b4-Abstract.html) | NeurIPS 2019 | `daab39de392df1187702cfb102d12bb37ede247148d235890730ed9e5944a60c` | 11 | [GitHub](https://github.com/kohpangwei/group-influence-release) and [CodaLab worksheet](https://worksheets.codalab.org/worksheets/0xfed2ae0b9e5b44b7a1af8096365592a5) are public. |
| Karthikeyan and Søgaard, [*Revisiting Methods for Finding Influential Examples*](https://arxiv.org/abs/2111.04683v1) | arXiv:2111.04683v1, 2021-11-08 | `56e73644b6bbd29aa15e409e0a47b0b47a7772f8eeeab4089c051ebfc689e0b8` | 8 | No code/data URL found in the paper or arXiv page checked. |
| Ilyas et al., [*Datamodels*](https://arxiv.org/abs/2202.00622) | ICML 2022 / arXiv:2202.00622 | `90eab4fd0cdcaec55a1fd08cb135e843e730af5b35fb72015f7d44a0f55f048d` | 63 | [Code](https://github.com/MadryLab/datamodels) and [data README](https://github.com/MadryLab/datamodels-data) are public. Data is requester-pays S3; the README estimates 4,155 GB and about USD 53 excluding logits. |
| Chen et al., [*Persona Vectors*](https://arxiv.org/abs/2507.21509v1) | arXiv:2507.21509v1, 2025-07-29 | `a61aa5783148114da67c2cc9c46e1974b6e81406e5fbdde1af15b4e74113d015` | 60 | Official [code](https://github.com/safety-research/persona_vectors) is linked in the paper. |
| Betley et al., [*Emergent Misalignment*](https://arxiv.org/abs/2502.17424) | ICML 2025 / arXiv:2502.17424 | `bc24cfee99e5cbda057ff21aad4ccd6f15e10d810e88fcebdd5337203ee1edbe` | 26 | Official [code/data repository](https://github.com/emergent-misalignment/emergent-misalignment) and [project page](https://www.emergent-misalignment.com/) are public. |
| Grosse et al., [*Studying LLM Generalization with Influence Functions*](https://arxiv.org/abs/2308.03296v1) | arXiv:2308.03296v1, 2023-08-07 | `74cc21aa67ba7f4309982321f1ef76b26e60c3b9359832696bef918df4cff66e` | 119 | PDF and arXiv page checked; no implementation/data URL found, and the scanned pretraining corpus is not public. |

## Record A: Concept Influence (Kowal et al., arXiv:2602.14869v1)

### Source facts

- The paper replaces a single test-output gradient with a semantic target vector: a linear probe, SAE latent, or other interpretable activation direction. Its core score is a Hessian-weighted alignment between that target direction and a training example's gradient (`§§2-3`, pp. 2-5). It also groups examples by active SAE features and scores groups rather than only points (`§4.1.3`, pp. 9-10).
- The emergent-misalignment experiment uses Qwen2.5-7B and Llama3.1-8B, four synthetic datasets (Misaligned Opinions, Bad Medical Advice, Insecure Code, GSM8K Mistakes), 50% benign/50% misaligned examples, and one epoch of LoRA SFT. Evil and sycophancy are judged after training (`§4.1`, pp. 6-7; Fig. 2).
- Filtering top-ranked examples changes the retrained trait. Training on only the top 10-20% most misaligned examples can make the model more evil than training on the full mixed dataset; in Insecure Code the full run is near 1% evil while the influence-selected subset is about an order of magnitude higher (`§4.1.1`, pp. 7-8). Projection Difference is strongest in-domain but can fail under shift, including GSM8K-to-Evil (`§4.1.2`, p. 8).
- The real-data experiment SFTs Qwen2-7B-Base on one epoch of English OASST1, which has more than 10,000 conversation trees and roughly 2-3% toxic data. The base model has evil score about 0.5 and MTBench 38; full SFT reaches about 2.3 and 67. Selecting the safest 5% by Vector Filter yields evil about 0.8 while matching full-data capability about 67 (`§4.1.2`, pp. 10-11; Fig. 5).
- SAE group filtering removes examples belonging to top or bottom concept groups at fractions `n={0.01, 0.10, 0.20, ...}`. In the Misaligned Opinions case, the top 20% Concept Influence concepts cover about 70% of training examples, versus about 50% for influence-function concepts; after more than 50% of examples are removed, Concept Influence has higher precision at finding misaligned examples (`Appendix D.2`, pp. 15-16; Fig. 15).
- Runtime is not free: for 1,000 examples, Concept Influence is reported at about 1,170 seconds, versus 57 seconds for Vector Filter and 1,161 seconds for influence functions; Hessian construction is excluded (`§4.1.2`, p. 8; Table 1). The vector methods are activation-alignment approximations and do not themselves compute a group-removal counterfactual.

### Interpretation, collision, and limits

Concept Influence is already a direct precedent for “semantic concept plus data filtering,” SAE-defined groups, and a fixed-size removal ladder. Therefore “use SAE features to filter” is not a sufficient novelty claim for Proposals 1, 3, or 6. Its strongest unresolved limitation is also the opening for the scratchpad: the paper does not compare a concept score with a true leave-group-out behavioral effect across matched groups, does not identify an optimal or stable cluster partition, and does not measure query-family coverage. The authors explicitly list principled group attribution, base-versus-finetuned comparison, and scaling as future work (`§6`, p. 12).

The paper's group “coverage” is coverage of the **training dataset by selected concepts**, not coverage of evaluation behaviors. Its 70% versus 50% result cannot establish Proposal 10's hypothesis. Its OASST1 and EM retraining results show that selection can change a trait, but they do not establish that the selected examples are uniquely causal or that a small dose is sufficient once redundant support is controlled.

### Proposal IDs affected

- **P1/P3:** `NOVELTY_COLLISION` for concept-conditioned/SAE filtering as the headline. A defensible extension requires a broad OLMo behavior, causal feature gate, matched selector baselines, and retraining-level outcome.
- **P4:** Supports separating representation-level semantic control from data-level removability; Concept Influence does not perform the two-axis test.
- **P6:** Direct precedent for SAE-group filtering, but no cluster-complete removal under a fixed document budget and no redundancy test.
- **P8:** Provides an existing 1%, 10%, 20%... group-removal ladder and 5% OASST1 selection point, but not a prevalence-by-dose phase diagram or a selector-precision/support-fraction/downstream-change decomposition.
- **P9:** Provides retraining after filtering but not exact group-effect calibration against multiple TDA quantities.
- **P10:** Dataset coverage is adjacent but not query coverage; no coverage/entropy predictor is evaluated.

## Record B: Simfluence (Guu et al., arXiv:2303.08114v1)

### Source facts

- Simfluence starts from the failure of scalar additive influence: redundancy, training order, and curriculum can make interactions submodular or supermodular. A redundant set can split credit across many examples, while a curriculum can make a pair jointly useful (`§1`, pp. 1-2).
- The proposed simulator predicts a test example's loss trajectory for a curriculum. Simfluence-Linear uses a Markov update with additive and multiplicative terms, so redundancy/order are represented; TracIn-Ideal is its additive special case and influence functions are another hypothetical-loss special case (`§§2-4`, pp. 2-6).
- Each example has two learned parameters, so at least `2n` observed training steps are required for `n` examples; the experiments use roughly `20n-60n` steps. A separate simulator is learned for each test example and cannot simulate examples never observed during fitting (`§3.3`, pp. 5-6; §7, p. 13).
- In LLM fine-tuning, T0 3B/T5-LMA XL/IA3 use RTE, COPA, and Winogrande. RTE/COPA runs train on 64 of 100 examples for four epochs; Winogrande runs train on 1,024 of 1,536 examples for 1.5 epochs. Thirty-two runs are split into 20 fit, 2 validation, and 10 held-out runs (`§5`, pp. 9-10).
- On held-out IA3 T0 3B runs, Simfluence-Linear reaches final Spearman 0.886±0.033 on COPA and 0.887±0.080 on RTE, versus TracIn-CP all-steps 0.469±0.077 and 0.034±0.200. Its MSE is 1.503±0.494 and 0.819±1.222, versus 5.792±0.331 and 3.733±0.275 (`Table 1`, p. 10). Across setups, final Spearman is roughly 0.628-0.813 for T0 full tuning and 0.383-0.565 for T5-LMA full tuning (`§5`, p. 11).
- The authors state that Simfluence-Linear models redundancy but not longer-range supermodular interactions, and ignores optimizer state, parameters, and learning rate (`§7`, pp. 13-14).

### Interpretation, collision, and limits

Simfluence is a direct **language-model counterfactual calibration precedent**: it compares attribution models against actual held-out training runs, rather than against “looks relevant” labels. It is not yet Proposal 9's experiment. Its target is loss trajectory under an observed curriculum, not a concept-conditioned behavioral effect after deleting a group; it does not compare MAGIC, TrackStar, EKFAC, dense probes, and SAE/concept scores on one modern causal-LM SFT organism. Its high data requirement and one-simulator-per-test-example design also do not solve the 20-hour scale problem without a smaller, carefully chosen calibration stage.

Simfluence is especially strong evidence for Proposal 6's redundancy premise: additive scores can under-credit interchangeable examples, and order can change the effect. It is not a solution to cluster-complete removal because its simulator does not provide an optimal semantic group partition and cannot represent all higher-order interactions.

### Proposal IDs affected

- **P4/P6:** Strong mechanism support for controllability/removability separation and redundancy-aware group effects.
- **P9:** `PARTIAL_COLLISION`: exact held-out LLM-run calibration exists, but concept-conditioned behavioral group effects and the proposed baseline ladder remain open.
- **P10:** No query-coverage metric; the simulator is test-example-specific and therefore reinforces the need to measure cross-query coverage explicitly.

## Record C: STRIDE (Dagli et al., arXiv:2606.05165v1)

### Source facts

- STRIDE treats subset retraining as the causal gold standard, learns low-rank activation-space steering operators to mimic subset responses, and uses sparse recovery to infer individual influence (`§§1, 3-4`, pp. 1-6). The paper's LDS metric compares predicted subset responses to held-out subset-retrained responses.
- Pretraining experiments use Nanochat at 286M, 537M, 897M, and 1.38B parameters, 256 subsets, and 500 held-out queries. STRIDE LDS is 0.1581, 0.1792, 0.1598, and 0.1671 respectively, with 9.9 hours runtime at 1.38B versus 52.3 hours LoGRA and 116.1 hours AirRep (`§5.2`, Table 1, pp. 7-8).
- SFT experiments use Qwen2.5-0.5B on FLAN 100K (6,520 queries, 100 subsets), Alpaca 51,760 (500 queries, 10 subsets), Tulu 100K (500 queries, 100 subsets), and SafeRLHF 251,963 (500 queries, 5 subsets). LDS is 0.2426, 0.1611, 0.1932, and 0.3995, respectively (`§5.3`, Table 2, pp. 8-9). AirRep is higher on FLAN and SafeRLHF, so STRIDE is not uniformly best.
- STRIDE's influence-distribution diagnostic reports near-perfect pretraining sparsity and top-10% coverage 1.0, but SFT sparsity only 60-68%. Top-10% coverage is 0.8153 FLAN, 0.8973 Alpaca, 0.7977 Tulu, and 0.1796 SafeRLHF, with median active sets 32,873, 16,732, 36,014, and 80,672 (`Appendix B.2`, Table 6, p. 22). The authors interpret this as broader SFT influence support.
- Controlled MNIST/FashionMNIST/Parkinsons/CIFAR experiments explicitly retrain masked subsets and compare predicted versus true responses; STRIDE's top-k removals are often more actionable than random, but TRAK or LoGRA wins in some settings (`Appendix B.6`, pp. 24-25). This is a direct causal check, but not an LLM behavior study.
- The method assumes local linearity and additive influence. The paper reports that dense Bernoulli subset matrices violate the sparse-expander regime and degrade LDS, and that extreme distribution shift or highly nonconvex memorization can break the approximation (`§4`, p. 6; `§6`, p. 23; Appendix B.1).

### Interpretation, collision, and limits

STRIDE is the most direct modern LLM precedent for Proposal 9's “faithfulness ladder”: it uses explicit subset retraining as LDS ground truth, includes SFT, and reports support-concentration diagnostics. It still does not calibrate concept-conditioned group scores against a held-out **behavioral** effect, and its SFT top-10% coverage is influence-mass concentration, not the fraction of behavior queries covered by a selected set. The SafeRLHF result is a useful counterexample to claiming that a sparse support prior transfers across SFT datasets.

The public HF release contains operators, base checkpoints, subset memberships, and LDS data, making a replication or calibration smoke test feasible. It does not release the project-page code button target; the released artifacts should not be treated as a complete training pipeline.

### Proposal IDs affected

- **P6:** Supports a concentrated-versus-broad support contrast and the need to test cluster-complete removal; it does not test semantic redundancy directly.
- **P8:** Supplies SFT support-concentration and dose-independent evidence that support can be broad; no prevalence-by-removal-dose experiment.
- **P9:** `STRONG COLLISION` for exact subset-retraining LDS calibration as a general methodology, but the concept-conditioned, behavior-level baseline ladder remains open.
- **P10:** Top-10% coverage is an adjacent but non-equivalent metric; it does not predict across behavior-query families.

## Record D: Group influence and exact removal (Koh et al., NeurIPS 2019)

### Source facts

- Koh et al. define the actual group effect by retraining after removing a subset and compare it to first-order influence. On real datasets, group rankings correlate at Spearman 0.8-1.0, but influence often underestimates the absolute effect (`Abstract`, `§1`, pp. 1-2).
- They construct about 1,700 coherent/random groups per dataset, with removal fractions from 0.25% to 25%, using feature values, feature/gradient clusters, random points, and high-positive/high-negative influence points (`§3`, pp. 3-5). The models are L2-regularized logistic/softmax regressions, not neural LMs.
- In CDR labeling functions, influence tracks LF coverage rather than precision (Spearman 1 with actual test-loss effects). The authors warn that this is not a causal claim because all-positive/all-negative labeling changes class balance. In MultiNLI, crowdworker influence is not predicted by example count; removing and retraining a prolific 35,000-example worker made performance worse, with genre-specific effects (`§5`, pp. 8-9).
- The paper gives a theoretical counterexample where influence and actual/Newton group effects can have opposite signs for test prediction; its favorable correlation/underestimation behavior is not universal (`§4.4`, p. 7).

### Interpretation, collision, and limits

This is the foundational direct precedent for Proposal 6's claim that low individual influence can coexist with a large group effect, and for Proposal 8's use of matched group-removal doses. It also provides a caution for Proposal 10: a “coverage” variable can predict apparent influence for confounded reasons and is not automatically causal. Because the model is convex linear and the labels/tasks are older NLP, it supports the causal design logic rather than the modern-LLM empirical claim.

### Proposal IDs affected

- **P6:** Strong support for group effects, redundancy, and cluster-complete tests; no SAE/concept partition or LLM behavior.
- **P8:** Direct prior for 0.25-25% removal doses; a modern behavior/prevalence phase diagram remains open.
- **P9:** Exact group retraining is the reference quantity, but no neural-LM concept-score calibration.
- **P10:** LF coverage is adjacent; class-balance confounding means it cannot be claimed as query coverage or causal support.

## Record E: Instability of LOO and influence metrics (Karthikeyan and Søgaard, arXiv:2111.04683v1)

### Source facts and limits

The paper evaluates MNIST and CoMNIST CNNs across seeds, data orderings, and batch sizes. LOO and influence functions are unstable: under initialization, MNIST LOO Pearson/Spearman are 0.043/0.037 and IF 0.023/0.014; under ordering, LOO is 0.112/0.046 and IF 0.071/0.066 (`Table 1`, p. 4). Expected influence over model states is much more stable and improves poisoning-attack detection, while the authors argue that redundant points can have low individual but high group effects (`§§3-5`, pp. 3-7).

This is not an LM or concept-filtering study, and it has no public artifact URL located in the paper. Its main relevance to Proposal 9 is experimental hygiene: “exact LOO” is not automatically a stable gold standard. A small calibration organism needs multiple seeds/orderings or an expected-effect estimate, and group effects should be measured directly rather than inferred from one run.

### Proposal IDs affected

- **P6:** Supports non-additivity and low-individual/high-group-effect warnings.
- **P9:** Requires seed/order/batch stability checks in any exact counterfactual calibration; otherwise the “gold standard” can itself drift.

## Record F: Datamodels (Ilyas et al., ICML 2022)

### Source facts

- Datamodels fit a surrogate `g_theta` for a fixed target example, learning from many subset masks and retrained model outputs. A linear model predicts the target output for arbitrary subsets, not only the fixed-size subsets used for fitting (`§§1-2`, pp. 1-4).
- On CIFAR-10 and FMoW, in-distribution subset predictions have correlations near 0.99. For arbitrary removed sets, six removal sizes `k=10...1280` (0.02%-2.6%) and five selection heuristics yield Pearson 0.96/0.90 and Spearman 0.98/0.94 between predicted and true counterfactuals (`§4.2`, Fig. 4, pp. 9-11).
- Datamodels estimate target-specific support: in CIFAR, more than 25% of test images can be misclassified after removing fewer than 100 target-specific training images, and about 20% have estimated support below 40 images (`§4.1`, Fig. 3, pp. 8-9).
- The authors explicitly note estimation and misspecification error, dependence on subset-size fraction alpha, and the need to fit several alpha values. The framework is image-only, not a causal-LM or SFT behavior experiment (`§4.2`, p. 11; Appendix E).

### Interpretation, collision, and limits

Datamodels is a strong exact-counterfactual and support-size precedent for Proposal 9 and a conceptual precursor to Proposal 10. It does not test language-model behavior, concept selectors, or a query-family coverage metric. Target-specific support size is not the same as the fraction or entropy of behavior expressions covered by a selected dataset group. The artifact is inspectable but large: the official README lists 4,155 GB total transfer and about USD 53 excluding logits, which makes it a methodological reference rather than a two-week reproduction target.

### Proposal IDs affected

- **P6:** Supports support-size and group-combination analyses, but not semantic cluster-complete removal.
- **P8:** Provides a matched removal-size ladder, but no prevalence manipulation or SFT behavior.
- **P9:** `STRONG COLLISION` for surrogate-vs-true counterfactual calibration; modern causal-LM behavioral calibration is still open.
- **P10:** Target-specific support is adjacent but does not test cross-query coverage.

## Record G: Persona Vectors (Chen et al., arXiv:2507.21509v1)

### Source facts

Persona Vectors extract contrastive activation directions for traits such as evil, sycophancy, and hallucination in Qwen2.5-7B and Llama3.1-8B. Steering changes trait expression, and dataset-level activation shift correlates with post-SFT trait expression (`§§2-4`, pp. 2-9). Their projection-difference score compares supplied training responses with base-model natural responses on the same prompts (`§6.1`, pp. 9-10).

For LMSYS-Chat-1M and other real datasets, they choose 500 high-projection-difference, 500 low-projection-difference, and 500 random examples, then SFT separate models. High-signal examples induce the strongest trait, random is intermediate, and low-signal is weakest; this persists after an LLM judge removes samples with explicit trait expression (`§6.2-6.3`, pp. 10-12; Fig. 10). This is a subset-selection/retraining result, not deletion or matched leave-group-out.

The authors caution that extraction is supervised and prompt-elicited, directions are coarse, GPT-4.1-mini judging is imperfect, only two mid-size chat models and a few traits are covered, and projection difference requires generating base responses for every training prompt (`§8`, pp. 12-13). The official repository is available, but no exact group-effect or deletion calibration is included.

### Interpretation and Proposal IDs affected

Persona Vectors is a direct novelty collision for P1/P2/P3/P7 if their claim is only “contrastive persona/assistantness direction identifies or filters trait-inducing SFT examples.” It supports using base-to-supplied-response projection difference as a selector and reinforces the distinction between a behaviorally causal direction and a causal data-removal effect. It does not settle P6, P8, P9, or P10: no redundancy-aware group deletion, prevalence-by-dose ladder, exact attribution calibration, or query coverage predictor is reported.

## Record H: Emergent Misalignment (Betley et al., ICML 2025)

### Source facts and limits

GPT-4o is SFT on 6,000 insecure-code completions. The insecure model writes insecure code over 80% of the time and gives broadly misaligned free-form answers around 27-28%, while secure-code and educational-insecure controls show little or no broad misalignment (`§§2-3`, pp. 1-6). The paper releases code, prompts, preprocessing, and evaluation details, but not proprietary GPT-4o weights.

The dose-adjacent ablation holds total tokens approximately constant and compares one epoch on all 6,000 completions, three epochs on 2,000-example disjoint subsets, and twelve epochs on 500-example subsets, with three or six repeats. Smaller, less diverse subsets show less broad misalignment (`§4.1`, pp. 8-10; Fig. 6). This is a **training-subset-size/diversity** experiment, not a matched removal-dose experiment: the models are trained on different data subsets rather than starting from one model and deleting a fixed fraction.

### Proposal IDs affected

This paper is useful positive-control context for P8's prevalence/dose argument and for P2/P6's diversity/redundancy mechanism, but it cannot establish a TDA selector's causal support or removal threshold. It motivates keeping prevalence, number of unique examples, repeat count, and removal fraction as separate variables.

## Record I: Scalable LLM influence and the faithfulness gap (Grosse et al., arXiv:2308.03296v1)

### Source facts

- EK-FAC scales influence estimates to pretrained LMs from 810M through 52B parameters. The authors define an add-example counterfactual but validate the estimator against a proximal Bregman response function (PBRF), not direct leave-one-out retraining (`§§2.1.1, 5.1`, pp. 9-11, 23-26).
- The paper states that neural influence can fail to match its motivating counterfactual and may miss nonlinear circuit formation or global representation rearrangements. It studies pretrained models, MLP parameters only, and only a fraction of the corpus (`§1`, pp. 5-6). Fine-tuning is explicitly left for future work.
- On 22B models, the top 1% of scanned sequences account for 12-52% of positive influence mass across queries, while values above 0.1 are rare and no sequence exceeds 1 for the eight visualized queries (`§5.2.1`, Fig. 9, pp. 27-28). The authors interpret this as heavy-tailed but collective support rather than a few memorized examples.

### Interpretation and Proposal IDs affected

Grosse et al. support P2/P6's concentration-versus-distributed-support framing and P10's need to distinguish mass concentration from behavioral coverage. The 12-52% number is **influence mass over sequences for one query**, not a query-family coverage metric. The PBRF validation and explicit fine-tuning limitation leave P9's modern SFT behavioral calibration open, but they make the faithfulness caveat non-negotiable: EK-FAC agreement with PBRF is not proof that removal/retraining outcomes are predicted.

## Cross-source synthesis by causal question

### Faithfulness: what is actually calibrated?

There are four distinct targets that should not be conflated:

1. **Exact retraining effect:** Koh group effects and Datamodels directly define or estimate retraining after removal. STRIDE uses subset-retrained responses as LDS ground truth. Simfluence predicts held-out loss trajectories from actual training runs.
2. **Local/proximal approximation:** Influence Functions and Grosse's EK-FAC are local approximations; Grosse validates PBRF, not LOO. Concept Influence's Hessian score and Vector Filter are activation-alignment approximations, and their filtering results do not show that their score equals a group-removal effect.
3. **Representation controllability:** Persona Vectors and Concept Influence show that activation directions/features can steer behavior. The papers do not establish that the examples associated with a steerable feature are uniquely removable.
4. **Behavior-level intervention:** The scratchpad asks for held-out behavioral changes after deletion/retraining. None of the reviewed records compares MAGIC, TrackStar, EKFAC, dense probes, and concept-conditioned scores against exact group effects in one modern causal-LM organism.

### Group influence, redundancy, and support

Simfluence gives the clearest language-model argument that scalar additive attribution fails under redundancy and order. Koh shows that groups can have large effects despite low individual effects and that first-order influence can understate or even sign-mismatch a true group effect. STRIDE's SFT support diagnostics show broad active sets in some datasets, especially SafeRLHF (median 80,672; top-10% mass coverage 0.1796), while pretraining is much sparser. Grosse's influence mass is heavy-tailed but still collective. These sources support P6's mechanism hypothesis, but no one implements a semantic cluster-complete removal with equal document budget and held-out behavioral outcomes.

### Coverage: three different quantities

- **Dataset coverage:** Concept Influence's top 20% concepts cover 70% of training examples; this is not evaluation-query coverage.
- **Influence-mass coverage:** STRIDE's top-10% coverage and Grosse's top-1% mass are concentration diagnostics for one query or dataset, not behavior-family coverage.
- **Target-specific support:** Datamodels estimates the smallest training subset that flips one target image, not the fraction or entropy of behavior expressions covered by a selected set.

Koh's CDR LF coverage is the closest semantic word match, but the authors explicitly note class-balance confounding and do not make a causal claim. No reviewed source tests Proposal 10's fraction/entropy of held-out behavior queries covered by a selected document group, then predicts an intervention effect.

### Removal dose and prevalence

Dose is not new by itself. Koh varies exact group removal from 0.25% to 25%; Datamodels tests 0.02%-2.6% removed; Concept Influence uses 1%, 10%, 20%, ... group filtering plus a 5% OASST1 selection point; Persona Vectors trains 500-example high/low/random subsets; and Emergent Misalignment varies 500/2,000/6,000 training-subset sizes at fixed token budget. None creates the Proposal 8 design: two behavior prevalences, matched removal budgets, a known-source oracle, and separate reporting of source classification, causal support removed, and held-out behavior change. The novelty claim must therefore be “dose-matched modern broad-SFT causal phase diagram,” not “first study of dose.”

## Explicit decisions for P4, P6, P9, and P10

| Proposal | Novelty decision | Method support | Required narrowing or control |
|---|---|---|---|
| **P4: representation causality versus data removability** | **Novel enough as a framework; no direct collision found.** Persona Vectors and Concept Influence show controllable semantic directions, but none tests whether their associated examples are removable. | **Strong conceptual support, incomplete causal precedent.** Persona steering, Concept Influence filtering, and STRIDE/Koh counterfactuals establish the two intervention families separately. | Use a bidirectional steering check and a separate matched leave-group-out retraining test. Do not infer removability from steering, ranking, or projection difference. |
| **P6: cluster-complete filtering for redundant concept support** | **Novel at the proposed endpoint; adjacent group-influence work is substantial.** Simfluence and Koh already establish non-additive/group effects, and Concept Influence already filters SAE groups. | **Strong mechanism support, no direct LLM cluster-complete result found.** Simfluence motivates redundancy; Koh supplies exact group effects; STRIDE shows broad SFT support. | Hold removed count, composition, and dose fixed. Define deterministic overlapping-feature assignment and compare top-k, cluster-complete, and random selectors on held-out behavior. |
| **P9: attribution against exact counterfactual retraining** | **General claim collides strongly; narrow modern behavior claim remains open.** Datamodels, Simfluence, STRIDE, and Koh all calibrate surrogates/attributions against retraining or held-out runs. | **Strongest support in the landscape.** STRIDE has LLM SFT LDS ground truth; Simfluence has held-out LLM runs; Datamodels has exact arbitrary-subset counterfactuals; Koh has exact group retraining. | Reframe as concept-conditioned **behavioral** calibration of MAGIC/TrackStar/EKFAC/probe scores in one small modern causal LM, with repeated seeds/orderings and tens of groups. |
| **P10: query coverage as causal-usefulness predictor** | **Cleanest novelty gap.** No reviewed source tests fraction/entropy of held-out behavior-query families covered by a selected set as a predictor of intervention effect. | **Adjacent diagnostics only.** STRIDE/Grosse measure influence-mass coverage, Concept Influence measures dataset coverage, and Datamodels measures target-specific support; none is query-family coverage. | Predefine behavior families, lock discovery/validation splits, match selector size/score/topic distribution, and label the NGDiff/unlearning endpoint as intervention predictability rather than direct SFT filter-and-retrain. |

**Strongest causal gold-standard evidence, in order.** For a literal data-removal effect, Koh et al. is the clearest exact group-retraining reference, but its convex logistic models are not language models. Datamodels is the strongest arbitrary-subset counterfactual demonstration, with true retraining and high held-out correlations, but it is image-only and costly. For modern LMs, STRIDE is the strongest directly relevant evidence because its LDS targets come from explicit subset retraining and it includes Qwen SFT; Simfluence is the strongest held-out-run calibration evidence because it predicts loss trajectories from actual LLM fine-tuning runs. Grosse's EK-FAC/PBRF result is useful scalability evidence but **not** a gold standard for removal faithfulness, and Concept Influence/Persona Vectors are representation-control or filtering evidence rather than exact group-effect evidence.

## Decision guidance for the scratchpad

- **P8 should survive only as a narrow dose-and-prevalence diagnosis.** Keep the original 10%/25% versus 50% motivation, but add matched oracle groups and report all three quantities separately. Treat the EM 500/2k/6k result as a diversity control, not prior evidence for a removal threshold.
- **P9 should be reframed as concept-conditioned behavioral calibration, not exact counterfactual calibration in general.** Simfluence, Datamodels, STRIDE, and Koh already establish the general method. A small modern causal-LM organism, repeated seeds/orderings, one broad behavior, tens of groups, and the specified baseline ladder would still fill a real gap.
- **P10 is the cleanest novelty gap among these three.** Define query families before ranking, lock discovery/validation families, match selector size and score distribution, and distinguish coverage from influence mass. The existing NGDiff/unlearning bank means the causal endpoint is intervention predictability rather than direct SFT filter-and-retrain; that caveat belongs in the claim.

DONE_WITH_CONCERNS
