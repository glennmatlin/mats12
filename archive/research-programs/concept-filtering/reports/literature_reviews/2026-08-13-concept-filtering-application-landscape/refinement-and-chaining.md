# Refinement and citation-chain report

## Status

The planned refinement round completed. Three of four academic seed families
completed bounded backward and forward chaining; the newest persona-attribution
seed was blocked by citation-index rate limiting. This is a terminal landscape
receipt with a coverage limitation, not an exhaustive search claim.

## Named-method refinement

The second search pass replaced broad terms with methods, organisms, and
interventions found during exploration. Exact queries and result caps are also
preserved in `raw_results/worker-b-adversarial.md`.

| Query ID | Exact source query | Returned/screened | Main use |
|---|---|---:|---|
| main-001 | `all:"concept influence" AND (all:"training data attribution" OR all:"sparse autoencoder")` | 4 | Direct concept-filtering collision |
| main-002 | `all:"sparse autoencoder" AND (all:"training data attribution" OR all:"data filtering" OR all:influence) AND (cat:cs.LG OR cat:cs.AI OR cat:cs.CL)` | 15 of 41 | SAE data engineering and causal validity |
| main-003 | `ti:"persona vectors" OR all:"persona vector" OR all:"assistantness" OR all:"persona direction"` | 15 capped | Assistantness and persona mechanisms |
| main-004 | `all:crosscoder AND (all:"model diffing" OR all:fine-tuning OR all:SAE)` | 11 | Inherited-versus-taught collision |
| main-005 | `(all:"data filtering" OR all:"filter training data") AND (all:SFT OR all:"supervised fine-tuning" OR all:refusal OR all:sycophancy) AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI)` | 15 of 27 | Broad SFT filtering |
| main-006 | `(all:"influence function" OR all:"training data attribution") AND (all:retrain OR all:counterfactual OR all:faithfulness OR all:leave-one-out) AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI)` | 15 of 75 | Counterfactual calibration |
| main-007 | `(all:"group influence" OR all:"data redundancy" OR all:"removal dose" OR all:"effective support" OR all:"query coverage") AND (all:attribution OR all:influence) AND (cat:cs.LG OR cat:cs.AI OR cat:cs.CL)` | 15 of 26 | Redundancy and coverage |
| main-008 | `all:"emergent misalignment" AND (all:filter OR all:attribution OR all:persona OR all:SAE)` | 15 of 26 | Persona-feature attribution and filtering |
| main-009 | `(all:"teacher rollout" OR all:"response replacement" OR all:"swap completion" OR all:"necessary and sufficient") AND (all:SFT OR all:"supervised fine-tuning" OR all:distillation)` | 15 of 66 | Swap-versus-delete precedent |
| main-010 | `ti:"datamodels" OR ti:TRAK OR ti:MAGIC OR ti:TrackStar OR ti:Bergson` | 12 capped | Attribution faithfulness methods |

Exact-title web retrieval resolved the two filtering posts, the hereditary-trait
replication, the official OLMo-3 SAE card, the base-model card, and the public
SFT dataset card. Linked-source chasing recovered the strongest response-swap,
persona-vector, feature-absorption, and exact-counterfactual precedents.

## Academic seed citation chains

The raw citation-index responses are retained in
`raw_results/semantic-scholar-seed-metadata.json` and
`raw_results/semantic-scholar-citation-chains.json`. A chain with more than 20
records was screened only through the declared cap.

| Seed | Index identity | Backward chain | Forward chain | Terminal status |
|---|---|---:|---:|---|
| Concept Influence, arXiv:2602.14869 | `27d3eb40ef4377577135909549b532641bd0b014` | 20 of 46 | 2 of 2 | Complete within cap |
| SAE post-training data engineering, arXiv:2605.27354 | `9190b18d10b6999071d44ff185256ce4337ad69e` | 20 of 52 | 0 of 0 | Complete within cap |
| Persona Features Control Emergent Misalignment, arXiv:2506.19823 | `473ea02396f57d35476a3a3c29e08d77d6ec471e` | 20 of 77 | 20 of 69 | Complete within cap |
| Data Attribution of Emergent Misalignment with Persona Features, arXiv:2608.11025 | Corpus ID `291024291` | 0 | 0 | Blocked: HTTP 429 at seed lookup |

The accessible chains recovered expected anchors rather than a new research
category: persona vectors, emergent-misalignment organisms, refusal directions,
large-scale influence functions, targeted instruction-data selection, and SAE
evaluation papers. Two forward citations to Concept Influence concerned
symbolic behavioral policies and preference-aware data selection. None removed
the gap around broad SFT behavior, matched deletion/replacement, and separate
representation-control versus data-removability outcomes.

## Research-post chain substitution

Forum posts lack reliable structured citation edges. The bounded substitute was
full-text exact-title retrieval, inspection of every linked primary source used
for a methodological claim, and one linked replication/artifact pass. This
found direct response-swap evidence and official code/weights for hereditary
traits, but no released sample manifest, prompt bank, or dedicated code package
for the OLMo speed-run filtering study.

## Coverage consequence

The fourth academic chain and formal forum citation graphs remain incomplete.
Absence claims in `synthesis.md` are therefore phrased as “no direct collision
was found within the declared sources and bounds,” never as proof that no such
paper exists.
