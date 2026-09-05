# Literature landscape protocol

## Identity

- Review title: Concept-level data filtering and filterability mechanisms for a short research application
- Mode: landscape
- Owner: project research team
- Created: 2026-08-13
- Last amended: 2026-08-13

## Question and purpose

- Review question: Which candidate direction in the accompanying project-idea
  scratchpad has a defensible novelty gap, a causally interpretable methodology,
  and an experiment that can produce useful evidence within 20 active hours and
  one to two calendar weeks?
- Intended decision: Select two or three approaches for final methodological
  design and adversarial grilling, without assuming that concept filtering is
  the winner.
- Accompanying proposals:
  `docs/specs/2026-08-13-mats-stream-project-idea-scratchpad.md`.
- Known seed works:
  - *Data Filtering Works a Lot Worse Than You Would Expect*.
  - *Concept Influence: Leveraging Interpretability to Improve Performance and
    Efficiency in Training Data Attribution* (arXiv:2602.14869).
  - *Why Do Naive SFT Filters for Safety Properties Fail?*
  - The public OLMo-3 SAE release and its training configuration.

## Scope

- Concepts: training-data attribution, concept-conditioned attribution, sparse
  autoencoders, causal feature intervention, model diffing, data filtering and
  replacement, SFT behavior inheritance, persona representations, attribution
  faithfulness, support concentration, redundancy, and removal dose.
- Population/context: causal language models and post-training or fine-tuning
  data. Small language-model studies are eligible when they directly validate an
  attribution or filtering mechanism intended to transfer to modern LLMs.
- Date range: 2017-01-01 through 2026-08-13.
- Languages: English.
- Eligible source types: peer-reviewed papers, preprints, primary technical
  reports, primary research forum posts, official model or dataset cards, and
  official code repositories needed to assess artifact availability.
- Excluded source types: secondary summaries, news, unsourced commentary,
  papers about inference-time feature attribution without a training-data or
  behavior-control connection, and non-language studies without a structurally
  transferable mechanism.

## Eligibility

- Include a record when it informs at least one proposal's novelty, mechanism,
  experimental design, baseline choice, causal interpretation, or feasibility.
- Include negative results and methodological critiques even when they argue
  against the proposed approach.
- Exclude records that use "concept influence" only in an unrelated sense or
  discuss dataset filtering solely for quality, deduplication, or toxicity
  classification without behavior-level retraining evidence.
- Screening stages: title/abstract, then targeted full text for included records.
- Full-text policy: retain an abstract-level record only for discovery; do not
  use it for detailed methodological or quantitative synthesis. Mark inaccessible
  full text separately.

## Search plan

- Sources: arXiv, OpenAlex or Semantic Scholar, OpenReview and venue proceedings,
  Alignment Forum and LessWrong primary research posts, and official artifact
  repositories.
- Exploratory query families:
  1. concept influence, concept-conditioned influence, and concept-level TDA;
  2. sparse autoencoders combined with training-data attribution or filtering;
  3. causal SAE feature steering, ablation, and mediation;
  4. fine-tuning model diffing, crosscoders, and inherited versus learned traits;
  5. persona vectors, assistantness directions, and post-training behavior;
  6. data filtering, removal, replacement, and retraining for model behavior;
  7. attribution faithfulness checked by counterfactual retraining;
  8. group influence, redundancy, coverage, effective support, and removal dose;
  9. emergent misalignment filtering and source-recovery controls;
  10. prompt or teacher-rollout swapping in SFT.
- Main queries: refine once using terminology and citations found during the
  exploratory stage. Preserve each exact source-specific query in
  `search_log.jsonl`.
- Citation chaining: backward-chain all core seed papers and forward-chain them
  through one available citation index, capped at 20 records per seed and
  direction.
- Planned last-search date: 2026-08-14.
- Result caps: 15 records per exploratory source-query and 20 records per main
  source-query. Search-engine results are discovery only; included claims must
  resolve to a primary source.
- Stop rule: all ten query families have terminal receipts in arXiv and one
  citation index; the three research-post seeds and four academic seed families
  have completed bounded citation chains; one refinement round has run; and
  every discovered record has a screening status. The review remains an
  indicative landscape and makes no completeness claim.

## Extraction and appraisal

- Data fields: candidate proposal IDs; research question; model and training
  stage; data and behavior type; representation or concept definition; selector;
  intervention and removal dose; comparators; whether retraining is performed;
  causal feature validation; held-out evaluation; main quantitative result;
  negative or mixed findings; author and reviewer limitations; code, data, and
  checkpoint availability; novelty collision; and estimated fit to the 20-hour
  constraint.
- Evidence tiers: metadata only, abstract read, full text read, and primary
  result checked.
- Quality method: structured methodological appraisal rather than a formal
  risk-of-bias score. Give most weight to studies with matched baselines,
  explicit retraining or data interventions, causal representation tests,
  held-out evaluation, uncertainty across seeds, and inspectable artifacts.
- Synthesis method: compare evidence by causal question, then produce a
  proposal-by-evidence matrix covering novelty, methodological support,
  contradictions, artifact readiness, and two-week risk.
- Human reviewers: one project owner will make the final research decision;
  human screening has not yet been performed.
- Automated reviewers/tools: one coordinating reviewer plus two independent
  discovery passes. The coordinator verifies citations, screens records, and
  owns the final synthesis.

## Amendments

### 2026-08-13: bounded citation-chain substitution and closure

- Reason: the citation index returned complete metadata and citation edges for
  three academic seed families, but rate-limited the new persona-attribution
  seed with HTTP 429 after bounded retries. Forum posts do not expose a
  structured forward/backward citation graph.
- Change: cap and screen the first 20 backward and forward records for each of
  the three accessible academic seeds; preserve the blocked fourth seed as a
  terminal limitation. For forum sources, use exact-title retrieval plus
  linked-source chasing rather than inventing citation edges.
- Affected stages: main search, citation chaining, stop rule, and coverage
  claims. The review may close as a landscape with named limitations, but it
  cannot claim exhaustive citation-chain completion.
- Last search date: 2026-08-13.
- Review labor: one coordinating automated reviewer, two independent landscape
  passes, four targeted full-text extraction passes, and one structured
  title/abstract screen. No human screening or adjudication was performed.
