# Reference verification audit

**Input:** `refs.bib`
**References checked:** 16
**Elapsed:** 60.2s

## Summary

| Verdict | Count |
|---|---|
| verified | 14 |
| source_verified | 2 |
| likely_real | 0 |
| needs_web_verification | 0 |
| suspicious | 0 |
| likely_hallucinated | 0 |
| unverifiable | 0 |

## Thresholds

- `title_similarity` = `0.92`
- `author_coverage` = `0.8`
- `extra_authors_max` = `1`
- `year_tolerance` = `1`

## Per-reference

### Ref 0: `verified`

> Andrew Draganov and Tolga H. Dur and Anandmayi Bhongade and Mary Phuong. (2026). {Phantom Transfer: Data Poisoning can Survive Data-Level Defences}. arXiv:2602.04899. https://arxiv.org/abs/2602.04899

**Matched against:**
- Title: 'Phantom Transfer: Data Poisoning can Survive Data-Level Defences'
- Authors: Andrew Draganov, Tolga H. Dur, Anandmayi Bhongade, Mary Phuong
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2602.04899v2`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 1: `source_verified`

> Dohun Lee and J Rosser and Josh Engels and Neel Nanda. (2026). {Data filtering works a lot worse than you would expect}. https://www.alignmentforum.org/posts/aTybJ6CPQrxEY8rE2/data-filtering-works-...

**Matched against:**
- Title: '{Data filtering works a lot worse than you would expect}'
- Authors: 
- Year: None
- Venue: 'www.alignmentforum.org'
- Source: `url:web`

**URL evidence:**
- Original: https://www.alignmentforum.org/posts/aTybJ6CPQrxEY8rE2/data-filtering-works-a-lot-worse-than-you-would-expect
- Status: 200
- Type: `web` / `text/html`
- Page title: 'Data filtering works a lot worse than you would expect — AI Alignment Forum'

**Rules fired:**
- rule:title-match (score=1.000)
- rule:year-ok
- verdict:url-source-resolved-and-title-corroborated

### Ref 2: `verified`

> Isaac Song and Mohammed Rehan Parwani and Glenn Matlin and Emile Anand and Akhil Theerthala and Arjun Chatterjee and Anthony Wen-Ming Zang and Maria Kostylew and Yonadav G. Shavit and Sebastien Kri...

**Matched against:**
- Title: 'Role Steering of Language Models for Social Simulations'
- Authors: Isaac Song, Mohammed Rehan Parwani, Glenn Matlin, Emile Anand, Akhil Theerthala, Arjun Chatterjee, Anthony Wen-Ming Zang, Maria Kostylew, Yonadav G. Shavit, Sebastien Krier, Mark Riedl
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2608.00023v2`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 3: `verified`

> Lyndon Drake and Zandi Eberstadt. (2026). {Transplanting, inverting, and preventing a misalignment persona: method-conditional emergent misalignment in Qwen2.5}. doi:10.48550/arXiv.2607.04510. arXi...

**Matched against:**
- Title: 'Transplanting, inverting, and preventing a misalignment persona: method-conditional emergent misalignment in Qwen2.5'
- Authors: Lyndon Drake, Zandi Eberstadt
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2607.04510v2`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 4: `verified`

> Viktor Moskvoretskii and Dominik Glandorf and Jorge Medina Moreira and Tanja Käser and Robert West. (2026). {Tracing Persona Vectors Through LLM Pretraining}. doi:10.48550/arXiv.2605.13329. arXiv:2...

**Matched against:**
- Title: 'Tracing Persona Vectors Through LLM Pretraining'
- Authors: Viktor Moskvoretskii, Dominik Glandorf, Jorge Medina Moreira, Tanja Käser, Robert West
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2605.13329v1`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 5: `verified`

> Todd Nief and Harvey Yiyun Fu and Mark Muchane and Ari Holtzman. (2026). {Subliminal Learning is a LoRA Artifact}. doi:10.48550/arXiv.2606.00831. arXiv:2606.00831. https://arxiv.org/abs/2606.00831

**Matched against:**
- Title: 'Subliminal Learning is a LoRA Artifact'
- Authors: Todd Nief, Harvey Yiyun Fu, Mark Muchane, Ari Holtzman
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2606.00831v1`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 6: `source_verified`

> Frank Xiao and Santiago Aranguri. (2026). {Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training}. doi:10.48550/arXiv.2602.11079. arXiv:2602.11079. htt...

**Matched against:**
- Title: '{Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training}'
- Authors: 
- Year: None
- Venue: 'arxiv.org'
- Source: `url:arxiv`

**URL evidence:**
- Original: https://arxiv.org/abs/2602.11079
- Status: 200
- Type: `arxiv` / `text/html`
- Page title: '[2602.11079] Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training'

**Rules fired:**
- rule:title-match (score=1.000)
- rule:year-ok
- verdict:url-source-resolved-and-title-corroborated

### Ref 7: `verified`

> Clemens Vetter and David Kaczér and Lucie Flek and Florian Mai. (2026). {Data Attribution of Emergent Misalignment with Persona Features}. doi:10.48550/arXiv.2608.11025. arXiv:2608.11025. https://a...

**Matched against:**
- Title: 'Data Attribution of Emergent Misalignment with Persona Features'
- Authors: Clemens Vetter, David Kaczér, Lucie Flek, Florian Mai
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2608.11025v1`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 8: `verified`

> Matthew Kowal and Goncalo Paulo and Louis Jaburi and Tom Tseng and Lev E McKinney and Stefan Heimersheim and Aaron David Tucker and Adam Gleave and Kellin Pelrine. (2026). {Concept Influence: Lever...

**Matched against:**
- Title: 'Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution'
- Authors: Matthew Kowal, Goncalo Paulo, Louis Jaburi, Tom Tseng, Lev E McKinney, Stefan Heimersheim, Aaron David Tucker, Adam Gleave, Kellin Pelrine
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2602.14869v1`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 9: `verified`

> Glenn Matlin and Chandreyi Chakraborty and Saehee Eom and Mika Okamoto and Rayan Castilla and Louis Jaburi and Alvin Deng and Taywon Min and Lucia Quirke and Stella Biderman and Mark Riedl. (2026)....

**Matched against:**
- Title: 'Capability Provenance in Language Models: A Case Study in Social Reasoning'
- Authors: Glenn Matlin, Chandreyi Chakraborty, Saehee Eom, Mika Okamoto, Rayan Castilla, Louis Jaburi, Alvin Deng, Taywon Min, Lucia Quirke, Stella Biderman, Mark Riedl
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2606.19625v3`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 10: `verified`

> Christina Lu and Jack Gallagher and Jonathan Michala and Kyle Fish and Jack Lindsey. (2026). {The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models}. doi:10.48550/arX...

**Matched against:**
- Title: 'The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models'
- Authors: Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2601.10387v1`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 11: `verified`

> Davi Bastos Costa and Renato Vicente. (2026). {Persona-Model Collapse in Emergent Misalignment}. doi:10.48550/arXiv.2605.12850. arXiv:2605.12850. https://arxiv.org/abs/2605.12850

**Matched against:**
- Title: 'Persona-Model Collapse in Emergent Misalignment'
- Authors: Davi Bastos Costa, Renato Vicente
- Year: 2026
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2605.12850v2`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 12: `verified`

> Edward Turner and Anna Soligo and Mia Taylor and Senthooran Rajamanoharan and Neel Nanda. (2025). {Model Organisms for Emergent Misalignment}. doi:10.48550/arXiv.2506.11613. arXiv:2506.11613. https...

**Matched against:**
- Title: 'Model Organisms for Emergent Misalignment'
- Authors: Edward Turner, Anna Soligo, Mia Taylor, Senthooran Rajamanoharan, Neel Nanda
- Year: 2025
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2506.11613v1`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 13: `verified`

> Bruce W. Lee and Addie Foote and Alex Infanger and Leni Shor and Harish Kamath and Jacob Goldman-Wetzler and Bryce Woodworth and Alex Cloud and Alexander Matt Turner. (2025). {Distillation Robustif...

**Matched against:**
- Title: 'Distillation Robustifies Unlearning'
- Authors: Bruce W. Lee, Addie Foote, Alex Infanger, Leni Shor, Harish Kamath, Jacob Goldman-Wetzler, Bryce Woodworth, Alex Cloud, Alexander Matt Turner
- Year: 2025
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2506.06278v3`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 14: `verified`

> Alex Cloud and Minh Le and James Chua and Jan Betley and Anna Sztyber-Betley and Jacob Hilton and Samuel Marks and Owain Evans. (2025). {Subliminal Learning: Language models transmit behavioral tra...

**Matched against:**
- Title: 'Subliminal Learning: Language models transmit behavioral traits via hidden signals in data'
- Authors: Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans
- Year: 2025
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2507.14805v1`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass

### Ref 15: `verified`

> Jin Hwa Lee and Matthew Smith and Maxwell Adam and Jesse Hoogland. (2025). {Influence Dynamics and Stagewise Data Attribution}. doi:10.48550/arXiv.2510.12071. arXiv:2510.12071. https://arxiv.org/ab...

**Matched against:**
- Title: 'Influence Dynamics and Stagewise Data Attribution'
- Authors: Jin Hwa Lee, Matthew Smith, Maxwell Adam, Jesse Hoogland
- Year: 2025
- Venue: 'arXiv'
- Source: `arxiv`
- arXiv: `2510.12071v2`

**Rules fired:**
- rule:title-match (score=1.000)
- rule:author-coverage-ok (100%)
- rule:year-ok
- verdict:identifier-resolved-and-all-checks-pass
