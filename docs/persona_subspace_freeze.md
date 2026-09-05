# Persona subspace — frozen specification

Frozen 2026-08-14 in a grill session with Glenn (map ticket #5). This spec
binds the recipe; the computed artifact (vectors, layer-selection curve,
basis matrix) lands in artifacts/persona_subspace/ and completes the ticket.
Changes after the computation run = dated addenda only.

1. **Basis model**: `allenai/Olmo-3-1025-7B` @ `main` (mid-train base) —
   canonical frozen basis. Pre-registered robustness recompute on the
   speedrun-SFT model (identical recipe): report persona_load stability of
   behavior directions across the two bases.
2. **Recipe**: Assistant-Axis contrast (vendored `assistant-axis` pipeline):
   axis = mean(default-assistant activation vectors) − mean(per-role
   vectors); per-role vectors are contrastive means over role prompts.
   Base-checkpoint prompting follows the EPFL `pretraining_persona`
   machinery (no chat template on the base model).
3. **Role set**: the vendored inventory verbatim — 275 roles from
   `vendor/assistant-axis/data/roles/role_list.json` (sha256:1e27b846235326bf) with
   `extraction_questions.jsonl` (sha256:31650b9a55d6b827). Zero researcher degrees of
   freedom. Glenn's 275-role instrument (arXiv:2608.00023) is deliberately
   NOT part of the frozen basis; reserved as a later robustness/extension
   column so the application basis stays fully third-party.
4. **Layer**: selected from L12–L22 maximizing d′ between role and
   default-assistant activation sets, computed only from the contrast
   activations (no attribution data consulted); the criterion, chosen layer,
   and full d′-by-layer curve are recorded in the artifact.
5. **k = 4**: assistant axis + top-3 PCs of the per-role mean vectors
   (roles centered, PCA at the frozen layer), orthonormalized axis-first
   (QR via `mats12.separability.orthonormal_basis`).
6. **Provenance string**: `aa-verbatim/olmo3-main/Lsel-dprime/k4/seed20260813`
   (layer index substituted at computation time).

Decision log: basis+robustness, layer rule, and k chosen by Glenn from
recommended options; role set chosen by Glenn (third-party verbatim over
own-instrument) for researcher-dof optics — 2026-08-14.

**Note (2026-08-14, post-freeze same-day):** the vendored role inventory is
275 roles — the same count (and evidently the same inventory) Glenn's
arXiv:2608.00023 validated on OLMo3-7B-Instruct. The frozen basis is
therefore simultaneously third-party-verbatim AND pre-validated on this
model family by prior published work; both properties are citable in the
write-up.

**Implementation notes (2026-08-14, recorded before the computation ran;
scripts/03_extract_subspace.py, jobs 5577693/5577694/5577695):**

- Prompt rendering follows the EPFL base branch of `get_text` verbatim:
  instruction and question joined by a single space, no chat template.
- Instruction variants rotate deterministically (question_id % n_variants);
  one sampled continuation per (persona, question) pair — 240 samples per
  persona at the vendored defaults (temperature 0.7, max_new_tokens 512).
- No judge filtering of role rollouts: the spec is silent, and the AA
  pipeline's score-3 filter runs a per-rollout LLM judge we did not freeze;
  all personas get the unfiltered mean — the same treatment the vendored
  4_vectors.py applies to its default role. Disclosed in the write-up.
- The default-assistant side uses the vendored default.json (5 variants
  including the empty instruction), `{model_name}` filled with "Olmo 3".
- Layer indices are hidden_states positions (0 = embeddings), the EPFL
  convention behind "L12–L22".

**Addendum (2026-08-16, dated; prompted by the adversarial code review — no
frozen choice changes):**

- Wording: the decision log's "for researcher-dof optics" meant, and is hereby
  restated as, "to eliminate researcher degrees of freedom".
- Role-inventory identity, upgraded from "evidently" to verified: the vendored
  role_list.json and the role inventory in the arXiv:2608.00023 artifact repo
  are byte-identical, sha256
  1e27b846235326bf1eb9bc2a79fc2312ab1c260996052dafb6ce00acca9e7918 on both
  sides (275/275 roles).
- Sampling divergence, disclosed: the vendored 1_generate.py passes
  top_p=0.9; our driver passes temperature only. Under transformers 5.15.0
  with this model's empty generation_config the sampling resolved to pure
  temperature-0.7 (top_p=None, top_k=None) for the frozen base artifact and
  the SFT recompute alike — internally consistent across both artifacts;
  extraction reports now record the resolved values.
- Basis sign convention: QR was free to negate basis rows, so the realized
  sign of row 0 (the assistant axis) was arbitrary. build_basis now aligns
  row 0 with the contrast axis and each PC row with its SVD right-singular
  vector — spans and all |projection| readouts unchanged. Realized signs of
  the already-committed base artifact are recorded in its extraction report
  addendum rather than regenerated.
- Weights identity: the staged olmo3-base snapshot is
  allenai/Olmo-3-1025-7B @ 15fe2e8b8b0fbf49416d3943ca6cf0174d0f2c07
  (recovered from the hf-download metadata); 00_download_assets.sh pins this
  revision and extraction reports carry it as weights_identity.
