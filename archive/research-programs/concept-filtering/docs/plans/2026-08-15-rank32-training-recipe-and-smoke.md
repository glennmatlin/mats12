# Plan: named rank-32 recipe and recovery smoke

## Purpose

This checkpoint turns the public rank-32 observations into one runnable named
recipe before any training outcome is observed. It does not claim to recover
the unavailable source recipe. The full-data, dense-probe-removal, matched-
random-removal, and ordinary behavior-query-EKFAC-removal conditions must later
cite the same recipe-core identity.

## Evidence and reconstruction choices

The public adapter fixes LoRA rank 32, alpha 64, dropout 0.1, all seven attention
and MLP projection targets, one epoch, a 2e-4 peak learning rate, a raw trainer
batch field of 2, 123 optimizer steps, and 63,696,407 logged tokens. The source
post fixes the 8,192-token length boundary. These are observations, not a
complete executable recipe.

Two independent public trainer summaries strongly constrain batching. The
reported 5.059 train samples per second over 1,545.2445 seconds implies about
7,817.39 packed sequences; divided by 123 steps, that is 63.56 sequences per
step. A 64-sequence effective batch has capacity for 64,487,424 tokens over 123
steps at length 8,192, and the observed token count fills 98.77% of that
capacity. The named recipe therefore selects stable packing and effective batch
64 as evidence-informed reconstruction choices. The arithmetic constrains those
choices but does not recover the source configuration.

The raw trainer batch field is retained as micro-batch 2. A single-worker
executor with 32 accumulation steps realizes the inferred effective batch of
64. AdamW, linear decay with five warmup steps, seed 42, bfloat16, stable
hash-key ordering, assistant-content-plus-terminal-EOS loss, overlength
dropping, and exact checkpoint state are predeclared reconstruction choices
because the public artifacts do not recover them.

## Frozen recipe

The machine-readable source of truth is
`configs/reconstruction/training/public-rank32-named-recipe-v1.json`. It binds:

- the immutable candidate base, tokenizer, chat template, and train sample;
- every LoRA, optimizer, scheduler, batch, precision, sequence, order, loss,
  checkpoint, and resume field named by the replication-backbone spec;
- the roles, internal body hashes, and relevant scientific contents of the
  public load-smoke, recipe, sample reconstruction, and trace-derivation
  receipts;
- retained primary `adapter_config.json` content, normalized only for its
  repository newline, exact `trainer_state.json` bytes, and the legacy adapter
  artifact receipt, so receipt rebinding within the frozen trust roots cannot
  replace rank, trace, path, or weight identity;
- a provenance entry for every executable leaf field;
- the one recipe-core identity required by all four fresh conditions;
- a separately hashed smoke authority envelope with scientific, provider,
  scheduler, full-condition, feature-selection, and concept-scoring authority
  false; and
- a bounded, non-scientific two-step recovery-smoke contract.

The loader establishes internal and upstream-evidence consistency, then emits a
recipe-core SHA-256 over the reusable recipe, artifacts, evidence, runtime, and
four-condition binding. Smoke-only fixture, scale, and authority settings are
excluded from that core. The complete config still has its own hash, and the
seven false authority flags have a separate canonical hash. The checked-in Git
checkpoint authenticates all three identities and their trust-root evidence.

## Recovery-smoke execution semantics

The bounded smoke reads only the checked-in four-row synthetic fixture, whose
schema and source identity are bound to the retained 25,007-row sample receipt.
Full-table loading is deferred to the shared-input materialization ticket. The
stable order defines each fixture example ID as the canonical SHA-256 of
`{"source_revision": <revision>, "orig_idx": <index>}`, sorts by the canonical
SHA-256 of `{"algorithm": "sha256_seeded_sample_id_v1", "seed": 42,
"sample_id": <example-id>}`, and uses the example ID as a tie-breaker.

Each fixture conversation is rendered with the pinned chat template, without an
added generation prompt or tokenizer-level special tokens. Rows beyond the
128-token smoke boundary are dropped and their identities are receipted. The
two selected conversations are concatenated in stable order and split into two
length-128 batches; prompts remain visible in the attention mask but have label
`-100`, while every assistant content span and each terminal conversation EOS
contribute to loss. The pinned-tokenizer preflight yields four input rows, three
eligible rows, one dropped row, two selected rows, 180 rendered tokens, 51
assistant/EOS labels in three supervised runs, two terminal EOS labels, and two
distinct packed batches. The smoke's linear
scheduler uses its bounded two-step horizon and the recipe's five warmup steps.
Those counts, the exact selected and dropped IDs, token-stream hash, and ordered
batch hashes are frozen in the smoke contract and checked after the backend
returns, so internally consistent preprocessing drift still fails closed. A
later full-condition scheduler horizon must be derived from the frozen full
packed data, epoch count, and effective batch rather than copied from this smoke
override.

The step-1 checkpoint contains exactly the adapter tensors, adapter recipe
identity, optimizer state, scheduler state, CPU RNG state, trainer state, and
next-batch cursor. The cursor binds the fixture, token stream, packed batches,
and next-batch hash. A fresh base-plus-LoRA model must load that checkpoint and
consume batch 1 after the initial batch 0. The result is compared against a
separately initialized uninterrupted two-step path over every adapter tensor.

## Implementation sequence

1. Validate the recipe schema, exact fields, arithmetic invariants, receipt
   roles and contents, file manifests, provenance bases, and non-scientific
   boundary.
2. Commit and push the frozen recipe core, fixture, authority envelope, and
   executor before executing a training smoke.
3. Implement deterministic conversation rendering, tokenization, stable order,
   packing, labels, LoRA application, optimizer/scheduler state, and checkpoint
   resume behind one receipt-producing command.
4. Run an uninterrupted two-step path and a one-step-checkpoint-resume path.
   Compare every adapter tensor with the predeclared tolerance and retain the
   terminal receipt.
5. Run the full software gates plus independent Standards, Spec, and clean
   verification before closing issue 4.

## Superseded 7B evidence

The pinned 7B run at executor revision
`b594b488ac54251a82304374750c43130c5869d6` completed both paths and wrote
`reports/reconstruction/smokes/2026-08-16-public-rank32-training-recovery-v1.json`.
Its receipt SHA-256 is
`f3ec1f9c9db080f3883eaefbeb9719a09a286b6261397c07de5f4159fa1afdc9`.
The two loss sequences are exactly `[2.7388014793395996,
3.973231077194214]`; the uninterrupted and resumed adapter hashes both equal
`62dd1a73cdc59b2e8ec6b055e6a27718feed2df86a6e3c67cb229417a4c0547f`,
and the maximum absolute tensor difference is zero. Independent replay of the
receipt body, checkpoint manifest and file hashes, cursor, trainer state,
selected IDs, four-condition core binding, and authority values passes.
Real-data inspection then found 583 multi-turn rows, while this executor
supervised only the final assistant turn and its fixture did not exercise that
shape. This receipt therefore proves the recorded checkpoint mechanics but not
Gate B2.

## Current repair

The repaired smoke keeps recipe core
`d0f8a5639ed7e347e58fab40f20a6be53efc4667c690d1d1fc9857b3d13c1567`
unchanged. Its fixture now exercises a normal exchange, a multi-turn exchange,
an empty user message, and an overlength drop. The real pinned tokenizer
produces 180 rendered tokens, three separately supervised assistant runs, two
terminal EOS labels, and two frozen packed batches. Gate B2 remains open until
a clean executor runs the pinned 7B recovery path and retains a terminal receipt
bound to this exact smoke contract.

## Boundary

The smoke loaded the model and performed bounded offline optimizer steps. It did
not submit Slurm work, call a provider, train a full condition, select SAE
features, intervene on activations, score concepts, or create a concept-selected
manifest. This superseded smoke proves its recorded mechanics and recovery
only; it is not the terminal B2 evidence.
