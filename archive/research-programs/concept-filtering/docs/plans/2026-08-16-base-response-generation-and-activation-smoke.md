# Base-response generation and activation-smoke plan

## Outcome

Complete the two remaining Gate B4 slices without crossing into SAE feature
selection or concept scoring. First, generate one receipt-bound base-model
response for every retained training example. Then prove that the same joined
example can flow through the frozen layer-16 activation hook and aggregation
schema in one bounded smoke.

This plan begins from shared-package evidence checkpoint
`53a608715857158ef7588afd200624db7346b96c`. Gate B4 remains open until both
slices have terminal receipts.

## Dependencies and authority

- The base-response executor consumes the checked-in shared-package receipt,
  its ignored canonical JSONL package, the pinned source Parquet, the exact
  six-file tokenizer snapshot, and the named rank-32 recipe.
- Execution also requires the corrected terminal B2 receipt for recipe core
  `d0f8a5639ed7e347e58fab40f20a6be53efc4667c690d1d1fc9857b3d13c1567`.
  A superseded or merely self-consistent recovery receipt fails closed.
- Base-response generation is local or scheduler execution, never provider
  execution. Scheduler submission requires its own later release-bound request;
  the local implementation and fixture tests do not grant it.
- Activation extraction remains false during response generation. Training,
  scientific use, the experiment timer, SAE feature selection, causal
  intervention, concept scoring, and concept-selected training remain false in
  every B4 receipt.

## Named reconstruction choices

The public source does not recover the base-response sampling, stop, batching,
or length settings. Freeze these as named reconstruction choices:

- Use the pinned OLMo-3 7B base model without the public LoRA adapter.
- Render only the prompt turns with the pinned public chat template and
  `add_generation_prompt=true`.
- Decode greedily with `do_sample=false`, batch size one, and explicit tokenizer
  EOS and padding IDs.
- Limit prompt plus completion to 8,192 tokens. For each row, derive
  `max_new_tokens = 8192 - prompt_token_count`; reject a prompt that leaves no
  completion capacity.
- Stop on terminal EOS. Record both EOS termination and cap termination; never
  silently retry or resample a parseable completion.
- Keep generation order out of scientific identity. Each output is keyed by the
  package's stable example ID, so operational shards can run independently and
  aggregate only after exact identity-set coverage is complete.

## Artifact flow

1. **Plan.** Validate the package receipt, B2 receipt, base model, tokenizer,
   generation core, all capability flags, and release-role tuple without loading
   the model.
2. **Run shard.** Load the pinned base once, claim a deterministic subset of
   example IDs, and write each response atomically before moving to the next.
   Existing valid responses replay; contradictory, aliased, non-regular, or
   partially written outputs fail closed.
3. **Status.** Report verified complete, missing, invalid, EOS-stopped, and
   cap-stopped counts without generating anything.
4. **Finalize.** Require exactly the retained package identity set, then write a
   compact aggregate receipt binding the ordered per-example response hashes,
   generation core, model/tokenizer identities, input producers, and executor.
5. **Verify.** Re-read every response and all upstream artifacts from a clean
   checkout. Verification never loads the model or executes generation.
6. **Activation smoke.** Join one retained example's prompt, supplied response,
   and base response by stable ID; execute the frozen layer-16 hook and declared
   aggregation on the base model; retain tensor shape, dtype, finite-value,
   token-alignment, and schema hashes. Do not select or score a feature.

## Command surface

```text
concept-filtering base-responses plan --config <config> ...
concept-filtering base-responses run-shard --config <config> --shard-index N --shard-count K ...
concept-filtering base-responses status --config <config> ...
concept-filtering base-responses finalize --config <config> ...
concept-filtering base-responses verify --config <config> ...
concept-filtering activation-contract-smoke --config <config> ...
```

`plan`, `status`, and `verify` are read-only. `run-shard` is the only command
that may load the base model and generate text. `finalize` may publish only the
aggregate receipt after complete coverage; it cannot fill missing responses.

## Acceptance gates

- Config loading rejects unknown fields, mutable revisions, incomplete model or
  tokenizer manifests, a wrong package/B2 receipt, and any expanded capability.
- Fixture tests prove interruption and resume, disjoint shard union, overlapping
  worker collision, malformed or hash-valid contradictory samples, symlink,
  hardlink, FIFO, non-finite timings, wrong EOS/cap flags, and absent-only
  aggregate publication.
- A tiny local backend test proves prompt rendering, base-only loading, greedy
  generation, terminal token accounting, and identical uninterrupted/resumed
  per-example bytes.
- The production response set has exactly 25,007 stable IDs, zero extras, one
  valid response per ID, and a replayed aggregate receipt.
- The activation smoke consumes the aggregate response receipt and emits one
  schema-compatible, finite layer-16 observation without feature selection or
  concept scoring.

## Explicitly deferred

This plan does not fit a dense probe, train or choose an SAE, extract a full
activation cache, compute any document score, rank a removal set, train a fresh
condition, execute provider evaluation, or start the experiment timer.
