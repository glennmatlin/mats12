# Compute ledger

This file is append-only. Entries distinguish receipt-bound facts from local
process observations and record bounded reconstruction compute before the
scientific experiment begins.

## 2026-08-16: superseded pinned-7B recovery attempt

- Purpose: test whether the frozen public rank-32 recipe can train, checkpoint, load
  into a fresh model, resume from the persisted cursor, and reproduce the
  uninterrupted adapter exactly.
- Executor: local Git revision
  `b594b488ac54251a82304374750c43130c5869d6` on CPU with bfloat16 weights; no
  GPU, Slurm allocation, provider call, or full-condition training was used.
- Work: two uninterrupted optimizer steps, followed by one checkpointed step
  and one fresh-model resumed step over the two frozen length-128 batches.
- Local observations: launch-to-receipt wall time was about 7 hours 34 minutes;
  the largest sampled resident set was at least 15.5 GB. These process samples
  are operational observations, not receipt-bound metrics.
- Result: terminal receipt
  `f3ec1f9c9db080f3883eaefbeb9719a09a286b6261397c07de5f4159fa1afdc9`
  records identical loss sequences and terminal adapter hashes with zero
  maximum absolute tensor difference. The receipt file SHA-256 is
  `2f25bb40b482a08ecfa9eba454a2e4634a3464ca3ac41625e19c71993f01b5dd`.
  Subsequent real-data inspection found 583 multi-turn rows that this executor
  did not model correctly, so the run is retained as superseded mechanics
  evidence and not counted as B2 completion.

## 2026-08-16: shared training-input materialization

- Purpose: deterministically materialize and replay the content-free shared
  training-input package before base-response generation or activation work.
- Executor: local Git revision
  `20ef48b9a2a09e76b3376255c4973194e136c00f` on CPU; no GPU, Slurm allocation,
  provider call, model training, or scientific evaluation was used.
- Work: render and hash all 25,007 source rows twice, once for publication and
  once for source-derived verification. The package contains 63,639,095 input
  tokens and 58,414,917 supervised tokens.
- Result: both passes produced receipt
  `f18681250c206ba035fa4b75f8dfbd777b1aa1f6744591f23a1afa3783729036`
  and package SHA-256
  `1f0a8d17cce76ecbab78e3a6aaf31dd9fede270ac930e4c2e720563409bc3b28`.
  This closes only the content-free package slice of B4; base responses and the
  activation-contract smoke remain open.
