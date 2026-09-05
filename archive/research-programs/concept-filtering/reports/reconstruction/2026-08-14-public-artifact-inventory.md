# Public artifact inventory for the source organism

Date: 2026-08-14

This narrow reconstruction audit found a public artifact bundle that the
completed landscape review missed. The bundle substantially reduces the
reconstruction search space, but it does not identify a clean producing code
checkout or a complete training recipe. The machine-readable source of truth is
[`configs/reconstruction/public-organism-v1.json`](../../configs/reconstruction/public-organism-v1.json).

## Status vocabulary

- `candidate` means the public object is pinned to an immutable repository
  revision, but its exact role in the source organism or its compatibility with
  another object is not fully receipted.
- `verified` would mean both the object identity and its claimed source-organism
  role are supported. No object currently reaches that stronger status.
- `unavailable` means the public lookup method and failure are recorded instead
  of substituting a guessed artifact.

## Inventory

| Role | Immutable public object | Receipt status | What remains unresolved |
|---|---|---:|---|
| Base model | [`allenai/Olmo-3-1025-7B@a81bae4`](https://huggingface.co/allenai/Olmo-3-1025-7B/tree/a81bae42db3975be1671e27b9c9a56da1a9f980f) | candidate | The post and adapter do not pin the producing base revision. |
| Tokenizer | [`tokenizer.json@a81bae4`](https://huggingface.co/allenai/Olmo-3-1025-7B/blob/a81bae42db3975be1671e27b9c9a56da1a9f980f/tokenizer.json) | candidate | The base tokenizer has no embedded template, but the pinned adapter directory supplies `chat_template.jinja`. Its source-study role remains provisional. |
| Source mixture | [`allenai/Dolci-Think-SFT-7B@72ec0fe`](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B/tree/72ec0fe32428bada1ec686a9168a0681eecd2094) | candidate | The post names the dataset but not this revision or its sampling code. |
| Approximate 25K sample | [`dare-data/data/train_1.parquet@6bc8235`](https://huggingface.co/datasets/GaloisTheory123/dare-data/blob/6bc823578cb26ea8d5b433bdd5f9ee77f7f83a6e/data/train_1.parquet) | candidate | Its 25,007 rows, schema, split, and stable reconstructed identities are receipt-bound, but no card or producer receipt binds it to adapter `split-1`, declares its license, or explains the stratification. |
| Full-data adapter | [`dare-adapter/split-1@767c0ec`](https://huggingface.co/GaloisTheory123/dare-adapter/tree/767c0ec9f3be0c469ccec246511ca46ac2812c5e/split-1) | candidate | The adapter says rank 32 while the post says rank 64; its base revision is null. |
| Validation evaluation | [`validate_feelings.eval@519cbb9`](https://huggingface.co/datasets/GaloisTheory123/dare-results/blob/519cbb977bb13af93bc257d964743981fe75aa20/evals_for_writeup/baselines/slop_aware_judge/custom_sft/validate_feelings.eval) | candidate | The 100 prompts, responses, rubrics, and scores are present, but the producing checkout is dirty. |
| Refusal evaluation | [`refuse_then_redirect.eval@519cbb9`](https://huggingface.co/datasets/GaloisTheory123/dare-results/blob/519cbb977bb13af93bc257d964743981fe75aa20/evals_for_writeup/baselines/slop_aware_judge/custom_sft/refuse_then_redirect.eval) | candidate | It has 101 completed samples rather than the described 100, and clean evaluator code is not public. |
| Layer-16 SAE | [`olmo-3-saes@ad1f790`](https://huggingface.co/decoderesearch/olmo-3-saes/tree/ad1f790dd36d44b5b77b1d08ace6a6f48d121649/olmo-3-1025-7b/btk-mat-layer-16-k-100) | candidate | The architecture conversion, static dimensions, full weights, and one functional layer-16 hook/encode/decode path are receipt-bound; the training-time base revision remains unavailable. |
| Producing code | [`jrosseruk/dare`](https://github.com/jrosseruk/dare) | unavailable | The evaluation archive names this origin at short commit `db9d656` with `dirty=true`; both the repository API and `git ls-remote` returned not found. |

Every file-level LFS SHA-256 is recorded in the corresponding receipt under
[`configs/reconstruction/receipts/`](../../configs/reconstruction/receipts/).
Repository revisions, paths, and file hashes are separate fields so a stable
repository commit is not mistaken for proof of a file's source-organism role.

## Data and training observations

The official source mixture is public and ungated at the pinned revision. Its
viewer reports 2,268,178 rows, 156 Parquet shards, and the fields `messages`,
`dataset_source`, and `id`.

The separate source-study data repository contains five `train_*.parquet`
files. Reading their Parquet footer metadata gave 25,007, 25,006, 24,999,
24,995, and 24,993 rows, exactly 125,000 in total. A later exact-file audit of
`train_1.parquet` found `messages`, `dataset_source`, `orig_idx`, and `split`.
All 25,007 `orig_idx` values are unique, and every `split` value is 1. The
[named reconstruction receipt](../../configs/reconstruction/receipts/public-rank32-train1-reconstruction-v1.json)
therefore defines stable row identities from the pinned source revision plus
`orig_idx` and hashes both row order and row content. That identity rule is a
recorded reconstruction choice: the public repositories still do not provide
sampling code, a complete source-to-sample receipt, a license declaration, or
proof that this file produced adapter `split-1`.

The public `split-1` adapter is a 319,876,032-byte safetensors file. Its config
records rank 32, alpha 64, dropout 0.1, and all query, key, value, output, gate,
up, and down projections. Its trainer state records one epoch, 123 steps,
`train_batch_size=2`, top-level `num_input_tokens_seen=0`, and a final
cumulative logged `num_tokens=63,696,407`. The trainer-state field does not
establish per-device or effective batch size. The
[recipe-evidence receipt](../../configs/reconstruction/receipts/public-rank32-split1-recipe-evidence-v1.json)
binds these observations and the exact 123-point learning-rate and token
traces, but it does not recover a runnable recipe. The directory does not
publish the optimizer configuration, configured scheduler and warmup,
gradient accumulation, initial seed, precision, sequence policy, or data-order
receipt. The same immutable directory includes a standalone chat template with
SHA-256
`fb2b8736121ab4001112a9ccfb33c0081230536c861c706bf24b1b390964c1cd`;
the interim smoke uses that template rather than inventing one.

## Evaluation and SAE observations

The validation and refusal archives were fetched to a temporary directory and
their bytes matched the published LFS SHA-256 values. A dependency-free parser
verified every header/sample ID and recomputed the archived means: validation
has 100 questions with mean 1.48, while refusal has 101 with mean
4.326732673267327. The checked-in normalized scores each have a durable
extraction receipt that binds the archive hash, approved task-to-behavior map,
scorer, source revision and dirty state, and normalized output hash. Their
headers identify different short producing commits, and both explicitly mark
the checkout dirty. The public result bundle is therefore enough to recover the
evaluation banks and reference outputs, but not enough to rebuild the evaluator
from an immutable source tree. Anthropic's official model-version policy does
resolve the archived `claude-sonnet-4-6` name to a canonical pinned model ID
with fixed weights and configuration, but it does not freeze serving
infrastructure.

The layer-16 SAE receipt pins a 2,148,024,728-byte safetensors file with
`d_in=4096`, `d_sae=65536`, `k=100`, hook `model.layers.16`, and Matryoshka
widths 2,048, 16,384, and 65,536. Its `cfg.json` labels the architecture
`jumprelu`, while `runner_cfg.json` labels it `matryoshka_batchtopk`. This is
not a contradictory decoder convention: the pinned SAELens 6.42.0 source
explicitly documents that Matryoshka BatchTopK training SAEs are saved as
JumpReLU SAEs for inference, and the published safetensors header contains the
expected 65,536-element threshold tensor. The
[structural-compatibility receipt](../../configs/reconstruction/receipts/public-layer16-structural-compatibility-v1.json)
also matches the pinned candidate base's 4,096 hidden width and 32 layers to
the SAE's input width, layer-16 hook, and encoder/decoder shapes. The SAE
metadata still names no immutable base revision. A separate
[functional-smoke receipt](../../configs/reconstruction/receipts/public-layer16-functional-smoke-v1.json)
binds the explicitly pinned runtime file hashes and executes one finite
4,096-to-65,536-to-4,096 hook/encode/decode path. That mechanical compatibility
does not recover the SAE's training-time base revision or establish causal
feature utility.

## Audit boundary

The original audit used repository metadata, small JSON/configuration files,
Parquet footer reads, and two small evaluation archives in a temporary
directory. Later reconstruction downloaded the single pinned 116 MB
`train_1.parquet` file and produced the compact receipt above; it did not
download the 36 GB source mixture. The interim-smoke setup fetched the adapter
metadata and chat template, then downloaded the pinned base and rank-32 adapter
weights for local MPS generation. It did not download SAE weights, run
training, or produce scientific evaluation scores. The later static SAE check
downloaded only the pinned small metadata, release README, SAELens 6.42.0
documentation, and a 408-byte HTTP range containing the safetensors header. The
later functional smoke downloaded and hashed the full 2.1 GB SAE, rehashed the
pinned base and adapter runtime files before and after execution, and ran one
non-scientific CPU forward/encode/decode path. It did not select a feature,
intervene on an activation, call a provider, score behavior, run training, or
start the experiment timer.

The result is an R0 inventory, not an R1 reproduction. The exact blocker set is
generated by the divergence command and summarized in the
[Gate R2 memo](../../docs/specs/2026-08-14-gate-r2-decision-memo.md).
