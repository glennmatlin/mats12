#!/bin/bash
# Reusable PACE launcher for one base-response shard (array element).
# On ICE: no --account and no --partition (free tier, auto-routed).
#
# Site-and-run facts are supplied at submit time, never hardcoded here:
#   sbatch --time=<HH:MM:SS> \
#     --array=0-$((CF_SHARD_COUNT - 1))%4 \
#     --export=ALL,CF_CHECKOUT=...,CF_CONFIG=...,CF_PACKAGE=...,CF_PARQUET=...,CF_BASE_SNAPSHOT=...,CF_TOKENIZER_SNAPSHOT=...,CF_OUTPUT_DIR=...,CF_SHARD_COUNT=... \
#     scripts/pace/base_response_shard.sbatch
#
# The runner is resume-safe: resubmitting an interrupted element replays
# existing valid samples and generates only the gap.
#SBATCH --job-name=base-responses
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=96G

set -euo pipefail

# Logs stay in the submission directory: the checkout must remain byte-clean
# (finalize verifies it with git status --untracked-files=all).
mkdir -p logs
cd "${CF_CHECKOUT:?set CF_CHECKOUT to the verified exact-revision checkout}"

uv run --frozen --extra inference concept-filtering base-responses run-shard \
  --config "${CF_CONFIG:?}" \
  --package "${CF_PACKAGE:?}" \
  --parquet "${CF_PARQUET:?}" \
  --base-snapshot "${CF_BASE_SNAPSHOT:?}" \
  --tokenizer-snapshot "${CF_TOKENIZER_SNAPSHOT:?}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:?}" \
  --shard-count "${CF_SHARD_COUNT:?}" \
  --output-dir "${CF_OUTPUT_DIR:?}" \
  --device cuda --dtype bfloat16
