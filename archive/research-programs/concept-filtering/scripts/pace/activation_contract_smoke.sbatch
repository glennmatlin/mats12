#!/bin/bash
# Reusable PACE launcher for the bounded layer-16 activation contract smoke.
# On ICE: no --account and no --partition (free tier, auto-routed).
#
# Site-and-run facts are supplied at submit time, never hardcoded here:
#   sbatch --time=<HH:MM:SS> \
#     --export=ALL,CF_CHECKOUT=...,CF_CONFIG=...,CF_GENERATION_CONFIG=...,CF_PACKAGE=...,CF_PARQUET=...,CF_BASE_SNAPSHOT=...,CF_TOKENIZER_SNAPSHOT=...,CF_OUTPUT_DIR=...,CF_RECEIPT=...,CF_CODE_REVISION=... \
#     scripts/pace/activation_contract_smoke.sbatch
#
# One example, two forwards. The receipt is published absent-only, so a rerun
# must name a newly declared absent CF_RECEIPT path.
#SBATCH --job-name=activation-contract
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=96G

set -euo pipefail

# Logs and the receipt stay outside the checkout: the smoke binds its executor
# revision and fails closed on any untracked file in the tree.
mkdir -p logs
cd "${CF_CHECKOUT:?set CF_CHECKOUT to the verified exact-revision checkout}"

uv run --frozen --extra inference concept-filtering activation-contract-smoke \
  --config "${CF_CONFIG:?}" \
  --generation-config "${CF_GENERATION_CONFIG:?}" \
  --package "${CF_PACKAGE:?}" \
  --parquet "${CF_PARQUET:?}" \
  --base-snapshot "${CF_BASE_SNAPSHOT:?}" \
  --tokenizer-snapshot "${CF_TOKENIZER_SNAPSHOT:?}" \
  --output-dir "${CF_OUTPUT_DIR:?}" \
  --output "${CF_RECEIPT:?}" \
  --code-revision "${CF_CODE_REVISION:?}" \
  --device cuda --dtype bfloat16
