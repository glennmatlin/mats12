#!/usr/bin/env bash
#SBATCH --job-name=mats12-retrain
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Template for one speedrun-SFT (re)train + behavior eval (~2.6h on 8xH200 per
# the anchor post; budget 4h wall). Node-local $TMPDIR for staging; artifacts to
# shared storage. Parameterized by a filter manifest (which doc ids to drop).
set -euo pipefail

: "${FILTER_MANIFEST:?path to doc-id drop list (or 'none')}"
: "${RUN_TAG:?short run tag, e.g. vf-judge25-decontam}"

echo "TODO(sprint): invoke training entrypoint with $FILTER_MANIFEST -> results/$RUN_TAG"
exit 1
