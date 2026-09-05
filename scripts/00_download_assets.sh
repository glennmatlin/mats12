#!/usr/bin/env bash
# Off-clock generic prep: pull models + data to PACE shared storage.
# Storage rule: artifacts go to the shared project volume, never home/scratch.
set -euo pipefail

DEST="${1:?usage: 00_download_assets.sh /storage/ice-shared/cs7634/staff/mats12}"

# VERIFY exact ids/revisions before running (config.py markers):
# Revision pinned 2026-08-16 to the snapshot already staged on the cluster
# (recovered from .cache/huggingface download metadata; recorded in the
# extraction reports as weights identity).
hf download allenai/Olmo-3-1025-7B \
    --revision 15fe2e8b8b0fbf49416d3943ca6cf0174d0f2c07 \
    --local-dir "$DEST/models/olmo3-base"
# TODO: pin mid-train and stage1-only checkpoint revisions once confirmed
# (Dohun's configs or HF branch listing):
# hf download allenai/Olmo-3-1025-7B --revision <midtrain-rev> --local-dir "$DEST/models/olmo3-midtrain"
hf download allenai/Dolci-Think-SFT-7B --repo-type dataset --local-dir "$DEST/data/dolci-think-sft"

# Persona-extraction prompt sets:
git clone https://github.com/epfl-dlab/pretraining_persona "$DEST/vendor/pretraining_persona" || true
git clone https://github.com/safety-research/assistant-axis "$DEST/vendor/assistant-axis" || true

echo "done: $(du -sh "$DEST" | cut -f1)"
