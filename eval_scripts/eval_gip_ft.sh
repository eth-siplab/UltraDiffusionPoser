#!/usr/bin/env bash
#
# Evaluate the fine-tuned UDP model on the GIP-DB (Multi-UWB-Merged) test set.
# Usage: ./eval_scripts/eval_gip_ft.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval \
    "data/checkpoints/gip_ft/ckpt" \
    "data/processed_data/Multi-UWB-Merged" \
    "gipdb_ft" \
    --eval_trans
