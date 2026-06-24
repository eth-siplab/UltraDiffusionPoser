#!/usr/bin/env bash
#
# Evaluate the base UDP model on the GIP-DB (Multi-UWB-Merged) test set.
# Usage: ./eval_scripts/eval_gip.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval \
    "data/checkpoints/uip_gip/ckpt" \
    "data/processed_data/Multi-UWB-Merged" \
    "gipdb" \
    --eval_trans
