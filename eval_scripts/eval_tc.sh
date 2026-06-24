#!/usr/bin/env bash
#
# Evaluate UDP on the TotalCapture test set (pose + global translation).
# Usage: ./eval_scripts/eval_tc.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval \
    "data/checkpoints/dancedb_tc/ckpt" \
    "data/processed_data/TotalCapture" \
    "tc" \
    --eval_trans
