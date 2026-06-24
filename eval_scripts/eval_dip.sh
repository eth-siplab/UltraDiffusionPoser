#!/usr/bin/env bash
#
# Evaluate UDP on the DIP-IMU test set (pose metrics only).
# Usage: ./eval_scripts/eval_dip.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval \
    "data/checkpoints/dip/ckpt" \
    "data/processed_data/DIP" \
    "dip"
