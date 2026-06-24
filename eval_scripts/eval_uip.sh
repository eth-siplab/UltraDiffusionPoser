#!/usr/bin/env bash
#
# Evaluate the base UDP model on the UIP-DB (SIGGRAPH UWB-IMU) test set.
# Usage: ./eval_scripts/eval_uip.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval \
    "data/checkpoints/uip_gip/ckpt" \
    "data/processed_data/UWB_IMU/SIGGRAPH_dataset" \
    "uipdb" \
    --eval_trans
