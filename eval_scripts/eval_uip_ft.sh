#!/usr/bin/env bash
#
# Evaluate the fine-tuned UDP model on the UIP-DB (SIGGRAPH UWB-IMU) test set.
# Usage: ./eval_scripts/eval_uip_ft.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval \
    "data/checkpoints/uip_ft/ckpt" \
    "data/processed_data/UWB_IMU/SIGGRAPH_dataset" \
    "uipdb_ft" \
    --eval_trans
