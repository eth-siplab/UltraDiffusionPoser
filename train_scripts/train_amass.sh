#!/usr/bin/env bash
#
# Pretrain UDP on noise-free synthetic AMASS.
# This single backbone IS the DanceDB and TotalCapture model, and the starting
# point for the DIP finetune. Written to $LOG_ROOT/pretrain_amass so the DIP
# finetune can find and reuse it.
#
# Usage: ./train_scripts/train_amass.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_pretrain "$CLEAN_CONFIG" "$CLEAN_PRETRAIN_TS"
