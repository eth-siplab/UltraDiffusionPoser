#!/usr/bin/env bash
#
# DIP model: pretrain on noise-free AMASS, then finetune on the DIP-IMU train set.
# Reuses the shared clean-AMASS backbone (pretrains it once if missing).
# DIP has no translation ground truth, so finetune uses the pose-only loss set.
#
# Usage: ./train_scripts/train_dip.sh [gpu_id]
# Env:   DIPIMU_DIR (default data/processed_data/DIP_smooth5)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ensure_pretrain clean

run_finetune "$CLEAN_CONFIG" "$PRETRAIN_CKPT" \
    --eval_dataset dip-imu \
    --epochs 30 \
    --losses        simple_diff_loss smpl_6d_loss contact_loss \
    --loss_weights  5 1 0.1 \
    --loss_start_epoch 0 0 0
