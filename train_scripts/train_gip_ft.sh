#!/usr/bin/env bash
#
# GIP-DB-ft model: pretrain on noisy AMASS, then finetune on the GIP-DB
# (Multi-UWB-Merged) train set. Reuses the SAME shared noisy-AMASS backbone as
# the UIP finetune (pretrains it once if missing).
#
# Identical config to train_uip_ft.sh; the ONLY difference is the finetune
# dataset, selected via UWBIMU_DIR (config.paths.uwbimu_dir).
#
# Usage: ./train_scripts/train_gip_ft.sh [gpu_id]
# Env:   UWBIMU_DIR (default data/processed_data/Multi-UWB-Merged)
#        UWB_GUIDANCE_LAMBDA (inference-time UWB guidance, default 50)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export UWBIMU_DIR="${UWBIMU_DIR:-data/processed_data/Multi-UWB-Merged}"
echo ">>> GIP-DB finetune data (UWBIMU_DIR) = $UWBIMU_DIR"

# Finetune from the final-epoch ("last") noisy backbone instead of the "best"
# (lowest val total_loss) one: best-model selection can lock onto an early epoch
# because smpl_tran_vel_loss activates at epoch 10 and inflates val total_loss.
export PRETRAIN_USE_LAST="${PRETRAIN_USE_LAST:-1}"

ensure_pretrain noisy

run_finetune "$NOISY_CONFIG" "$PRETRAIN_CKPT" \
    --eval_dataset uwb-imu \
    --epochs 5 \
    --uwb_guidance_lambda "${UWB_GUIDANCE_LAMBDA:-50}"
