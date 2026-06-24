#!/usr/bin/env bash
#
# UIP-DB-ft model: pretrain on noisy AMASS, then finetune on the UIP-DB
# (SIGGRAPH UWB-IMU) train set. Reuses the shared noisy-AMASS backbone
# (pretrains it once if missing).
#
# The finetune dataset is selected via UWBIMU_DIR (config.paths.uwbimu_dir).
#
# Usage: ./train_scripts/train_uip_ft.sh [gpu_id]
# Env:   UWBIMU_DIR (default data/processed_data/UWB_IMU/SIGGRAPH_dataset)
#        UWB_GUIDANCE_LAMBDA (inference-time UWB guidance, default 50)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export UWBIMU_DIR="${UWBIMU_DIR:-data/processed_data/UWB_IMU/SIGGRAPH_dataset}"
echo ">>> UIP-DB finetune data (UWBIMU_DIR) = $UWBIMU_DIR"

# Finetune from the final-epoch ("last") noisy backbone instead of the "best"
# (lowest val total_loss) one: best-model selection can lock onto an early epoch
# because smpl_tran_vel_loss activates at epoch 10 and inflates val total_loss.
export PRETRAIN_USE_LAST="${PRETRAIN_USE_LAST:-1}"

ensure_pretrain noisy

run_finetune "$NOISY_CONFIG" "$PRETRAIN_CKPT" \
    --eval_dataset uwb-imu \
    --epochs 5 \
    --uwb_guidance_lambda "${UWB_GUIDANCE_LAMBDA:-50}"
