#!/usr/bin/env bash
#
# Pretrain UDP on noisy synthetic AMASS (IMU acc/ori noise + bias, UWB noise).
# This single backbone IS the UIP-DB and GIP-DB model, and the starting point for
# both UWB finetunes. Written to $LOG_ROOT/pretrain_amass_noisy so the UIP/GIP
# finetunes can find and reuse it (it is trained only once).
#
# Usage: ./train_scripts/train_amass_noisy.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_pretrain "$NOISY_CONFIG" "$NOISY_PRETRAIN_TS"
