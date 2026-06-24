#!/usr/bin/env bash
#
# Train every UDP model in sequence. The two pretrains run first, then the
# three finetunes reuse them (so the clean and noisy backbones are each trained
# exactly once).
#
# Produced models:
#   train_amass        -> DanceDB + TotalCapture backbone (clean AMASS)
#   train_amass_noisy  -> UIP-DB + GIP-DB backbone (noisy AMASS)
#   train_dip          -> DIP (clean backbone, finetuned on DIP-IMU)
#   train_uip_ft       -> UIP-DB-ft (noisy backbone, finetuned on SIGGRAPH UWB)
#   train_gip_ft       -> GIP-DB-ft (noisy backbone, finetuned on Multi-UWB-Merged)
#
# Usage: ./train_scripts/train_all.sh [gpu_id]
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"

for s in train_amass train_amass_noisy train_dip train_uip_ft train_gip_ft; do
    echo "================================================================"
    echo "Running $s"
    echo "================================================================"
    "$HERE/$s.sh" "$GPU"
done
