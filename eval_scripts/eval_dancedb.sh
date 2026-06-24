#!/usr/bin/env bash
#
# Evaluate UDP on the AMASS DanceDB test split (with body shape given).
# Usage: ./eval_scripts/eval_dancedb.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_eval \
    "data/checkpoints/dancedb_tc/ckpt" \
    "data/processed_data/AMASS_syn/test_split" \
    "dancedb" \
    --eval_trans
