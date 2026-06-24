#!/usr/bin/env bash
#
# Evaluate the RETRAINED models (under output/trainUDP/), the counterpart to
# eval_all.sh which evaluates the released/official checkpoints (data/checkpoints/).
# Compare the my_* results here against the official results in results_orig.txt.
#
# 5 models -> 7 evaluations (the two backbones are each scored on two datasets).
# Results land in output/evaluation_res_UDP/<exp_name>/.
#
# Usage: ./eval_scripts/eval_mine.sh [gpu_id]
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# clean AMASS backbone (role: dancedb_tc) -- already matched the paper
run_eval output/trainUDP/pretrain_amass/ckpt              data/processed_data/AMASS_syn/test_split     my_dancedb  --eval_trans
run_eval output/trainUDP/pretrain_amass/ckpt              data/processed_data/TotalCapture             my_tc       --eval_trans

# DIP finetune (pose only)
run_eval output/trainUDP/2026_06_23_03_40_28/ckpt        data/processed_data/DIP                       my_dip

# noisy AMASS backbone (role: uip_gip), evaluated at epoch 49 to match the
# original (whose "best" was epoch 49). Our retrain's "best" was epoch 8, so we
# point at the staged epoch-49 ("last") checkpoint instead.
run_eval output/trainUDP/pretrain_amass_noisy_ep49/ckpt  data/processed_data/UWB_IMU/SIGGRAPH_dataset  my_uipdb    --eval_trans
run_eval output/trainUDP/pretrain_amass_noisy_ep49/ckpt  data/processed_data/Multi-UWB-Merged          my_gipdb    --eval_trans

# UWB finetunes (retrained from the epoch-49 noisy backbone; gip now on Multi-UWB-Merged)
run_eval output/trainUDP/2026_06_23_18_03_56/ckpt        data/processed_data/UWB_IMU/SIGGRAPH_dataset  my_uipdb_ft --eval_trans
run_eval output/trainUDP/2026_06_23_18_04_15/ckpt        data/processed_data/Multi-UWB-Merged          my_gipdb_ft --eval_trans
