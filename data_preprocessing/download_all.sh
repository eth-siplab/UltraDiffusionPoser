#!/usr/bin/env bash
#
# Fetch the body model + datasets + checkpoints. Designed to be a "set it up and
# walk away" flow: ALL the interaction happens up front, then it downloads
# unattended.
#
#   1. choose "everything" or pick datasets individually;
#   2. it checks what's already on disk and which logins are still needed;
#   3. you enter those credentials once (the dip account covers BOTH the DIP
#      dataset and the TotalCapture ground truth — asked only once);
#   4. everything downloads with no further prompts.
#
# Credentials live only in non-exported shell variables of this single process
# (never `export`ed into the environment, never written to disk) — the sub-steps
# are sourced and run in-process, so nothing leaks to other programs.
#
# Usage: ./data_preprocessing/download_all.sh
# Env:   AUTO_YES=1 select everything and accept all prompts (non-interactive;
#        pre-seed logins via TUE_*_USERNAME/PASSWORD, CVSSP_USERNAME/PASSWORD).
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the helpers + every step so we can call their *_present/_accounts/_run
# functions in this process (the steps' own "run standalone" blocks stay dormant
# when sourced).
source "$HERE/_common.sh"
for s in download_smpl download_uip download_gip download_dip download_tc download_amass download_checkpoints; do
    source "$HERE/$s.sh"
done

# step key -> function prefix + human label.
STEPS=(smpl uip gip dip tc amass checkpoints)
declare -A LABEL=(
    [smpl]="SMPL v1.0.0 body model"
    [uip]="UIP-DB processed data (SIGGRAPH UWB-IMU)"
    [gip]="GIP two-person UWB dataset"
    [dip]="DIP-IMU dataset (~2.6 GB)"
    [tc]="TotalCapture (official S1..S5 + ground truth)"
    [amass]="AMASS SMPL-H datasets"
    [checkpoints]="released model checkpoints"
)

# ---------------------------------------------------------------- 1. selection ---
selected=()
if [ "${AUTO_YES:-0}" = "1" ]; then
    selected=("${STEPS[@]}")
else
    echo "What would you like to download?"
    echo "  1) Everything"
    echo "  2) Choose individually"
    read -r -p "Select [1/2] (default 1): " _choice
    if [ "$_choice" = "2" ]; then
        echo
        for s in "${STEPS[@]}"; do
            ask_yes_no "Include ${LABEL[$s]}?" && selected+=("$s")
        done
    else
        selected=("${STEPS[@]}")
    fi
fi

[ "${#selected[@]}" -gt 0 ] || { echo "Nothing selected — nothing to do."; exit 0; }

# --------------------------------------------------- 2. plan logins (presence) ---
echo
echo "Checking what's already on disk and which logins are needed ..."
_accts=()
for s in "${selected[@]}"; do
    mapfile -t _a < <("${s}_accounts")
    [ "${#_a[@]}" -gt 0 ] && _accts+=("${_a[@]}")
done
mapfile -t uniq_accts < <(printf '%s\n' "${_accts[@]}" | grep -v '^$' | sort -u)

# ----------------------------------------------- 3. collect all credentials now ---
echo
if [ "${#uniq_accts[@]}" -gt 0 ]; then
    echo "Enter the logins below. After this, downloading runs unattended — you can step away."
    echo "(The dip account covers both the DIP dataset and the TotalCapture ground truth.)"
    echo
    for a in "${uniq_accts[@]}"; do
        case "$a" in
            smpl|dip|amass) tue_login "$a" ;;
            cvssp)          cvssp_login ;;
        esac
    done
else
    echo "No logins required — everything selected is public or already present."
fi

# ------------------------------------------------------- 4. download unattended ---
echo
echo "================================================================"
echo "All prompts done — downloading: ${selected[*]}"
echo "================================================================"
for s in "${selected[@]}"; do
    echo
    echo ">>> ${LABEL[$s]}"
    "${s}_run"
done

echo
echo "All requested datasets processed."
