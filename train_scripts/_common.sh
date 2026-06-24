#!/usr/bin/env bash
#
# Shared setup for the UDP TRAINING scripts. Sourced by every
# train_scripts/train_*.sh, not meant to be run on its own.
#
# Override any of these by exporting them before calling a script, e.g.:
#   WANDB_MODE=online UDP_ENV=myenv ./train_scripts/train_dip.sh 0
#
set -eo pipefail

# Resolve the repo root (parent of this script's directory) and work from there
# so all relative data/output/config paths resolve regardless of where you invoke.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_COMMON_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Path to the compiled rbdl python bindings (needed for forward kinematics).
# Defaults to where install.sh builds it: a sibling of the repo root.
# Override by exporting RBDL_PYTHON_PATH before calling a training script.
RBDL_PYTHON_PATH="${RBDL_PYTHON_PATH:-$(dirname "$REPO_ROOT")/rbdl/rbdl-build/python}"

# Activate the conda environment (skip silently if conda isn't where we expect).
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "${UDP_ENV:-UDP}"
fi

# GPU to use: first CLI argument, else $CUDA_DEVICE, else 0.
export CUDA_VISIBLE_DEVICES="${1:-${CUDA_DEVICE:-0}}"

# SMPL body model used by the network and aitviewer.
export SMPL_MODEL_PATH="${SMPL_MODEL_PATH:-$REPO_ROOT/data/smpl_m_lbs_10_207_0_v1.0.0.pkl}"

# Make the repo + the compiled rbdl bindings importable. rbdl is required by the
# in-training evaluator (forward kinematics), so warn loudly if the build is missing.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
if [ -d "$RBDL_PYTHON_PATH" ]; then
    export PYTHONPATH="$PYTHONPATH:$RBDL_PYTHON_PATH"
else
    echo "WARNING: rbdl python bindings not found at '$RBDL_PYTHON_PATH'." >&2
    echo "         Build them with ./install.sh or export RBDL_PYTHON_PATH=<your rbdl-build/python>." >&2
fi

# Training calls wandb.init(project="UDP"). Default to offline so it never blocks
# on a login; export WANDB_MODE=online to upload, or =disabled to turn it off.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Where checkpoints/logs land (Train_model.py appends the run timestamp).
LOG_ROOT="${LOG_ROOT:-output/trainUDP}"

# Base training configs (verified against the wandb run configs).
CLEAN_CONFIG="${CLEAN_CONFIG:-config/train_config_udp.ini}"        # noise-free AMASS pretrain
NOISY_CONFIG="${NOISY_CONFIG:-config/train_config_udp_uipgip.ini}" # noisy AMASS pretrain

# Deterministic output dirs for the two *shared* pretrains, so the finetune scripts
# can find and reuse them instead of pretraining the same backbone repeatedly.
CLEAN_PRETRAIN_TS="${CLEAN_PRETRAIN_TS:-pretrain_amass}"
NOISY_PRETRAIN_TS="${NOISY_PRETRAIN_TS:-pretrain_amass_noisy}"

# Headless preview rendering (aitviewer) needs a display. Set XVFB=1 to wrap the
# training process in `xvfb-run -a` when no DISPLAY is available.
_py() {
    if [ "${XVFB:-0}" = "1" ] && command -v xvfb-run >/dev/null 2>&1; then
        xvfb-run -a python "$@"
    else
        python "$@"
    fi
}

# find_best_ckpt <log_dir> -> prints the baseline best-model checkpoint, if any.
find_best_ckpt() {
    ls -1 "$1/ckpt/baseline_diffusion_all_best_model_"*.pt 2>/dev/null | sort -V | tail -n1 || true
}

# find_last_ckpt <log_dir> -> prints the baseline last-model (final-epoch) checkpoint, if any.
find_last_ckpt() {
    ls -1 "$1/ckpt/baseline_diffusion_all_last_model_"*.pt 2>/dev/null | sort -V | tail -n1 || true
}

# resolve_pretrain_ckpt <log_dir> -> prints the checkpoint to finetune from.
# Honors PRETRAIN_USE_LAST=1 to select the final-epoch ("last") checkpoint instead
# of the "best" (lowest val total_loss) one. The "best" criterion can lock onto an
# early epoch when a loss term activates mid-run (e.g. smpl_tran_vel_loss at epoch
# 10 inflates total_loss), so the noisy-AMASS finetunes prefer the last checkpoint.
resolve_pretrain_ckpt() {
    if [ "${PRETRAIN_USE_LAST:-0}" = "1" ]; then
        find_last_ckpt "$1"
    else
        find_best_ckpt "$1"
    fi
}

# run_pretrain <config> <timestamp> [extra args...]
# Pretrains from scratch (pretrain_model="") into $LOG_ROOT/<timestamp>.
run_pretrain() {
    local config="$1"; local ts="$2"; shift 2
    echo ">>> Pretraining with $config -> $LOG_ROOT/$ts (gpu $CUDA_VISIBLE_DEVICES)"
    _py Train_model.py \
        --config_file "$config" \
        --network UDP \
        --log_dir "$LOG_ROOT" \
        --timestamp "$ts" \
        --pretrain_model "" \
        "$@"
}

# PRETRAIN_CKPT is set by ensure_pretrain() for the caller to consume.
PRETRAIN_CKPT=""

# ensure_pretrain <clean|noisy>
#   1. If {CLEAN,NOISY}_PRETRAIN_CKPT is exported, use that checkpoint.
#   2. Else if a pretrain already exists at $LOG_ROOT/<fixed-ts>, reuse it.
#   3. Else run the matching pretrain script once, then use its checkpoint.
ensure_pretrain() {
    local kind="$1" override ts script
    if [ "$kind" = "clean" ]; then
        override="${CLEAN_PRETRAIN_CKPT:-}"; ts="$CLEAN_PRETRAIN_TS"; script="train_amass.sh"
    else
        override="${NOISY_PRETRAIN_CKPT:-}"; ts="$NOISY_PRETRAIN_TS"; script="train_amass_noisy.sh"
    fi

    if [ -n "$override" ]; then
        [ -f "$override" ] || { echo "ERROR: ${kind} pretrain checkpoint not found: $override" >&2; return 1; }
        PRETRAIN_CKPT="$override"
        echo ">>> Using provided $kind pretrain: $PRETRAIN_CKPT"
        return 0
    fi

    PRETRAIN_CKPT="$(resolve_pretrain_ckpt "$LOG_ROOT/$ts")"
    if [ -n "$PRETRAIN_CKPT" ]; then
        echo ">>> Reusing existing $kind pretrain: $PRETRAIN_CKPT"
        return 0
    fi

    echo ">>> No $kind pretrain found at $LOG_ROOT/$ts — running $script first ..."
    "$_COMMON_DIR/$script" "$CUDA_VISIBLE_DEVICES"
    PRETRAIN_CKPT="$(resolve_pretrain_ckpt "$LOG_ROOT/$ts")"
    [ -n "$PRETRAIN_CKPT" ] || { echo "ERROR: $kind pretrain produced no checkpoint in $LOG_ROOT/$ts/ckpt" >&2; return 1; }
    echo ">>> Pretrained $kind model: $PRETRAIN_CKPT"
}

# run_finetune <config> <pretrain_ckpt> [extra args...]
# Loads <pretrain_ckpt> and finetunes (fresh timestamp picked by Train_model.py).
run_finetune() {
    local config="$1"; local pretrain="$2"; shift 2
    echo ">>> Finetuning from $pretrain (gpu $CUDA_VISIBLE_DEVICES)"
    _py Train_model.py \
        --config_file "$config" \
        --network UDP \
        --log_dir "$LOG_ROOT" \
        --pretrain_model "$pretrain" \
        --finetune \
        --training_phase finetune_diffusion_all \
        "$@"
}
