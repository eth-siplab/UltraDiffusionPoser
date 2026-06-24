#!/usr/bin/env bash
#
# Shared setup for the UDP evaluation scripts. Sourced by every
# eval_scripts/eval_*.sh, not meant to be run on its own.
#
# Override any of these by exporting them before calling a script, e.g.:
#   CONDA_SH=~/anaconda3/etc/profile.d/conda.sh UDP_ENV=myenv ./eval_scripts/eval_tc.sh
#
set -eo pipefail

# Resolve the repo root (parent of this script's directory) and work from there
# so that all relative data/output paths resolve no matter where you invoke from.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_COMMON_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Path to the compiled rbdl python bindings (needed for forward kinematics in the
# evaluator). Defaults to where install.sh builds it: a sibling of the repo root.
# Override by exporting RBDL_PYTHON_PATH before calling an eval script.
RBDL_PYTHON_PATH="${RBDL_PYTHON_PATH:-$(dirname "$REPO_ROOT")/rbdl/rbdl-build/python}"

# Activate the conda environment (skip silently if conda isn't where we expect;
# in that case just make sure the right env is active before running).
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
# evaluator (forward kinematics), so warn loudly if the build is missing.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
if [ -d "$RBDL_PYTHON_PATH" ]; then
    export PYTHONPATH="$PYTHONPATH:$RBDL_PYTHON_PATH"
else
    echo "WARNING: rbdl python bindings not found at '$RBDL_PYTHON_PATH'." >&2
    echo "         Build them with ./install.sh or export RBDL_PYTHON_PATH=<your rbdl-build/python>." >&2
fi

# Headless rendering needs a display. Uncomment / point at your Xvfb if needed.
# export DISPLAY="${DISPLAY:-:0}"

EVAL_SAVE_DIR="${EVAL_SAVE_DIR:-output/evaluation_res_UDP}"
DEVICE="${DEVICE:-cuda}"

# run_eval <ckpt_dir> <data_dir> <exp_name> [extra evaluator args...]
# Results (CSV + translation-error plot) land in $EVAL_SAVE_DIR/<exp_name>/.
run_eval() {
    local ckpt="$1"; local data_dir="$2"; local exp_name="$3"; shift 3
    echo ">>> Evaluating ckpt=$ckpt on data=$data_dir (exp=$exp_name)"
    python modules/evaluate/evaluator.py \
        --network UDP \
        --ckpt_path "$ckpt" \
        --data_dir "$data_dir" \
        --exp_name "$exp_name" \
        --eval_save_dir "$EVAL_SAVE_DIR" \
        --device "$DEVICE" \
        --flush_cache \
        "$@"
}
