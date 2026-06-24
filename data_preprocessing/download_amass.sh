#!/usr/bin/env bash
#
# Make the SMPL-H AMASS datasets this project uses available under the names
# config/preprocess.py expects, WITHOUT renaming or touching any AMASS data the
# user already has.
#
# It populates a "view" directory (config.paths.raw_amass_dir, override with
# AMASS_RAW_DIR) with one entry per required dataset, named with the CONFIG name:
#   * already there        -> left as-is
#   * found in AMASS_SRC_DIR (your existing AMASS, possibly under the new server
#     names like HDM05/BMLrub/PosePrior) -> a symlink into it (source untouched)
#   * otherwise            -> downloaded from amass.is.tue.mpg.de into the view
#
# Only the datasets in config.amass_data + config.amass_test_data are handled
# (not all of AMASS). Requires an amass.is.tue.mpg.de account for any download.
#
# Usage: ./data_preprocessing/download_amass.sh
#        (or sourced by download_all.sh, which calls amass_run)
# Env:   AMASS_RAW_DIR  the view dir the code reads (default: config.paths.raw_amass_dir)
#        AMASS_SRC_DIR  an existing AMASS install to symlink from (read-only, never modified)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# config/local name -> current AMASS server file basename (only those that differ).
# Also used as the alternate folder name to look for inside AMASS_SRC_DIR.
declare -A AMASS_RENAME=(
    [MPI_HDM05]=HDM05
    [MPI_mosh]=MoSh
    [Transitions_mocap]=Transitions
    [SSM_synced]=SSM
    [Eyes_Japan_Dataset]=EyesJapanDataset
    [TCD_handMocap]=TCDHands
    [BioMotionLab_NTroje]=BMLrub
    [MPI_Limits]=PosePrior
    [DFaust_67]=DFaust
)
AMASS_PREFIX="amass_per_dataset/smplh/gender_specific/mosh_results"

# True if <dir> (following symlinks) contains at least one *_poses.npz.
has_poses() { [ -n "$(find -L "$1" -name '*_poses.npz' -print -quit 2>/dev/null)" ]; }

# Figure out the view dir, what's already present, what can be symlinked from an
# existing install, and what still needs downloading. Populates AMASS_VIEW and
# AMASS_TO_DOWNLOAD. Idempotent (symlinking with `ln -sfn`); all chatter -> stderr
# so amass_accounts can be captured cleanly. Cached after the first call.
amass_resolve() {
    [ -n "${AMASS_RESOLVED:-}" ] && return 0

    local _cfg NEEDED SRC ds cand found srcabs present=0 linked=0
    mapfile -t _cfg < <(python - <<'PY'
from config.config import paths, amass_data, amass_test_data
print(paths.raw_amass_dir)
for d in list(amass_data) + list(amass_test_data):
    print(d)
PY
)
    [ -n "${_cfg[0]:-}" ] || { echo "ERROR: could not read AMASS config from config.py" >&2; return 1; }
    AMASS_VIEW="${AMASS_RAW_DIR:-${_cfg[0]}}"
    NEEDED=("${_cfg[@]:1}")
    SRC="${AMASS_SRC_DIR:-}"
    AMASS_TO_DOWNLOAD=()

    mkdir -p "$AMASS_VIEW"
    echo "AMASS view dir   : $AMASS_VIEW   (config.paths.raw_amass_dir)" >&2
    [ -n "$SRC" ] && echo "AMASS source dir : $SRC   (read-only; will be symlinked, never modified)" >&2

    for ds in "${NEEDED[@]}"; do
        if has_poses "$AMASS_VIEW/$ds"; then
            present=$((present + 1)); continue
        fi
        if [ -n "$SRC" ]; then
            found=""
            for cand in "$ds" "${AMASS_RENAME[$ds]:-}"; do
                [ -n "$cand" ] || continue
                if has_poses "$SRC/$cand"; then found="$cand"; break; fi
            done
            if [ -n "$found" ]; then
                srcabs="$(cd "$SRC/$found" && pwd -P)"
                ln -sfn "$srcabs" "$AMASS_VIEW/$ds"
                echo "  link  $ds -> $srcabs" >&2
                linked=$((linked + 1)); continue
            fi
        fi
        AMASS_TO_DOWNLOAD+=("$ds")
    done

    echo "present: $present | linked: $linked | to download: ${#AMASS_TO_DOWNLOAD[@]}" >&2
    [ "${#AMASS_TO_DOWNLOAD[@]}" -gt 0 ] && echo "To download: ${AMASS_TO_DOWNLOAD[*]}" >&2
    if [ "${#AMASS_TO_DOWNLOAD[@]}" -gt 0 ] && [ -z "$SRC" ]; then
        echo "(Tip: if you already have AMASS elsewhere, set AMASS_SRC_DIR to symlink it" >&2
        echo " instead of downloading — your existing data is never renamed or modified.)" >&2
    fi
    AMASS_RESOLVED=1
}

amass_present()  { amass_resolve && [ "${#AMASS_TO_DOWNLOAD[@]}" -eq 0 ]; }
amass_accounts() { amass_resolve && [ "${#AMASS_TO_DOWNLOAD[@]}" -gt 0 ] && echo amass; }

amass_run() {
    amass_resolve
    if [ "${#AMASS_TO_DOWNLOAD[@]}" -eq 0 ]; then
        echo "All required AMASS datasets are available (no download needed)."
        return 0
    fi

    local ds server out stage npz dsroot
    for ds in "${AMASS_TO_DOWNLOAD[@]}"; do
        server="${AMASS_RENAME[$ds]:-$ds}"
        out="$DOWNLOAD_CACHE/amass_smplh/${server}.tar.bz2"
        echo "================================================================"
        echo ">>> $ds  (server file: ${server}.tar.bz2)"
        tue_download amass \
            "https://download.is.tue.mpg.de/download.php?domain=amass&sfile=${AMASS_PREFIX}/${server}.tar.bz2" \
            "$out"

        # Extract into a staging dir on the SAME filesystem (fast rename), locate the
        # dataset root via a *_poses.npz, and install it under the CONFIG name.
        stage="$AMASS_VIEW/.amass_stage.$$"
        rm -rf "$stage"; mkdir -p "$stage"
        tar -xjf "$out" -C "$stage"
        npz="$(find "$stage" -name '*_poses.npz' -print -quit)"
        if [ -z "$npz" ]; then
            echo "ERROR: no *_poses.npz found inside $out" >&2; rm -rf "$stage"; return 1
        fi
        dsroot="$(dirname "$(dirname "$npz")")"   # <root>/<subject>/<motion>_poses.npz -> <root>
        rm -rf "$AMASS_VIEW/$ds"
        mv "$dsroot" "$AMASS_VIEW/$ds"
        rm -rf "$stage"
        echo "    -> $AMASS_VIEW/$ds ($(find "$AMASS_VIEW/$ds" -name '*_poses.npz' | wc -l) sequences)"
    done

    echo "AMASS ready in $AMASS_VIEW."
    echo "(Cached archives in $DOWNLOAD_CACHE/amass_smplh — safe to delete.)"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    amass_resolve
    if amass_present; then amass_run; exit 0; fi
    if ! ask_yes_no "Download the ${#AMASS_TO_DOWNLOAD[@]} remaining AMASS SMPL-H datasets into $AMASS_VIEW?"; then
        skip_note "the remaining AMASS datasets" "paths.raw_amass_dir (or AMASS_RAW_DIR / AMASS_SRC_DIR)" "$AMASS_VIEW"
        exit 0
    fi
    amass_run
fi
