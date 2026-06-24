#!/usr/bin/env bash
#
# Download the released UDP model weights (the evaluated checkpoints) from Google
# Drive into data/checkpoints/, where the eval scripts expect them:
#   data/checkpoints/{dancedb_tc,dip,uip_gip,uip_ft,gip_ft}/ckpt/*.pt
#                                                       (+ model_args.json, config.ini)
#
# Usage: ./data_preprocessing/download_checkpoints.sh
#        (or sourced by download_all.sh, which calls checkpoints_run)
# Env:   CKPT_DIR  target dir (default data/checkpoints)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Just the Drive file ID (the part after /file/d/ in the share link). We build the
# uc?id= download URL from it below: passing a /file/d/<id> link straight to gdown
# (even with --fuzzy) makes it fetch the Drive HTML page instead of the file.
GDRIVE_ID="1fjOovPqk7DyQMSmAcrItF7uU1FtYUUf9"

ckpt_dir() { echo "${CKPT_DIR:-$DATA_DIR/checkpoints}"; }

checkpoints_present() {
    local d; d="$(ckpt_dir)"
    [ -d "$d/dancedb_tc/ckpt" ] && [ -d "$d/uip_gip/ckpt" ]
}

# Google Drive download — no login required.
checkpoints_accounts() { :; }

checkpoints_run() {
    local CKPT STAGE archive EXTRACT ma root
    CKPT="$(ckpt_dir)"
    if checkpoints_present; then echo "Model weights already present in $CKPT"; return 0; fi

    mkdir -p "$CKPT"
    STAGE="$(mktemp -d)"
    trap 'rm -rf "$STAGE"' RETURN

    # Download the single Drive file (gdown keeps the original filename/extension).
    ( cd "$STAGE" && gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" )
    archive="$(find "$STAGE" -maxdepth 1 -type f -print -quit)"
    [ -n "$archive" ] || { echo "ERROR: download produced no file" >&2; return 1; }
    echo ">>> downloaded $(basename "$archive")"

    # Unpack by type. -x "__MACOSX/*" drops the macOS metadata tree (._* sidecars)
    # that zips made on a Mac carry, so it can't end up under CKPT_DIR.
    EXTRACT="$STAGE/extracted"; mkdir -p "$EXTRACT"
    case "$archive" in
        *.zip)            unzip -q "$archive" -d "$EXTRACT" -x "__MACOSX/*" ;;
        *.tar.gz|*.tgz)   tar -xzf "$archive" -C "$EXTRACT" ;;
        *.tar.bz2|*.tbz2) tar -xjf "$archive" -C "$EXTRACT" ;;
        *.tar)            tar -xf  "$archive" -C "$EXTRACT" ;;
        *) echo "ERROR: unsupported archive type: $(basename "$archive")" >&2; return 1 ;;
    esac

    # Find the checkpoint-set root (the dir whose children are <name>/model_args.json)
    # regardless of whether the archive wraps them in a top-level folder.
    ma="$(find "$EXTRACT" -name model_args.json -not -path '*/__MACOSX/*' -print -quit)"
    [ -n "$ma" ] || { echo "ERROR: no model_args.json found inside the archive" >&2; return 1; }
    root="$(dirname "$(dirname "$ma")")"

    # Install the model folders under CKPT_DIR.
    mv "$root"/* "$CKPT"/
    echo "Model weights installed in $CKPT:"
    ls -1 "$CKPT"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if checkpoints_present; then checkpoints_run; exit 0; fi
    if ! ask_yes_no "Download the UDP model weights from Google Drive into $(ckpt_dir)?"; then
        skip_note "the model weights" "the eval scripts' data/checkpoints/ path" "$(ckpt_dir)"
        exit 0
    fi
    checkpoints_run
fi
