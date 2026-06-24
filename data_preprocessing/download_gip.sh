#!/usr/bin/env bash
#
# Download the GIP two-person UWB dataset (a single .zip) from Google Drive into
# the raw GIP-DB dir, where preprocess.process_gip() expects it:
#   <GIP_RAW_DIR>/{train,test}/person{1,2}/{train,test}.pt
#   default GIP_RAW_DIR = data/processed_data/GIP-DB
#
# After downloading, stack the two people into a single-person dataset with:
#   python modules/dataset/preprocess.py   (uncomment the process_gip(...) calls)
# which writes the merged set to paths.gip_dir (data/processed_data/Multi-UWB-Merged).
#
# Usage: ./data_preprocessing/download_gip.sh
#        (or sourced by download_all.sh, which calls gip_run)
# Env:   GIP_RAW_DIR  target dir (default data/processed_data/GIP-DB)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Just the Drive file ID (the part after /file/d/ in the share link). We build the
# uc?id= download URL from it below: passing a /file/d/<id> link straight to gdown
# makes it fetch the Drive HTML page instead of the file.
GDRIVE_ID="1FAxHqkBAqJkXpfm7iPgK089hl6S5qXEd"

gip_target_dir() { echo "${GIP_RAW_DIR:-$DATA_DIR/processed_data/GIP-DB}"; }

gip_present() {
    local d; d="$(gip_target_dir)"
    [ -f "$d/train/person1/train.pt" ] && [ -f "$d/test/person1/test.pt" ]
}

# Google Drive download — no login required.
gip_accounts() { :; }

gip_run() {
    local TARGET_DIR STAGE archive EXTRACT p1 root
    TARGET_DIR="$(gip_target_dir)"
    if gip_present; then echo "GIP data already present in $TARGET_DIR"; return 0; fi

    mkdir -p "$TARGET_DIR"
    STAGE="$(mktemp -d)"
    trap 'rm -rf "$STAGE"' RETURN

    # Download the single Drive zip (gdown keeps the original filename).
    ( cd "$STAGE" && gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" )
    archive="$(find "$STAGE" -maxdepth 1 -type f -name '*.zip' -print -quit)"
    [ -n "$archive" ] || { echo "ERROR: download produced no .zip file" >&2; return 1; }
    echo ">>> downloaded $(basename "$archive")"

    EXTRACT="$STAGE/extracted"; mkdir -p "$EXTRACT"
    # Skip the macOS metadata tree (__MACOSX/._*) that zips made on a Mac carry;
    # it mirrors the real folders with tiny sidecar files and would otherwise fool
    # the root detection below.
    unzip -q "$archive" -d "$EXTRACT" -x "__MACOSX/*"

    # Locate the GIP-DB root (the dir whose <split>/person1 subtree we expect),
    # regardless of whether the zip wraps it in a top-level folder.
    p1="$(find "$EXTRACT" -type d -name person1 -not -path '*/__MACOSX/*' -print -quit)"
    [ -n "$p1" ] || { echo "ERROR: no person1/ folder found inside the archive" >&2; return 1; }
    root="$(dirname "$(dirname "$p1")")"

    mv "$root"/* "$TARGET_DIR"/
    echo "GIP data installed in $TARGET_DIR:"
    ls -1 "$TARGET_DIR"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if gip_present; then gip_run; exit 0; fi
    if ! ask_yes_no "Download the GIP two-person UWB dataset from Google Drive into $(gip_target_dir)?"; then
        skip_note "the GIP dataset" "paths.raw_gip_dir (or GIP_RAW_DIR)" "$(gip_target_dir)"
        exit 0
    fi
    gip_run
fi
