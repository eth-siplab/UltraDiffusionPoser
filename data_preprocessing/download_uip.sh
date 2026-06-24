#!/usr/bin/env bash
#
# Download the UIP-DB (SIGGRAPH UWB-IMU) processed dataset (train.pt + test.pt)
# from Google Drive.
#
# Result: <UWBIMU_DIR>/train.pt and <UWBIMU_DIR>/test.pt
#         default UWBIMU_DIR = data/processed_data/UWB_IMU/SIGGRAPH_dataset
#
# Usage: ./data_preprocessing/download_uip.sh
#        (or sourced by download_all.sh, which calls uip_run)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GDRIVE_FOLDER="https://drive.google.com/drive/u/1/folders/1JTa6EbfacEbwkWja1SqTbuD63SXCx1tj"

uip_target_dir() { echo "${UWBIMU_DIR:-$DATA_DIR/processed_data/UWB_IMU/SIGGRAPH_dataset}"; }

uip_present() {
    local d; d="$(uip_target_dir)"
    [ -f "$d/train.pt" ] && [ -f "$d/test.pt" ]
}

# Google Drive download — no login required.
uip_accounts() { :; }

uip_run() {
    local TARGET_DIR STAGE moved=0 pt src
    TARGET_DIR="$(uip_target_dir)"
    if uip_present; then echo "UIP data already present in $TARGET_DIR"; return 0; fi

    mkdir -p "$TARGET_DIR"
    STAGE="$(mktemp -d)"
    trap 'rm -rf "$STAGE"' RETURN

    # gdown nests the drive folder under its own name; stage then move the .pt files.
    gdown --folder --remaining-ok -O "$STAGE" "$GDRIVE_FOLDER"

    for pt in train.pt test.pt; do
        src="$(find "$STAGE" -name "$pt" -print -quit)"
        if [ -n "$src" ]; then
            mv -f "$src" "$TARGET_DIR/$pt"
            echo "  -> $TARGET_DIR/$pt"
            moved=$((moved + 1))
        else
            echo "WARNING: $pt not found in the downloaded folder" >&2
        fi
    done

    [ "$moved" -gt 0 ] || { echo "ERROR: no UIP data files were downloaded." >&2; return 1; }
    echo "UIP data installed in $TARGET_DIR"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if uip_present; then uip_run; exit 0; fi
    if ! ask_yes_no "Download the UIP (SIGGRAPH UWB-IMU) processed data from Google Drive?"; then
        skip_note "the UIP processed data" "paths.uwbimu_dir (or UWBIMU_DIR)" "$(uip_target_dir)"
        exit 0
    fi
    uip_run
fi
