#!/usr/bin/env bash
#
# Download & extract the DIP-IMU dataset. Requires a free account at
# https://dip.is.tue.mpg.de. The archive is large (~2.6 GB) and nested
# (DIPIMUandOthers.zip -> DIP_IMU_and_Others/DIP_IMU.zip -> DIP_IMU/s_*/*.pkl).
#
# Result: data/DIP_IMU/s_01 .. s_10/*.pkl   (config.paths.raw_dipimu_dir)
#
# Usage: ./data_preprocessing/download_dip.sh
#        (or sourced by download_all.sh, which calls dip_run)
# Env:   DIP_ZIP   reuse an already-downloaded DIPIMUandOthers.zip (skips download)
#        TMPDIR    where the ~2 GB inner zip is unpacked (default /tmp)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

dip_raw_dir() { echo "${DIPIMU_RAW_DIR:-$DATA_DIR/DIP_IMU}"; }
dip_outer()   { echo "${DIP_ZIP:-$DOWNLOAD_CACHE/DIPIMUandOthers.zip}"; }

dip_present() {
    local d; d="$(dip_raw_dir)"
    [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]
}

# Needs the dip login only if the outer archive isn't already on disk.
dip_accounts() {
    dip_present && return 0
    [ -f "$(dip_outer)" ] || echo dip
}

dip_run() {
    local RAW_DIP OUTER WORK INNER
    RAW_DIP="$(dip_raw_dir)"
    if dip_present; then echo "DIP-IMU already present at $RAW_DIP"; return 0; fi

    # 1) Get the outer archive (reuse DIP_ZIP / cached copy if present).
    OUTER="$(dip_outer)"
    if [ -f "$OUTER" ]; then
        echo "Using existing archive: $OUTER"
    else
        tue_download dip 'https://download.is.tue.mpg.de/download.php?domain=dip&resume=1&sfile=DIPIMUandOthers.zip' "$OUTER"
    fi

    # 2) Pull the inner DIP_IMU.zip out of the outer archive into $TMPDIR (~2 GB).
    WORK="$(mktemp -d)"
    trap 'rm -rf "$WORK"' RETURN
    echo ">>> Extracting inner DIP_IMU.zip (temporary, ~2 GB, in $WORK) ..."
    unzip -o "$OUTER" "DIP_IMU_and_Others/DIP_IMU.zip" -d "$WORK" >/dev/null
    INNER="$WORK/DIP_IMU_and_Others/DIP_IMU.zip"
    [ -f "$INNER" ] || { echo "ERROR: inner DIP_IMU.zip not found in $OUTER" >&2; return 1; }

    # 3) Unpack the inner zip into data/ -> yields data/DIP_IMU/.
    echo ">>> Unzipping DIP_IMU into $DATA_DIR/ ..."
    mkdir -p "$DATA_DIR"
    unzip -o "$INNER" -d "$DATA_DIR" >/dev/null

    dip_present || { echo "ERROR: extraction did not produce $RAW_DIP" >&2; return 1; }

    echo "DIP-IMU extracted to $RAW_DIP"
    echo "Subjects: $(ls "$RAW_DIP" | tr '\n' ' ')"
    echo "(You can delete the cached archive $OUTER to reclaim space.)"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if dip_present; then dip_run; exit 0; fi
    if ! ask_yes_no "Download & extract the DIP-IMU dataset (~2.6 GB, login required)?"; then
        skip_note "the DIP-IMU dataset" "paths.raw_dipimu_dir" "$(dip_raw_dir)"
        exit 0
    fi
    dip_run
fi
