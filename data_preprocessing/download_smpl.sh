#!/usr/bin/env bash
#
# Download the SMPL v1.0.0 body model (male) used by the network and aitviewer.
# Requires a free account at https://smpl.is.tue.mpg.de.
#
# Result: data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl  (config.paths.smpl_file default)
#         data/smpl_m_lbs_10_207_0_v1.0.0.pkl        (alias used by the run scripts)
#
# Usage: ./data_preprocessing/download_smpl.sh
#        (or sourced by download_all.sh, which calls smpl_run)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

smpl_target() { echo "$DATA_DIR/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"; }
smpl_alias()  { echo "$DATA_DIR/smpl_m_lbs_10_207_0_v1.0.0.pkl"; }

smpl_present()  { [ -f "$(smpl_target)" ]; }
smpl_accounts() { smpl_present || echo smpl; }

smpl_run() {
    local TARGET ALIAS ZIP TMP MALE
    TARGET="$(smpl_target)"; ALIAS="$(smpl_alias)"
    if [ -f "$TARGET" ]; then
        echo "SMPL model already present: $TARGET"
        [ -f "$ALIAS" ] || cp "$TARGET" "$ALIAS"
        return 0
    fi

    ZIP="$DOWNLOAD_CACHE/SMPL_python_v.1.0.0.zip"
    tue_download smpl 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' "$ZIP"

    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' RETURN
    unzip -o "$ZIP" -d "$TMP" >/dev/null

    MALE="$(find "$TMP" -iname 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl' -print -quit)"
    [ -n "$MALE" ] || MALE="$(find "$TMP" -iname '*_m_lbs_10_207_0*.pkl' -print -quit)"
    [ -n "$MALE" ] || { echo "ERROR: male SMPL model not found inside $ZIP" >&2; return 1; }

    mkdir -p "$DATA_DIR"
    cp "$MALE" "$TARGET"
    cp "$TARGET" "$ALIAS"
    echo "Installed SMPL model:"
    echo "  $TARGET"
    echo "  $ALIAS (alias)"
    echo "(You can delete the cached archive $ZIP to reclaim space.)"
}

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if smpl_present; then smpl_run; exit 0; fi
    if ! ask_yes_no "Download the SMPL v1.0.0 body model (login required)?"; then
        skip_note "the SMPL body model" "paths.smpl_file (or SMPL_MODEL_PATH)" "$(smpl_target)"
        exit 0
    fi
    smpl_run
fi
