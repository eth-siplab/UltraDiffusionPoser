#!/usr/bin/env bash
#
# Download the TotalCapture dataset (two parts):
#   1. Vicon ground-truth global skeleton (S1..S5) from cvssp.org -- the
#      "position and orientation" csv archives (sX_vicon_pos_ori.tar.gz), which
#      contain the gt_skel_gbl_pos.txt that preprocessing reads.
#        -> data/TotalCapture/S{1..5}/<motion>/gt_skel_gbl_{pos,ori}.txt
#      Requires a cvssp TotalCapture account (its own, non-MPG login).
#   2. DIP-recomputed SMPL ground truth (TotalCapture_Real_60FPS) from
#      download.is.tue.mpg.de  -> data/TotalCapture_Real_60FPS/*.pkl
#      Hosted under the SAME dip account as the DIP dataset (reused, not re-asked).
#
# Usage: ./data_preprocessing/download_tc.sh
#        (or sourced by download_all.sh, which calls tc_run)
#
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TC_SUBJECTS=(S1 S2 S3 S4 S5)
tc_official_dir() { echo "${TC_OFFICIAL_DIR:-$DATA_DIR/TotalCapture}"; }            # paths.raw_totalcapture_official_dir
tc_gt_dir()       { echo "${TC_GT_DIR:-$DATA_DIR/TotalCapture_Real_60FPS}"; }       # paths.raw_totalcapture_dip_dir

tc_official_present() {
    local d s; d="$(tc_official_dir)"
    for s in "${TC_SUBJECTS[@]}"; do
        [ -n "$(find "$d/$s" -name 'gt_skel_gbl_pos.txt' -print -quit 2>/dev/null)" ] || return 1
    done
    return 0
}
tc_gt_present() {
    local d; d="$(tc_gt_dir)"
    [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]
}

tc_present()  { tc_official_present && tc_gt_present; }
tc_accounts() {
    tc_official_present || echo cvssp
    tc_gt_present       || echo dip
}

# Part 1: Vicon ground-truth global skeleton S1..S5 (cvssp.org).
tc_official_run() {
    local TC_OFFICIAL s ls tgz tmp gt subjectdir
    TC_OFFICIAL="$(tc_official_dir)"
    if tc_official_present; then
        echo "TotalCapture ground-truth skeleton (S1..S5) already present at $TC_OFFICIAL"
        return 0
    fi
    cvssp_login
    for s in "${TC_SUBJECTS[@]}"; do
        ls="${s,,}"                                   # S1 -> s1 (cvssp uses lowercase here)
        tgz="$DOWNLOAD_CACHE/${ls}_vicon_pos_ori.tar.gz"
        echo ">>> ${ls}_vicon_pos_ori.tar.gz"
        cvssp_download "https://cvssp.org/data/totalcapture/data/dataset/vicon/${ls}_vicon_pos_ori.tar.gz" "$tgz"

        # Extract to a temp dir, find a gt_skel_gbl_pos.txt, and move its <subject>
        # dir (two levels up: <subject>/<motion>/gt_skel_gbl_pos.txt) into place.
        tmp="$(mktemp -d)"
        tar -xzf "$tgz" -C "$tmp"
        gt="$(find "$tmp" -name 'gt_skel_gbl_pos.txt' -print -quit)"
        [ -n "$gt" ] || { echo "ERROR: gt_skel_gbl_pos.txt not found in $tgz" >&2; rm -rf "$tmp"; return 1; }
        subjectdir="$(dirname "$(dirname "$gt")")"
        mkdir -p "$TC_OFFICIAL"
        rm -rf "$TC_OFFICIAL/$s"
        mv "$subjectdir" "$TC_OFFICIAL/$s"
        rm -rf "$tmp"
    done
    echo "TotalCapture ground-truth skeleton extracted to $TC_OFFICIAL ($(ls "$TC_OFFICIAL" | tr '\n' ' '))"
}

# Part 2: DIP-recomputed SMPL GT (is.tue.mpg.de, dip account).
tc_gt_run() {
    local TC_GT ZIP tmp src
    TC_GT="$(tc_gt_dir)"
    if tc_gt_present; then
        echo "TotalCapture ground truth already present at $TC_GT"
        return 0
    fi
    ZIP="$DOWNLOAD_CACHE/TotalCapture_Real_60FPS.zip"
    tue_download dip 'https://download.is.tue.mpg.de/download.php?domain=dip&resume=1&sfile=TotalCapture_Real_60FPS.zip' "$ZIP"

    # Extract to temp, then move the *.pkl files into place (handles nested or flat zip).
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    unzip -oq "$ZIP" -d "$tmp"
    src="$(dirname "$(find "$tmp" -name '*.pkl' -print -quit)")"
    [ -n "$src" ] || { echo "ERROR: no .pkl files found inside $ZIP" >&2; return 1; }
    mkdir -p "$TC_GT"
    mv -f "$src"/*.pkl "$TC_GT"/
    echo "TotalCapture ground truth extracted to $TC_GT ($(ls "$TC_GT" | wc -l) files)"
    echo "(You can delete the cached archive $ZIP to reclaim space.)"
}

# Both parts (each no-ops if already present) — used by download_all.sh.
tc_run() { tc_official_run; tc_gt_run; }

# Run standalone (skipped when sourced by download_all.sh).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if tc_official_present; then
        tc_official_run
    elif ask_yes_no "Download the TotalCapture Vicon ground-truth skeleton (S1..S5) from cvssp.org?"; then
        tc_official_run
    else
        skip_note "the TotalCapture ground-truth skeleton (S1..S5)" "paths.raw_totalcapture_official_dir" "$(tc_official_dir)"
    fi

    if tc_gt_present; then
        tc_gt_run
    elif ask_yes_no "Download the TotalCapture ground truth (TotalCapture_Real_60FPS, dip account)?"; then
        tc_gt_run
    else
        skip_note "the TotalCapture ground truth" "paths.raw_totalcapture_dip_dir" "$(tc_gt_dir)"
    fi
fi
