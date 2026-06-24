#!/usr/bin/env bash
#
# Run every UDP evaluation in sequence.
# Usage: ./eval_scripts/eval_all.sh [gpu_id]
#
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"

for s in eval_dip eval_dancedb eval_tc eval_uip eval_uip_ft eval_gip eval_gip_ft; do
    echo "================================================================"
    echo "Running $s"
    echo "================================================================"
    "$HERE/$s.sh" "$GPU"
done
