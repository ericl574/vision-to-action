#!/usr/bin/env bash
# Pull curated outputs (results, best checkpoints, training curves, rendered videos) from the GPU box to the repo.
#   bash benchmarks/carla/sync_from_box.sh <ssh-port> <user@host>
set -euo pipefail
PORT=$1; HOST=$2
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -p $PORT"
mkdir -p "$HERE/results" "$HERE/runs" "$HERE/demo"
rsync -az -e "$SSH" "$HOST:/workspace/vision-to-action/benchmarks/carla/results/" "$HERE/results/" || true
rsync -az -e "$SSH" --include '*/' --include 'best.pt' --include 'history.csv' --include 'data_stats.json' --include 'split.json' \
  --exclude '*' --prune-empty-dirs "$HOST:/workspace/runs/" "$HERE/runs/" || true
rsync -az -e "$SSH" --include '*.mp4' --exclude '*' "$HOST:/workspace/runs/" "$HERE/demo/" || true
du -sh "$HERE/results" "$HERE/runs" "$HERE/demo"
