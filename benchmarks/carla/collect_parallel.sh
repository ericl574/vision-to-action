#!/usr/bin/env bash
# Run N expert (or DAgger) collectors in parallel, each with its own CARLA server, inside a tmux session.
#
#   bash benchmarks/carla/collect_parallel.sh <routes-file> <out-dir> <n-workers> [extra collect.py args...]
#   e.g. bash benchmarks/carla/collect_parallel.sh benchmarks/carla/routes/train_town10.json /workspace/data/expert_town10 3
#
# Every worker gets a disjoint contiguous slice of the route file; finished routes are skipped on rerun,
# so the whole command can simply be re-run after an interruption.
set -euo pipefail
ROUTES=$1; OUT=$2; N=$3; shift 3
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "${CARLA_WORK:-/workspace}/carla_env.sh"
TOTAL=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$ROUTES")
PER=$(( (TOTAL + N - 1) / N ))
SESSION=$(basename "$OUT")
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$REPO"
mkdir -p "$OUT" "${CARLA_WORK:-/workspace}/logs"
for i in $(seq 0 $((N - 1))); do
  START=$((i * PER))
  PORT=$((2000 + i * 10)); TM=$((8000 + i * 10))
  CMD="cd $REPO && source ${CARLA_WORK:-/workspace}/carla_env.sh && python $REPO/benchmarks/carla/collect.py --routes-file $ROUTES --out $OUT \
       --route-start $START --route-count $PER --port $PORT --tm-port $TM --launch $* 2>&1 | tee -a ${CARLA_WORK:-/workspace}/logs/${SESSION}_w$i.log"
  if [ "$i" -eq 0 ]; then tmux send-keys -t "$SESSION" "$CMD" C-m
  else tmux split-window -t "$SESSION" -c "$REPO" && tmux select-layout -t "$SESSION" tiled && tmux send-keys -t "$SESSION" "$CMD" C-m
  fi
  sleep 20  # stagger server start-up (GPU memory spike while loading the map)
done
echo "started $N workers in tmux session '$SESSION' ($TOTAL routes, $PER per worker). Logs: ${CARLA_WORK:-/workspace}/logs/${SESSION}_w*.log"
