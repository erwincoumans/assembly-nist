#!/usr/bin/env bash
# Start simulation worker processes on this machine for the learner of RUN_DIR
# (python -m assembly_policy.train --run-dir RUN_DIR --remote-workers N). Each process simulates its slice of
# environments single-threaded; a crashed worker is restarted, and workers exit once no learner has been
# reachable for WAIT seconds (default 3600). Start enough processes in total for --remote-workers.
#   bash scripts/workers.sh runs/e8192 32          # 32 processes on this node
set -uo pipefail
RUN_DIR=$(realpath "$1"); NPROC=${2:-$(nproc)}; WAIT=${WAIT:-3600}; PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
serve() {
  for attempt in $(seq 1 20); do
    OMP_NUM_THREADS=1 $PY -m assembly_policy.rl.remote --run-dir "$RUN_DIR" --nthread 1 --poll 20 --wait "$WAIT" && return
    echo "worker exited rc=$? (attempt $attempt)"; sleep 30
  done
}
for i in $(seq 1 "$NPROC"); do
  serve 2>&1 | sed -u "s/^/[w$i] /" &
  sleep 0.2
done
wait
