#!/bin/bash
#SBATCH -J assemblenist-workers
#SBATCH -c 32
#SBATCH --mem=72G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.log
# CPU simulation workers (one process per core) for the learner of RUN_DIR; submit as many jobs as needed for
# the learner's NCONN, in any order (workers wait for the learner and reconnect after it restarts).
#   for i in $(seq 30); do sbatch --export=ALL,RUN_DIR=runs/e8192 scripts/slurm/workers.sbatch; done
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
bash scripts/workers.sh "${RUN_DIR:?}" "$SLURM_CPUS_PER_TASK"
