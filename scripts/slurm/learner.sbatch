#!/bin/bash
#SBATCH -J assemblenist-learner
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.log
# GPU learner of the reported setting (8,192 environments, 10K iterations); the environments are simulated by
# NCONN worker processes started with scripts/slurm/workers.sbatch (e.g. 30 jobs x 32 cores = 960). Resubmit to resume.
#   sbatch --export=ALL,RUN_DIR=runs/e8192,NCONN=960 scripts/slurm/learner.sbatch
# Add your cluster's partition / account options (#SBATCH -p ..., -A ...) as needed.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
python -m assembly_policy.train --run-dir "${RUN_DIR:?}" --remote-workers "${NCONN:?}" --max-hours 23.5 ${TRAIN_ARGS:-}
