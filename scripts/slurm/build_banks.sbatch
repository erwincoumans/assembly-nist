#!/bin/bash
#SBATCH -J assemblenist-banks
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --array=0-19
#SBATCH -o logs/%x-%A_%a.log
# Reset banks, one array task per assembly task (about 10-60 min each on 16 cores).
#   sbatch scripts/slurm/build_banks.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
TASKS=(R4 R8 R12 R16 P4 P8 P12 P16 Gs Gm Gl N4 N8 N12 N16 BNC DSUB RJ45 USB WP)
python scripts/build_banks.py --tasks "${TASKS[$SLURM_ARRAY_TASK_ID]}" --workers "$SLURM_CPUS_PER_TASK"
