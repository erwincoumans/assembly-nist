#!/bin/bash
#SBATCH -J assemblenist-eval
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.log
# Paper-protocol evaluation of every kept checkpoint of RUN_DIR that has no result yet (DET=1: mean actions).
#   sbatch --export=ALL,RUN_DIR=runs/e8192 scripts/slurm/evaluate.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
OUT=${DET:+eval_det}; OUT=${OUT:-eval}
for ck in "${RUN_DIR:?}"/model_*.pt; do
  out="$RUN_DIR/$OUT/$(basename "$ck" .pt).json"
  [ -f "$out" ] || python -m assembly_policy.evaluate --policy "$ck" --out "$out" --nthread "$SLURM_CPUS_PER_TASK" ${DET:+--deterministic}
done
