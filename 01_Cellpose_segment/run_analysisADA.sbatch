#!/bin/bash
#SBATCH --job-name=20q
#SBATCH --partition=ada_gpu          # <-- use the Anansi GPU partition name at your site
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --gres=shard:3                  # 1 shard = 25% GPU (use 1..4)

module purge
module load Mamba
source $EBROOTMAMBA/etc/profile.d/conda.sh

set -euo pipefail

PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ_ROOT"

SIF="${PROJ_ROOT}/singularity/cp2M_quant.sif"
CFG="${PROJ_ROOT}/config_v6.yaml"
PY="${PROJ_ROOT}/analysis_v7.py"

[[ -f "$SIF" ]] || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[[ -f "$CFG" ]] || { echo "ERROR: YAML not found: $CFG"; exit 1; }
[[ -f "$PY"  ]] || { echo "ERROR: Pipeline script not found: $PY"; exit 1; }

echo "-----------------------------------------------------"
echo " Running EPCAM–OCT4 pipeline via Apptainer (Slurm job)"
echo " Project    : $PROJ_ROOT"
echo " SIF        : $SIF"
echo " Config     : $CFG"
echo " GPU shards : ${SLURM_JOB_GRES:-unknown}"
echo "-----------------------------------------------------"

apptainer run --nv \
  -B "$PROJ_ROOT":/work \
  "$SIF" \
  python /work/$(basename "$PY") /work/$(basename "$CFG")
