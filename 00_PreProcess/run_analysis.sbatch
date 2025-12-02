#!/bin/bash
#SBATCH --job-name=20q
#SBATCH --partition=pascal_gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out


module purge
module load Mamba
source $EBROOTMAMBA/etc/profile.d/conda.sh

set -euo pipefail

# (Optional) load modules
# module load Apptainer

# Project root = directory where you ran `sbatch` from
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJ_ROOT"

# Paths inside your project
SIF="${PROJ_ROOT}/singularity/cp2M_quant.sif"
CFG="${PROJ_ROOT}/config_final.yaml"
PY="${PROJ_ROOT}/analysis_v5.py"

# Quick checks
[[ -f "$SIF" ]] || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[[ -f "$CFG" ]] || { echo "ERROR: YAML not found: $CFG"; exit 1; }
[[ -f "$PY"  ]] || { echo "ERROR: Pipeline script not found: $PY"; exit 1; }

echo "-----------------------------------------------------"
echo " Running EPCAM–OCT4 pipeline via Apptainer (Slurm job)"
echo " Project    : $PROJ_ROOT"
echo " SIF        : $SIF"
echo " Config     : $CFG"
echo "-----------------------------------------------------"

# Run container with GPU and bind the whole project at /work
apptainer run --nv \
  -B "$PROJ_ROOT":/work \
  "$SIF" \
  python /work/$(basename "$PY") /work/$(basename "$CFG")
