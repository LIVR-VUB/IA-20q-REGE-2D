#!/bin/bash
#SBATCH --job-name=20qdowncellstate
#SBATCH --partition=zen5_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=cellstate_%j.log

# ------------------------------
# Load environment
# ------------------------------
module purge
module load Mamba
source $EBROOTMAMBA/etc/profile.d/conda.sh
conda activate sam   # adjust if your env is named differently

# ------------------------------
# Input / Output directories
# ------------------------------
ROOT_DIR="/scratch/brussel/vo/000/bvo00026/vsc11013/Projects/REGE-Nusa/Output_down_final"

# Number of worker processes = number of CPUs
export N_WORKERS=$SLURM_CPUS_PER_TASK

# Prevent oversubscription by BLAS/OpenMP
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[INFO] Starting cell state analysis with $N_WORKERS workers..."
echo "[INFO] Root directory: $ROOT_DIR"
echo "---------------------------------------------"

# ------------------------------
# Run your Python script
# ------------------------------
python down_analysis_cellstate_parallel_HPC.py \
    --root_dir "$ROOT_DIR" \
    --n_workers $N_WORKERS

echo "---------------------------------------------"
echo "[INFO] Job completed successfully."
