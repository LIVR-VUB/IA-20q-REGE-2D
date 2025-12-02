#!/bin/bash
#SBATCH --job-name=20qdown
#SBATCH --partition=zen5_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --output=20qdown_%j.log

module purge
module load Mamba
module load tqdm/4.66.5-GCCcore-13.3.0
source $EBROOTMAMBA/etc/profile.d/conda.sh
conda activate sam

INPUT_DIR=/scratch/brussel/vo/000/bvo00026/vsc11013/Projects/REGE-Nusa/Output_Final
OUTPUT_DIR=/scratch/brussel/vo/000/bvo00026/vsc11013/Projects/REGE-Nusa/Output_down_final

mkdir -p "$OUTPUT_DIR"

export N_WORKERS=${SLURM_CPUS_PER_TASK:-1}

# Avoid oversubscription in BLAS/OpenMP
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[INFO] Starting per-folder quantification using $N_WORKERS workers..."

python down_analysis_parallel_HPC.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_workers $N_WORKERS

if [ $(find "$OUTPUT_DIR" -name "*_per_cell_features.csv" | wc -l) -eq 0 ]; then
    echo "[ERROR] down_analysis_HPC.py did not produce any *_per_cell_features.csv files; aborting."
    exit 1
fi

echo "[INFO] Quantification complete. Starting batch analysis..."

python down_analysis_cellstate_parallel_HPC.py \
    --root_dir "$OUTPUT_DIR" \
    --n_workers $N_WORKERS

echo "[INFO] Job completed successfully."
