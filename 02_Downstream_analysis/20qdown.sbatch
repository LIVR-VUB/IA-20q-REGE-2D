#!/bin/bash
#SBATCH --job-name=20qdown
#SBATCH --partition=zen5_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

#python down_analysis_HPC.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --n_workers $N_WORKERS

python marker_classification_HPC_v2.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --n_workers $N_WORKERS
