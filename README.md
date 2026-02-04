# IA-20q-REGE-2D: Image Analysis Pipeline

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)

## Overview

**IA-20q-REGE-2D** is a specialized image analysis repository designed for the **LIVR-VUB REGE-Nusa project**. It provides a complete pipeline for segmenting and analyzing 2D microscopy images of *20q-mutant* and *Wild-Type (WT)* cells.

The pipeline consists of two main stages:
1.  **Segmentation**: Using custom Cellpose models to segment cells (cytoplasm) and nuclei.
2.  **Downstream Analysis**: Quantifying marker expression per cell, classifying cell phenotypes based on marker combinations, and generating publication-ready visualizations.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/LIVR-VUB/IA-20q-REGE-2D.git
cd IA-20q-REGE-2D
```

### 2. Environment Setup

We recommend using **Python 3.11**. You can set up your environment using `venv` or `conda`.

#### Option A: Using `venv` (Standard Python)

```bash
# Create a virtual environment named 'env'
python3.11 -m venv env

# Activate the environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using `conda`

```bash
# Create a conda environment with Python 3.11
conda create -n rege_analysis python=3.11 -y

# Activate the environment
conda activate rege_analysis

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Setting Up Dependencies (Crucial Steps)

Before running any analysis, you **MUST** download the custom segmentation models and the Singularity container.

### Step 1: Download Cellpose Models

We use custom-trained Cellpose models hosted on Hugging Face (`LIVR-VUB/20q-REGE-2D`).

**Run the download script:**
```bash
python3 01_Cellpose_segment/0.download_models_20q.py
```

*   **Script Location**: `01_Cellpose_segment/0.download_models_20q.py`
*   **What it does**: Downloads `cpsam_20q_cells` and `cpsam_20qP_nuclei` into a local `downloaded_models` directory.

### Step 2: Pull the Singularity Container

The segmentation pipeline runs inside a reproducible Singularity/Apptainer container to ensure consistency.

**Run the pull script:**
```bash
bash 01_Cellpose_segment/1.pull_container.sh
```

*   **Script Location**: `01_Cellpose_segment/1.pull_container.sh`
*   **What it does**: Pulls the `cp2m_quant.sif` container from the GitHub Container Registry (ghcr.io) and saves it to a `singularity/` folder.

---

## 🔬 Running the Analysis

The analysis is split into **Segmentation** (HPC/GPU) and **Downstream Quantification** (HPC/CPU).

### Part 1: Segmentation (HPC/GPU)

Use the container downloaded in Step 2 to run Cellpose segmentation on your raw TIFF images. 
*(Note: Specific SLURM submission scripts for this step are typically located in `01_Cellpose_segment` or adapted from template scripts.)*

### Part 2: Downstream Analysis (Quantification & Classification)

Once segmentation is complete and you have `_mask.tif` or `_cp.tif` files, use the scripts in `02_Downstream_analysis`.

#### A. Quantify Cell Features

The `down_analysis_HPC.py` script measures per-cell intensity for every marker channel and nucleus area.

```bash
python3 02_Downstream_analysis/down_analysis_HPC.py \
    --input_dir /path/to/your/segmented_data \
    --output_dir /path/to/save/csv_results
```

*   **Input**: Directory containing `*_img.tif` and `*_cp.tif` files.
*   **Output**: One CSV file per image folder (e.g., `Folder_per_cell_features.csv`).

#### B. Classify Phenotypes & Plot

The `marker_classification_HPC_v2.py` script automatically detects staining combinations (e.g., "OCT4+TFAP2A"), classifies cells as positive/negative for each marker, and generates stacked bar plots.

```bash
python3 02_Downstream_analysis/marker_classification_HPC_v2.py \
    --input_dir /path/to/your/segmented_data \
    --output_dir /path/to/save/plots_and_tables \
    --overlap 0.15
```

*   **--overlap**: Fraction of nucleus area that must overlap with a marker mask to be considered "Positive" (default: 0.15).
*   **Output**: 
    *   aggregated CSVs (`cell_data.csv`)
    *   Stacked bar charts showing population composition (Aggregated and Individual).
    *   Publication-quality figures (PNG, PDF, SVG).

---

## 📂 Repository Structure

```
IA-20q-REGE-2D/
├── 00_PreProcess/              # Scripts for initial data handling (if any)
├── 01_Cellpose_segment/        # Segmentation pipeline ingredients
│   ├── 0.download_models_20q.py  # <--- CRITICAL: Downloads models
│   ├── 1.pull_container.sh       # <--- CRITICAL: Downloads container
│   ├── HOW_TO_DOWNLOAD_MODELS.md # Detailed manual instructions
│   └── ...
├── 02_Downstream_analysis/     # Post-segmentation quantification
│   ├── down_analysis_HPC.py           # Measurements (Intensity, Area)
│   ├── marker_classification_HPC_v2.py # Phenotype classification & Plotting
│   └── 20qdown.sbatch                # SLURM job submission example
└── requirements.txt            # Python dependencies
```

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 LIVR-VUB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
