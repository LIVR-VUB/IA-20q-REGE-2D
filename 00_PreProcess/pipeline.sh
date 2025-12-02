#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (the folder containing this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_ROOT="${SCRIPT_DIR}"

# Paths inside your project
SIF="${PROJ_ROOT}/singularity/cp2M_quant.sif"
CFG="${PROJ_ROOT}/configF3PDb.yaml"
PY="${PROJ_ROOT}/analysis_v5.py"   # put the pipeline script here

# Quick checks (nice errors instead of silent failures)
[[ -f "$SIF" ]] || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[[ -f "$CFG" ]] || { echo "ERROR: YAML not found: $CFG"; exit 1; }
[[ -f "$PY"  ]] || { echo "ERROR: Pipeline script not found: $PY"; exit 1; }

echo "-----------------------------------------------------"
echo " Running EPCAM–OCT4 pipeline via Apptainer"
echo " Project    : $PROJ_ROOT"
echo " SIF        : $SIF"
echo " Config     : $CFG"
echo "-----------------------------------------------------"

# Run container with GPU and bind the whole project at /work
# NOTE: Your container's %runscript just activates the venv and execs "$@",
# so we pass the python command explicitly here.
apptainer run --nv \
  -B "$PROJ_ROOT":/work \
  "$SIF" \
  python /work/$(basename "$PY") /work/$(basename "$CFG")
