#!/bin/bash
# =============================================================================
# Script to pull cp2M_quant.sif from GHCR
# Target Directory: ./singularity
# =============================================================================

set -e

# Configuration
CONTAINER_NAME="cp2m_quant"       # Lowercase for GHCR
FILE_NAME="cp2M_quant.sif"        # Local filename with uppercase M preferred
GITHUB_ORG="livr-vub"
OUTPUT_DIR="singularity"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=========================================="
echo "  Pulling Container: $FILE_NAME"
echo "=========================================="
echo ""

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${CYAN}Creating directory: $OUTPUT_DIR${NC}"
    mkdir -p "$OUTPUT_DIR"
fi

# Pull command
REGISTRY_URL="oras://ghcr.io/${GITHUB_ORG}/${CONTAINER_NAME}:latest"
OUTPUT_PATH="${OUTPUT_DIR}/${FILE_NAME}"

echo -e "${CYAN}Downloading from: $REGISTRY_URL${NC}"
echo -e "${CYAN}Saving to:        $OUTPUT_PATH${NC}"
echo ""

if command -v singularity &> /dev/null; then
    singularity pull --force "$OUTPUT_PATH" "$REGISTRY_URL"
elif command -v apptainer &> /dev/null; then
    apptainer pull --force "$OUTPUT_PATH" "$REGISTRY_URL"
else
    echo "Error: Neither singularity nor apptainer found."
    exit 1
fi

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Successfully downloaded $FILE_NAME${NC}"
    echo "Location: $(pwd)/$OUTPUT_PATH"
else
    echo "Failed to download container."
    exit 1
fi
