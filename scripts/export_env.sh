#!/bin/bash

# Define environment name
ENV_NAME="toddleroid"

# Check if the script is run from the root of the repo
if [ ! -f "README.md" ] || [ ! -f "setup.py" ]; then
    echo "Error: Script must be run from the root directory of the toddleroid repository."
    exit 1
fi

# Activate the environment
echo "Activating the environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Export the environment dependencies to requirements.txt
echo "Generating requirements.txt..."
pigar generate toddleroid

echo "Export complete. requirements.txt is updated."
