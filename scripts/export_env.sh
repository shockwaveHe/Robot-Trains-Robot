#!/bin/bash

# Define environment name
ENV_NAME="toddlerbot"

# Check if the script is run from the root of the repo
if [ ! -f "README.md" ] || [ ! -f "setup.py" ]; then
    echo "Error: Script must be run from the root directory of the toddlerbot repository."
    exit 1
fi

conda_info=$(conda info)
if [[ $conda_info == *"miniconda"* ]]; then
    conda_name="miniconda3"
elif [[ $conda_info == *"anaconda"* ]]; then
    conda_name="anaconda3"
else
    echo "Unable to determine the Conda distribution."
fi

# Activate the environment
echo "Activating the environment..."
source ~/$conda_name/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Export the environment dependencies to requirements.txt
echo "Generating requirements.txt..."
pigar generate toddlerbot

echo "Export complete. requirements.txt is updated."
