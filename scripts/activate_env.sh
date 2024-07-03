#!/bin/bash

# Check if motor type is passed as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 env_name"
  exit 1
fi

ENV_NAME=$1

# Activate the conda environment
echo "Activating the environment..."

# Determine the correct conda.sh path dynamically
CONDA_PATH=$(conda info --base)/etc/profile.d/conda.sh

if [ -f "$CONDA_PATH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_PATH"
    conda activate "$ENV_NAME"
else
    echo "conda.sh not found. Make sure conda is installed correctly."
    exit 1
fi