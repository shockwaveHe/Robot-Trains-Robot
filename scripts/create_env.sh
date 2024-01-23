#!/bin/bash

# Define environment name and Python version
ENV_NAME="toddleroid"
PYTHON_VERSION="3.8"

# Check if the script is run from the root of the repo
if [ ! -f "README.md" ] || [ ! -f "setup.py" ]; then
    echo "Error: Script must be run from the root directory of the toddleroid repository."
    exit 1
fi

# Create the Conda environment
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y

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

# Install dependencies using pip
echo "Installing dependencies from requirements.txt..."
pip install -e .

echo "Environment setup is complete."
