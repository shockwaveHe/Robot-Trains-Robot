#!/bin/bash

# Define environment name and Python version
ENV_NAME="toddlerbot"
PYTHON_VERSION="3.8"

# Check if the script is run from the root of the repo
if [ ! -f "README.md" ] || [ ! -f "setup.py" ]; then
    echo "Error: Script must be run from the root directory of the toddlerbot repository."
    exit 1
fi

# Create the Conda environment
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate the environment
bash scripts/activate_env.sh "$ENV_NAME"

# Install dependencies using pip
echo "Installing dependencies from requirements.txt..."
pip install -e .

echo "Environment setup is complete."
