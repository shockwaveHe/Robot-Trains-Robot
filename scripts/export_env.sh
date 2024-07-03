#!/bin/bash

# Define environment name
ENV_NAME="toddlerbot"

# Check if the script is run from the root of the repo
if [ ! -f "README.md" ] || [ ! -f "setup.py" ]; then
    echo "Error: Script must be run from the root directory of the toddlerbot repository."
    exit 1
fi

bash scripts/activate_env.sh "$ENV_NAME"

# Export the environment dependencies to requirements.txt
echo "Generating requirements.txt..."
pigar generate toddlerbot

echo "Export complete. requirements.txt is updated."
