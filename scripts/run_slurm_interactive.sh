#!/bin/bash

# Check if a GPU type was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <gpu_type>"
  echo "Example: $0 l40s"
  exit 1
fi

# Set the GPU type from the first argument
GPU_TYPE=$1

# Run the srun command with the specified GPU type
srun --account=move --partition=move-interactive --gres=gpu:${GPU_TYPE}:1 --time=3-00:00:00 --mem-per-cpu=4G --cpus-per-task=16 --pty bash
