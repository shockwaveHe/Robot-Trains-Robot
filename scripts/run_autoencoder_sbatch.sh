#!/bin/bash
#
#SBATCH --job-name=toddlerbot_runs
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=/move/u/hshi74/output/%A.out

# Path to the Python executable in the toddlerbot environment
PYTHON="/move/u/hshi74/.conda/envs/toddlerbot/bin/python"

# Optional arguments from command line (passed to sbatch)
DYNAMICS_ARGS="${1:-}"
AUTOENCODER_ARGS="${2:-}"

# Run 1: dynamics/train.py with optional args
echo "Training dynamics model..."
# Capture all output (stdout and stderr) from the Python run
output=$("$PYTHON" toddlerbot/dynamics/train.py $DYNAMICS_ARGS)
# Extract time_str from the output
time_str1=$(echo "$output" | grep "OUTPUT_DIR:" | sed 's/.*dynamics_model_\([0-9_]*\).*/\1/')
if [ -z "$time_str1" ]; then
    echo "Error: Could not extract time_str from dynamics/train.py output"
    exit 1
fi

# Run 2: autoencoder/train.py with optional args
echo "Training autoencoder from dynamics_model_${time_str1}..."
output=$("$PYTHON" toddlerbot/autoencoder/train.py data.time_str="'${time_str1}'" $AUTOENCODER_ARGS)
time_str2=$(echo "$output" | grep "OUTPUT_DIR:" | sed 's/.*dynamics_encoder_\([0-9_]*\).*/\1/')
if [ -z "$time_str2" ]; then
    echo "Error: Could not extract time_str from autoencoder/train.py output"
    exit 1
fi

# Run 3: dynamics/train.py with run_mode=eval (fixed)
echo "Evaluating the decoded dynamics model..."
"$PYTHON" toddlerbot/dynamics/train.py run_mode=eval hydra.run.dir=results/dynamics_encoder_$time_str2 $DYNAMICS_ARGS