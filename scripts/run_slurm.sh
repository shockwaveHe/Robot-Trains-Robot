#!/bin/bash
#
#SBATCH --job-name=toddlerbot
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --gres=gpu:a5000:1 
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=/move/u/hshi74/output/%A.out

# Default values for the arguments
ROBOT="toddlerbot"
ENV="walk"
EVAL=""
RESTORE=""

# Parse command-line arguments in the bash script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --robot) ROBOT="$2"; shift ;;
        --env) ENV="$2"; shift ;;
        --eval) EVAL="$2"; shift ;;
        --restore) RESTORE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# Construct the command dynamically based on provided arguments
COMMAND="/move/u/hshi74/.conda/envs/toddlerbot/bin/python toddlerbot/envs/train_mjx.py --robot $ROBOT --env $ENV"

# Only add --eval and --restore if they are non-empty
if [ -n "$EVAL" ]; then
    COMMAND="$COMMAND --eval $EVAL"
fi

if [ -n "$RESTORE" ]; then
    COMMAND="$COMMAND --restore $RESTORE"
fi

# Run the Python script with the constructed command
srun bash -c "$COMMAND"