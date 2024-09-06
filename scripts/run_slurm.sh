#!/bin/bash
#
#SBATCH --job-name="toddlerbot"
#SBATCH --output=/move/u/hshi74/output/%A.out
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a5000:1 
#SBATCH --account=move
#SBATCH --partition=move

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

# Run the Python script with the parsed arguments
srun bash -c "/move/u/hshi74/.conda/envs/toddlerbot/bin/python toddlerbot/envs/train_mjx.py \
    --robot $ROBOT \
    --env $ENV \
    --eval $EVAL \
    --restore $RESTORE"