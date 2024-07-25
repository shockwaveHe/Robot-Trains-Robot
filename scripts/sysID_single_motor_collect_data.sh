#!/bin/bash
MOTOR_TYPE="XC330"
RESULT_DIR="results"
N_TRIALS=15
N_LOADS=4
EXP_FOLDER_PATH=""
# EXP_FOLDER_PATH="results/20240725_154005_sysID_XC330_joint_0"

# Collect initial data
python toddlerbot/tools/sysID/collect_data.py --robot-name sysID_$MOTOR_TYPE --joint-names joint_0 --n-trials $N_TRIALS --n-loads $N_LOADS --exp-folder-path "$EXP_FOLDER_PATH"

# Overwrite EXP_FOLDER_PATH if it is empty
if [ -z "$EXP_FOLDER_PATH" ]; then
    # Find the most recent exp folder
    EXP_FOLDER_PATH=$(find "$RESULT_DIR" -type d -print0 | xargs -0 ls -td | head -1)
    printf "The experiment folder is: %s\n" "$EXP_FOLDER_PATH"
fi

# Loop to prompt user to remove loads one by one
for ((i = N_LOADS-1; i >= 0; i--)); do
    printf "Have you removed 1 load? (y/n) > "
    read -r is_ready
    if [ "$is_ready" == "y" ]; then
        python toddlerbot/tools/sysID/collect_data.py --robot-name sysID_$MOTOR_TYPE --joint-names joint_0 --n-trials $N_TRIALS --n-loads "$i" --exp-folder-path "$EXP_FOLDER_PATH"
    else
        echo "Please remove the load and try again."
        exit 1
    fi
done
