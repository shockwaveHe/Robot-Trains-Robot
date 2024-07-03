#!/bin/bash

# Check if motor type is passed as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 motor_type"
  exit 1
fi

MOTOR_TYPE=$1
EXP_FOLDER_PATH=""

# Determine the corresponding experiment folder path based on the motor type
case "$MOTOR_TYPE" in
  "XC430")
    EXP_FOLDER_PATH="results/20240702_173732_sysID_XC430_joint_0"
    ;;
  "XC330")
    EXP_FOLDER_PATH="results/20240703_161744_sysID_XC330_joint_0"
    ;;
  "2XL430")
    EXP_FOLDER_PATH="results/20240703_171234_sysID_2XL430_joint_0"
    ;;
  "2XC430")
    EXP_FOLDER_PATH="results/20240703_201739_sysID_2XC430_joint_0"
    ;;
  *)
    echo "Unknown motor type: $MOTOR_TYPE"
    exit 1
    ;;
esac

# Activate the environment
bash scripts/activate_env.sh "toddlerbot"

# Run the Python program with the specified motor type and experiment folder path
python toddlerbot/tools/sysID/optimize_parameters.py \
  --robot-name "sysID_$MOTOR_TYPE" \
  --sim "mujoco" \
  --joint-names "joint_0" \
  --n-iters "500" \
  --exp-folder-path "$EXP_FOLDER_PATH"
