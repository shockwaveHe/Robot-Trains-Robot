#!/bin/bash

# shellcheck disable=SC2086

# Assign positional arguments to variables
# Default values
TIME_STR=""
ROBOT="toddlerbot_gripper"
TASK="pick"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --time-str) TIME_STR="$2"; shift ;;
        --robot) ROBOT="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "TIME_STR: $TIME_STR"
echo "ROBOT: $ROBOT"
echo "TASK: $TASK"

REMOTE_USER="hshi74"
REMOTE_HOST="scdt.stanford.edu"
REMOTE_REPO_PATH="/move/u/${REMOTE_USER}/projects/toddlerbot"
# REMOTE_USER="haochen"
# REMOTE_HOST="192.168.0.170"
# REMOTE_REPO_PATH="/home/${REMOTE_USER}/projects/toddlerbot"
LOCAL_REPO_PATH="/home/$(whoami)/projects/toddlerbot"
RELATIVE_PATH="results/${ROBOT}_${TASK}_dp_${TIME_STR}"
REMOTE_FOLDER_PATH="${REMOTE_REPO_PATH}/${RELATIVE_PATH}"
LOCAL_FOLDER_PATH="${LOCAL_REPO_PATH}/${RELATIVE_PATH}"

rsync -avzP --exclude '__pycache__' ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FOLDER_PATH}/{*best_ckpt.pth,*.png,*.mp4} ${LOCAL_FOLDER_PATH}/
