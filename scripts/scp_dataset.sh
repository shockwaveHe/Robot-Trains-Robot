#!/bin/bash

# shellcheck disable=SC2086

# Assign positional arguments to variables
# Default values
TIME_STR=""
ROBOT="toddlerbot_OP3"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --time-str) TIME_STR="$2"; shift ;;
        --robot) ROBOT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "TIME_STR: $TIME_STR"
echo "ROBOT: $ROBOT"

REMOTE_USER="toddlerbot"
REMOTE_HOST="toddlerbot.local"
REMOTE_REPO_PATH="/home/${REMOTE_USER}/projects/toddlerbot"
LOCAL_REPO_PATH="/home/haochen/projects/toddlerbot"
RELATIVE_PATH="results/${ROBOT}_teleop_follower_pd_real_world_${TIME_STR}"
REMOTE_FOLDER_PATH="${REMOTE_REPO_PATH}/${RELATIVE_PATH}"
LOCAL_FOLDER_PATH="${LOCAL_REPO_PATH}/${RELATIVE_PATH}"

rsync -avzP --exclude '__pycache__' ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FOLDER_PATH}/{*.lz4,log_data.pkl} ${LOCAL_FOLDER_PATH}/
