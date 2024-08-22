#!/bin/bash

# Assign positional arguments to variables
TIME_STR="$1"
ROBOT="${2:-toddlerbot}"  # Use the second argument or a default value
ENV="${3:-walk}"      # Use the third argument or a default value

REMOTE_USER="hshi74"
REMOTE_HOST="scdt.stanford.edu"
REMOTE_REPO_PATH="/move/u/${REMOTE_USER}/projects/toddlerbot"
LOCAL_REPO_PATH="/home/haochen/projects/toddlerbot"
RELATIVE_PATH="results/${ROBOT}_${ENV}_ppo_${TIME_STR}"
REMOTE_FOLDER_PATH="${REMOTE_REPO_PATH}/${RELATIVE_PATH}"
LOCAL_FOLDER_PATH="${LOCAL_REPO_PATH}/${RELATIVE_PATH}"

rsync -avz ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FOLDER_PATH}/policy ${LOCAL_FOLDER_PATH}/
