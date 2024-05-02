#!/bin/bash

# Variables
REMOTE_USER="haochen"
REMOTE_HOST="10.5.66.54"
REMOTE_REPO_PATH="/home/haochen/projects/toddlerbot"
LOCAL_REPO_PATH="/Users/haochen/Projects/toddlerbot"
RELATIVE_PATH="toddlerbot/sim/humanoid_gym/logs/walk_toddlerbot_legs_isaac"
REMOTE_FOLDER_PATH="${REMOTE_REPO_PATH}/${RELATIVE_PATH}"
LOCAL_FOLDER_PATH="${LOCAL_REPO_PATH}/${RELATIVE_PATH}"

# SSH to remote host and list directories
ssh -n ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_FOLDER_PATH} && ls -d */" | while read subdir; do
    # Check if "exported" is in the directory name
    if [[ "$subdir" == *"exported"* ]]; then
        echo "Syncing directory ${subdir} because it is marked as exported"
        rsync -avz --ignore-existing ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FOLDER_PATH}/${subdir}/ ${LOCAL_FOLDER_PATH}/${subdir}/
        continue
    fi
    
    # Use SSH to find the highest model number in the directory
    max_model=$(ssh -n ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_FOLDER_PATH}/${subdir} && ls model_*.pt 2>/dev/null | sed 's/model_\\([0-9]*\\).pt/\\1/' | sort -n | tail -1")

    # Check if the maximum model number is greater than or equal to 1000
    if [[ "$max_model" -ge 1000 ]]; then
        echo "Syncing directory ${subdir} with max model ${max_model}"
        mkdir -p ${LOCAL_FOLDER_PATH}/${subdir}
        rsync -avz --ignore-existing ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FOLDER_PATH}/${subdir} ${LOCAL_FOLDER_PATH}/${subdir}
    else
        echo "Skipping directory ${subdir} with max model ${max_model}"
    fi
done