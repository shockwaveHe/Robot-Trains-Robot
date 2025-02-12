#!/bin/bash

# shellcheck disable=SC2086

# This script is used to select the best checkpoint for a given task

TIME_STR=""
ROBOT="toddlerbot"
ENV="walk"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --time-str) TIME_STR="$2"; shift ;;
        --robot) ROBOT="$2"; shift ;;
        --env) ENV="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "TIME_STR: $TIME_STR"
echo "ROBOT: $ROBOT"
echo "ENV: $ENV"


RESULT_FOLDER_PATH="results/${ROBOT}_${ENV}_ppo_${TIME_STR}"
CKPT_PATH="toddlerbot/policies/checkpoints/${ROBOT}_${ENV}_policy"


cp ${RESULT_FOLDER_PATH}/best_policy ${CKPT_PATH}