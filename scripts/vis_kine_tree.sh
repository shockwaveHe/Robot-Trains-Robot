#!/bin/bash
REPO_NAME="toddlerbot"
ROBOT_NAME="robotis_op3"
URDF_PATH=$REPO_NAME/descriptions/$ROBOT_NAME/$ROBOT_NAME.urdf

python $REPO_NAME/utils/vis_kine_tree.py \
    --path $URDF_PATH \
    -o $REPO_NAME/descriptions/${ROBOT_NAME}_kine_tree.png
