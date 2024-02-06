#!/bin/bash
URDF_PATH=toddleroid/robot_descriptions/robotis_op3/robotis_op3.urdf
# YAML_PATH=toddleroid/robot_descriptions/kine_tree_template.yml

python toddleroid/utils/vis_kine_tree.py \
    --path $URDF_PATH \
    -o toddleroid/robot_descriptions/kine_tree_template.png

# python toddleroid/utils/vis_kine_tree.py \
#     --path $YAML_PATH \
#     -o toddleroid/robot_descriptions/kine_tree_template.png