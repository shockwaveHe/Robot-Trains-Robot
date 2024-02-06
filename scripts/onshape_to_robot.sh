#!/bin/bash

ROBOT_NAME="base_act"
N=3

onshape-to-robot toddleroid/robot_descriptions/$ROBOT_NAME

echo "(1/${N}) Robot description generated!"

python toddleroid/robot_descriptions/process_urdf.py --robot-name $ROBOT_NAME

echo "(2/${N}) Processed robot description generated!"

python toddleroid/utils/vis_kine_tree.py \
    --path toddleroid/robot_descriptions/$ROBOT_NAME/$ROBOT_NAME.urdf \
    -o toddleroid/robot_descriptions/$ROBOT_NAME/${ROBOT_NAME}_kine_tree.png

echo "(3/${N}) Kinematic tree generated!"

# Ask user if they want to run the simulation
echo "Do you want to run the simulation? (y/n)"
read -r -p "> " run_simulation

if [ "$run_simulation" == "y" ]; then
    echo "Simulation running..."
    
    python toddleroid/robot_descriptions/run_urdf_pybullet.py --robot-name $ROBOT_NAME
else
    echo "Simulation skipped."
fi
