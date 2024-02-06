#!/bin/bash

ROBOT_NAME="base"
URDF_PATH=toddleroid/robot_descriptions/$ROBOT_NAME/$ROBOT_NAME.urdf
MJCF_PATH=toddleroid/robot_descriptions/$ROBOT_NAME/$ROBOT_NAME.xml

printf "Do you want to export urdf from onshape? (y/n)"
read -r -p " > " run_onshape

if [ "$run_onshape" == "y" ]; then
    printf "Exporting...\n\n"
    onshape-to-robot toddleroid/robot_descriptions/$ROBOT_NAME
else
    printf "Export skipped.\n\n"
fi

printf "Do you want to process the urdf? (y/n)"
read -r -p " > " run_process
if [ "$run_process" == "y" ]; then
    printf "Processing...\n\n"
    python toddleroid/robot_descriptions/process_urdf.py --robot-name $ROBOT_NAME
    
    printf "Visualizing the kinematic tree...\n\n"
    python toddleroid/utils/vis_kine_tree.py \
        --path $URDF_PATH \
        -o toddleroid/robot_descriptions/$ROBOT_NAME/${ROBOT_NAME}_kine_tree.png
else
    printf "Process skipped.\n\n"
fi

# Ask user if they want to run the simulation
printf "Do you want to run the pybullet simulation? (y/n)"
read -r -p " > " run_pybullet

if [ "$run_pybullet" == "y" ]; then
    printf "Simulation running...\n\n"
    python toddleroid/robot_descriptions/test_urdf.py --robot-name $ROBOT_NAME
else
    printf "Simulation skipped.\n\n"
fi


printf "Do you want to convert to MJCF (y/n)"

read -r -p " > " run_convert
if [ "$run_convert" == "y" ]; then
    printf "Converting... Click the button save_xml to save the model to mjmodel.xml in the root directory.\n\n"
    python -m mujoco.viewer --mjcf=$URDF_PATH

    printf "Processing...\n\n"
    python toddleroid/robot_descriptions/process_mjcf.py --robot-name $ROBOT_NAME
else
    printf "Process skipped.\n\n"
fi

printf "Do you want to run the mujoco simulation? (y/n)"
read -r -p " > " run_mujoco

if [ "$run_mujoco" == "y" ]; then
    printf "Simulation running...\n\n"
    python -m mujoco.viewer --mjcf=$MJCF_PATH
else
    printf "Simulation skipped.\n\n"
fi