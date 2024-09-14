#!/bin/bash

# YELLOW='\033[0;33m'
# NC='\033[0m' # No Color

##### toddlerbot #####
ROBOT_NAME="toddlerbot"
BODY_NAME="toddlerbot"
ARM_NAME="arm_gripper"
LEG_NAME="leg_XM430"
DOC_ID_LIST="6f1a2a766fbbc097a49abb91 d364b4c22233fe6e37effabe d364b4c22233fe6e37effabe cddbcb685a34c68f46ce1d48 cddbcb685a34c68f46ce1d48"
ASSEMBLY_LIST="toddlerbot left_leg_XM430 right_leg_XM430 left_arm_gripper right_arm_gripper"
# DOC_ID_LIST="cddbcb685a34c68f46ce1d48 cddbcb685a34c68f46ce1d48"
# ASSEMBLY_LIST="left_arm_gripper right_arm_gripper"

#### toddlerbot_legs #####
# ROBOT_NAME="toddlerbot_legs"
# BODY_NAME="toddlerbot_legs"
# LEG_NAME="leg_XM430"
# DOC_ID_LIST="6f1a2a766fbbc097a49abb91 d364b4c22233fe6e37effabe d364b4c22233fe6e37effabe"
# ASSEMBLY_LIST="toddlerbot_legs left_leg_XM430 right_leg_XM430"
# DOC_ID_LIST="6f1a2a766fbbc097a49abb91"
# ASSEMBLY_LIST="toddlerbot_legs"

#### toddlerbot_arms #####
# ROBOT_NAME="toddlerbot_arms"
# BODY_NAME="toddlerbot_arms"
# ARM_NAME="arm_hand"
# DOC_ID_LIST="6f1a2a766fbbc097a49abb91 cddbcb685a34c68f46ce1d48 cddbcb685a34c68f46ce1d48"
# ASSEMBLY_LIST="toddlerbot_arms left_arm_hand right_arm_hand"
# DOC_ID_LIST="cddbcb685a34c68f46ce1d48 cddbcb685a34c68f46ce1d48"
# ASSEMBLY_LIST="left_arm_hand right_arm_hand"

##### sysID_device #####
# MOTOR_TYPE="XC330"
# ROBOT_NAME="sysID_$MOTOR_TYPE"
# BODY_NAME="sysID_$MOTOR_TYPE"
# DOC_ID_LIST="4b8df5a39fb5e7db7afa93b4"
# ASSEMBLY_LIST="sysID_$MOTOR_TYPE"

REPO_NAME="toddlerbot"
URDF_PATH=$REPO_NAME/robot_descriptions/$ROBOT_NAME/$ROBOT_NAME.urdf
MJCF_FIXED_SCENE_PATH=$REPO_NAME/robot_descriptions/$ROBOT_NAME/${ROBOT_NAME}_fixed_scene.xml
CONFIG_PATH=$REPO_NAME/robot_descriptions/$ROBOT_NAME/config.json

# shellcheck disable=SC1091
source "$HOME/.bashrc"

printf "Do you want to export urdf from onshape? (y/n)"
read -r -p " > " run_onshape

if [ "$run_onshape" == "y" ]; then
    printf "Exporting...\n\n"
    # shellcheck disable=SC2086
    python $REPO_NAME/robot_descriptions/get_urdf.py --doc-id-list $DOC_ID_LIST --assembly-list $ASSEMBLY_LIST
else
    printf "Export skipped.\n\n"
fi


printf "Do you want to process the urdf? (y/n)"
read -r -p " > " run_process
if [ "$run_process" == "y" ]; then
    printf "Processing...\n\n"
    # Construct the command with mandatory arguments
    cmd="python $REPO_NAME/robot_descriptions/assemble_urdf.py --robot $ROBOT_NAME --body-name $BODY_NAME"
    if [ -n "$ARM_NAME" ]; then
        cmd+=" --arm-name $ARM_NAME"
    fi
    if [ -n "$LEG_NAME" ]; then
        cmd+=" --leg-name $LEG_NAME"
    fi
    eval "$cmd"

    # printf "Visualizing the kinematic tree...\n\n"
    # python $REPO_NAME/visualization/vis_kine_tree.py \
    #     --path $URDF_PATH \
    #     -o $REPO_NAME/robot_descriptions/$ROBOT_NAME/${ROBOT_NAME}_kine_tree.png
else
    printf "Process skipped.\n\n"
fi

# Check if the config file exists
if [ -f "$CONFIG_PATH" ]; then
    printf "Configuration file already exists. Do you want to overwrite it? (y/n)"
    read -r -p " > " overwrite_config
    if [ "$overwrite_config" == "y" ]; then
        printf "Overwriting the configuration file...\n\n"
        python $REPO_NAME/robot_descriptions/add_configs.py --robot $ROBOT_NAME
    else
        printf "Configuration file not written.\n\n"
    fi
else
    printf "Generating the configuration file...\n\n"
    python $REPO_NAME/robot_descriptions/add_configs.py --robot $ROBOT_NAME
fi

printf "Do you want to update the collision files? If so, make sure you have edited config_collision.json! (y/n)"
read -r -p " > " update_collision
if [ "$update_collision" == "y" ]; then
    printf "Generating the collision files...\n\n"
    python $REPO_NAME/robot_descriptions/update_collisions.py --robot $ROBOT_NAME
else
    printf "Collision files not updated.\n\n"
fi

printf "Do you want to convert to MJCF (y/n)"

read -r -p " > " run_convert
if [ "$run_convert" == "y" ]; then
    printf "Converting... \n1. Click the button save_xml to save the model to mjmodel.xml to the current directory.\n2. Close MuJoCo.\n\n"
    python -m mujoco.viewer --mjcf=$URDF_PATH

    printf "Processing...\n\n"
    python $REPO_NAME/robot_descriptions/process_mjcf.py --robot $ROBOT_NAME
else
    printf "Process skipped.\n\n"
fi

printf "Do you want to run the mujoco simulation? (y/n)"
read -r -p " > " run_mujoco

if [ "$run_mujoco" == "y" ]; then
    printf "Simulation running...\n\n"
    python -m mujoco.viewer --mjcf=$MJCF_FIXED_SCENE_PATH
else
    printf "Simulation skipped.\n\n"
fi