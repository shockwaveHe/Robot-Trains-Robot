#!/bin/bash

# YELLOW='\033[0;33m'
# NC='\033[0m' # No Color

##### toddlerbot #####
# ROBOT_NAME="toddlerbot"
# BODY_NAME="4R_body"
# ARM_NAME="OP3_arm"
# LEG_NAME="3R+RH5_leg"
# DOC_ID_LIST="d2a8be5ce536cd2e18740efa d364b4c22233fe6e37effabe d364b4c22233fe6e37effabe 93bd073d2ef7800c8ba429de 93bd073d2ef7800c8ba429de"
# ASSEMBLY_LIST="4R_body left_3R+RH5_leg right_3R+RH5_leg left_OP3_arm right_OP3_arm"
# # DOC_ID_LIST="d2a8be5ce536cd2e18740efa"
# # ASSEMBLY_LIST="4R_body"

##### toddlerbot_legs #####
# ROBOT_NAME="toddlerbot_legs"
# BODY_NAME="no_body"
# LEG_NAME="3R+RH5_leg"
# DOC_ID_LIST="dca63e30dcbfe66f561f5fd4 d364b4c22233fe6e37effabe d364b4c22233fe6e37effabe"
# ASSEMBLY_LIST="no_body left_3R+RH5_leg right_3R+RH5_leg"
# # DOC_ID_LIST="dca63e30dcbfe66f561f5fd4"
# # ASSEMBLY_LIST="no_body"

##### toddlerbot_legs #####
ROBOT_NAME="sysID_XC430"
BODY_NAME="sysID_XC430"
DOC_ID_LIST="4b8df5a39fb5e7db7afa93b4"
ASSEMBLY_LIST="sysID_XC430"

REPO_NAME="toddlerbot"
URDF_PATH=$REPO_NAME/robot_descriptions/$ROBOT_NAME/$ROBOT_NAME.urdf
MJCF_FIXED_PATH=$REPO_NAME/robot_descriptions/$ROBOT_NAME/${ROBOT_NAME}_fixed.xml
CONFIG_PATH=$REPO_NAME/robot_descriptions/$ROBOT_NAME/config.json

# shellcheck disable=SC1091
source "$HOME/.bashrc"

printf "Do you want to export urdf from onshape? (y/n)"
read -r -p " > " run_onshape

if [ "$run_onshape" == "y" ]; then
    printf "Exporting...\n\n"
    python $REPO_NAME/robot_descriptions/get_urdf.py --doc-id-list $DOC_ID_LIST --assembly-list $ASSEMBLY_LIST
else
    printf "Export skipped.\n\n"
fi


printf "Do you want to process the urdf? (y/n)"
read -r -p " > " run_process
if [ "$run_process" == "y" ]; then
    printf "Processing...\n\n"
    # Construct the command with mandatory arguments
    cmd="python $REPO_NAME/robot_descriptions/assemble_urdf.py --robot-name $ROBOT_NAME --body-name $BODY_NAME"
    if [ -n "$ARM_NAME" ]; then
        cmd+=" --arm-name $ARM_NAME"
    fi
    if [ -n "$LEG_NAME" ]; then
        cmd+=" --leg-name $LEG_NAME"
    fi
    eval "$cmd"

    printf "Visualizing the kinematic tree...\n\n"
    python $REPO_NAME/visualization/vis_kine_tree.py \
        --path $URDF_PATH \
        -o $REPO_NAME/robot_descriptions/$ROBOT_NAME/${ROBOT_NAME}_kine_tree.png

    # Check if the config file exists
    if [ -f "$CONFIG_PATH" ]; then
        printf "Configuration file already exists. Do you want to overwrite it? (y/n)"
        read -r -p " > " overwrite_config
        if [ "$overwrite_config" == "y" ]; then
            printf "Overwriting the configuration file...\n\n"
            python $REPO_NAME/robot_descriptions/write_config.py --robot-name $ROBOT_NAME
        else
            printf "Configuration file not overwritten.\n\n"
        fi
    else
        printf "Generating the configuration file...\n\n"
        python $REPO_NAME/robot_descriptions/write_config.py --robot-name $ROBOT_NAME
    fi

    printf "Have you updated the configs in the auto-generated config.json and collision_config.json? (y/n)"
    read -r -p " > " update_collision
    if [ "$update_collision" == "y" ]; then
        printf "Generating the collision files...\n\n"
        python $REPO_NAME/robot_descriptions/update_collisions.py --robot-name $ROBOT_NAME
    else
        printf "Collision files not updated.\n\n"
    fi

else
    printf "Process skipped.\n\n"
fi

# TODO: bring this back
# # Ask user if they want to run the simulation
# printf "Do you want to run the pybullet simulation? (y/n)"
# read -r -p " > " run_pybullet

# if [ "$run_pybullet" == "y" ]; then
#     printf "Simulation running...\n\n"
#     python $REPO_NAME/robot_descriptions/test_urdf.py --robot-name $ROBOT_NAME
# else
#     printf "Simulation skipped.\n\n"
# fi


printf "Do you want to convert to MJCF (y/n)"

read -r -p " > " run_convert
if [ "$run_convert" == "y" ]; then
    printf "Converting... \n1. Click the button save_xml to save the model to mjmodel.xml to the current directory.\n2. Close MuJoCo.\n\n"
    python -m mujoco.viewer --mjcf=$URDF_PATH

    printf "Processing...\n\n"
    python $REPO_NAME/robot_descriptions/process_mjcf.py --robot-name $ROBOT_NAME
else
    printf "Process skipped.\n\n"
fi

printf "Do you want to run the mujoco simulation? (y/n)"
read -r -p " > " run_mujoco

if [ "$run_mujoco" == "y" ]; then
    printf "Simulation running...\n\n"
    python -m mujoco.viewer --mjcf=$MJCF_FIXED_PATH
else
    printf "Simulation skipped.\n\n"
fi

printf "Do you want to create the isaac gym environment? (y/n)"
read -r -p " > " run_gym

if [ "$run_gym" == "y" ]; then
    printf "Creating the gym environment...\n\n"
    python $REPO_NAME/robot_descriptions/create_gym_env.py --robot-name $ROBOT_NAME
else
    printf "Gym environment creation skipped.\n\n"
fi