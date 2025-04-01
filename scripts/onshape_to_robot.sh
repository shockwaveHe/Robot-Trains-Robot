#!/bin/bash

# shellcheck disable=SC1091
# shellcheck disable=SC2086

# This script is used to convert onshape assembly to urdf and mjcf

YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --robot)
        ROBOT_NAME="$2"
        case "$ROBOT_NAME" in
            toddlerbot)
            BODY_NAME="toddlerbot"
            ARM_NAME="arm_umi"
            LEG_NAME="leg_reorder"
            DOC_ID_LIST="ff1e767f256dd9c8abf2206a 873c4e55df11ed20432d2975 873c4e55df11ed20432d2975 1b5c9dbba7df364619e54663 1b5c9dbba7df364619e54663"
            ASSEMBLY_LIST="$BODY_NAME left_$LEG_NAME right_$LEG_NAME left_$ARM_NAME right_$ARM_NAME"
            # DOC_ID_LIST="ff1e767f256dd9c8abf2206a"
            # ASSEMBLY_LIST="$BODY_NAME"
            ;;
            toddlerbot_2xm)
            BODY_NAME="toddlerbot_2xm"
            ARM_NAME="arm_umi"
            LEG_NAME="leg_2xm"
            # DOC_ID_LIST="ff1e767f256dd9c8abf2206a 873c4e55df11ed20432d2975 873c4e55df11ed20432d2975 1b5c9dbba7df364619e54663 1b5c9dbba7df364619e54663"
            # ASSEMBLY_LIST="$BODY_NAME left_$LEG_NAME right_$LEG_NAME left_$ARM_NAME right_$ARM_NAME"
            DOC_ID_LIST="6f1a2a766fbbc097a49abb91"
            ASSEMBLY_LIST="$BODY_NAME"
            ;;
            toddlerbot_gripper)
            BODY_NAME="toddlerbot"
            ARM_NAME="arm_gripper"
            LEG_NAME="leg_reorder"
            # DOC_ID_LIST="ff1e767f256dd9c8abf2206a 873c4e55df11ed20432d2975 873c4e55df11ed20432d2975 1b5c9dbba7df364619e54663 1b5c9dbba7df364619e54663"
            # ASSEMBLY_LIST="$BODY_NAME left_$LEG_NAME right_$LEG_NAME left_$ARM_NAME right_$ARM_NAME"
            DOC_ID_LIST="1b5c9dbba7df364619e54663 1b5c9dbba7df364619e54663"
            ASSEMBLY_LIST="left_$ARM_NAME right_$ARM_NAME"
            ;;
            toddlerbot_arms)
            BODY_NAME="toddlerbot_teleop"
            ARM_NAME="arm_teleop"
            # DOC_ID_LIST="ff1e767f256dd9c8abf2206a 1b5c9dbba7df364619e54663 1b5c9dbba7df364619e54663"
            # ASSEMBLY_LIST="$BODY_NAME left_$ARM_NAME right_$ARM_NAME"
            DOC_ID_LIST="ff1e767f256dd9c8abf2206a 1b5c9dbba7df364619e54663 1b5c9dbba7df364619e54663"
            ASSEMBLY_LIST="$BODY_NAME left_$ARM_NAME right_$ARM_NAME"
            ;;
            toddlerbot_active)
            BODY_NAME="toddlerbot_active"
            ARM_NAME="arm_active"
            LEG_NAME="leg_active"
            DOC_ID_LIST="ff1e767f256dd9c8abf2206a 873c4e55df11ed20432d2975 873c4e55df11ed20432d2975 1b5c9dbba7df364619e54663 1b5c9dbba7df364619e54663"
            ASSEMBLY_LIST="$BODY_NAME left_$LEG_NAME right_$LEG_NAME left_$ARM_NAME right_$ARM_NAME"
            # DOC_ID_LIST="ff1e767f256dd9c8abf2206a"
            # ASSEMBLY_LIST="$BODY_NAME"
            ;;
            sysID_XC330)
            DOC_ID_LIST="1fb5d9a88ac086a053c4340b"
            ASSEMBLY_LIST="sysID_XC330"
            BODY_NAME="sysID_XC330"
            ;;
            sysID_XC430)
            DOC_ID_LIST="1fb5d9a88ac086a053c4340b"
            ASSEMBLY_LIST="sysID_XC430"
            BODY_NAME="sysID_XC430"
            ;;
            sysID_2XC430)
            DOC_ID_LIST="1fb5d9a88ac086a053c4340b"
            ASSEMBLY_LIST="sysID_2XC430"
            BODY_NAME="sysID_2XC430"
            ;;
            sysID_2XL430)
            DOC_ID_LIST="1fb5d9a88ac086a053c4340b"
            ASSEMBLY_LIST="sysID_2XL430"
            BODY_NAME="sysID_2XL430"
            ;;
            sysID_XM430-W210)
            DOC_ID_LIST="1fb5d9a88ac086a053c4340b"
            ASSEMBLY_LIST="sysID_XM430-W210"
            BODY_NAME="sysID_XM430-W210"
            ;;
            sysID_XM430-W350)
            DOC_ID_LIST="1fb5d9a88ac086a053c4340b"
            ASSEMBLY_LIST="sysID_XM430-W350"
            BODY_NAME="sysID_XM430-W350"
            ;;
            *)
            echo -e "${YELLOW}Unknown robot name: $ROBOT_NAME.${NC}"
            ;;
        esac
        shift # past argument
        shift # past value
        ;;
        *)
        echo -e "${YELLOW}Unknown option: $1${NC}"
        shift # past unknown argument
        ;;
    esac
done

# Check if ROBOT_NAME is set
if [[ -z "$ROBOT_NAME" ]]; then
    echo -e "${YELLOW}Error: --robot argument is required.${NC}"
    exit 1
fi

REPO_NAME="toddlerbot"
URDF_PATH=$REPO_NAME/descriptions/$ROBOT_NAME/$ROBOT_NAME.urdf
MJCF_VIS_SCENE_PATH=$REPO_NAME/descriptions/$ROBOT_NAME/${ROBOT_NAME}_vis_scene.xml
CONFIG_PATH=$REPO_NAME/descriptions/$ROBOT_NAME/config.json

source "$HOME/.bashrc"

printf "Do you want to export urdf from onshape? (y/n)"
read -r -p " > " run_onshape

if [ "$run_onshape" == "y" ]; then
    # Check if the system is macOS
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Running update_onshape_config.py on macOS..."
        # Run the Python script
        python $REPO_NAME/descriptions/update_onshape_config.py
    fi

    printf "Exporting...\n\n"
    python $REPO_NAME/descriptions/get_urdf.py --doc-id-list $DOC_ID_LIST --assembly-list $ASSEMBLY_LIST
else
    printf "Export skipped.\n\n"
fi

if [ -n "$BODY_NAME" ]; then
    printf "Do you want to process the urdf? (y/n)"
    read -r -p " > " run_process
    if [ "$run_process" == "y" ]; then
        printf "Processing...\n\n"
        # Construct the command with mandatory arguments
        cmd="python $REPO_NAME/descriptions/assemble_urdf.py --robot $ROBOT_NAME --body-name $BODY_NAME"
        if [ -n "$ARM_NAME" ]; then
            cmd+=" --arm-name $ARM_NAME"
        fi
        if [ -n "$LEG_NAME" ]; then
            cmd+=" --leg-name $LEG_NAME"
        fi
        eval "$cmd"
    else
        printf "Process skipped.\n\n"
    fi
fi

# Check if the config file exists
if [ -f "$CONFIG_PATH" ]; then
    printf "Configuration file already exists. Do you want to overwrite it? (y/n)"
    read -r -p " > " overwrite_config
    if [ "$overwrite_config" == "y" ]; then
        printf "Overwriting the configuration file...\n\n"
        python $REPO_NAME/descriptions/add_configs.py --robot $ROBOT_NAME
    else
        printf "Configuration file not written.\n\n"
    fi
else
    printf "Generating the configuration file...\n\n"
    python $REPO_NAME/descriptions/add_configs.py --robot $ROBOT_NAME
fi

printf "Do you want to update the collision files? If so, make sure you have edited config_collision.json! (y/n)"
read -r -p " > " update_collision
if [ "$update_collision" == "y" ]; then
    printf "Generating the collision files...\n\n"
    python $REPO_NAME/descriptions/update_collisions.py --robot $ROBOT_NAME
else
    printf "Collision files not updated.\n\n"
fi

printf "Do you want to convert to MJCF (y/n)"

read -r -p " > " run_convert
if [ "$run_convert" == "y" ]; then
    printf "Converting... \n1. Click the button save_xml to save the model to mjmodel.xml to the current directory.\n2. Close MuJoCo.\n\n"
    python -m mujoco.viewer --mjcf=$URDF_PATH

    printf "Processing...\n\n"
    python $REPO_NAME/descriptions/process_mjcf.py --robot $ROBOT_NAME
else
    printf "Process skipped.\n\n"
fi

printf "Do you want to run the mujoco simulation? (y/n)"
read -r -p " > " run_mujoco

if [ "$run_mujoco" == "y" ]; then
    printf "Simulation running...\n\n"
    python -m mujoco.viewer --mjcf=$MJCF_VIS_SCENE_PATH
else
    printf "Simulation skipped.\n\n"
fi