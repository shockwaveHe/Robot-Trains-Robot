import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class URDFConfig:
    robot_name: str = ""
    arm_name: str = ""
    leg_name: str = ""
    # The following are the default values for the config.json file
    documentId: str = "d2a8be5ce536cd2e18740efa"
    mergeSTLs: str = "all"
    mergeSTLsCollisions: bool = True
    simplifySTLs: str = "all"
    maxSTLSize: int = 1


def process_assembly(assembly_name, urdf_config, robot_dir, json_file_name):
    is_base = assembly_name == urdf_config.robot_name
    if is_base:
        assembly_path = robot_dir
    else:
        assembly_path = os.path.join(robot_dir, assembly_name)

    os.makedirs(assembly_path, exist_ok=True)
    json_file_path = os.path.join(assembly_path, json_file_name)

    # Map the URDFConfig to the desired JSON structure
    json_data = {
        "documentId": urdf_config.documentId,
        "outputFormat": "urdf",
        "assemblyName": assembly_name,
        "robotName": assembly_name,
        "addDummyBaseLink": is_base,
        "mergeSTLs": urdf_config.mergeSTLs,
        "mergeSTLsCollisions": urdf_config.mergeSTLsCollisions,
        "simplifySTLs": urdf_config.simplifySTLs,
        "maxSTLSize": urdf_config.maxSTLSize,
    }

    # Write the JSON data to a file
    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    # Execute the command
    subprocess.run(f"onshape-to-robot {assembly_path}", shell=True)


def run_onshape_to_robot(urdf_config, parallel=False):
    # Define the JSON file name
    assembly_list = [
        "left_" + urdf_config.arm_name,
        "right_" + urdf_config.arm_name,
        "left_" + urdf_config.leg_name,
        "right_" + urdf_config.leg_name,
        urdf_config.robot_name,
    ]
    json_file_name = "config.json"

    robot_dir = os.path.join("toddleroid", "robot_descriptions", urdf_config.robot_name)

    if parallel:
        # Use ThreadPoolExecutor to parallelize processing
        with ThreadPoolExecutor(max_workers=len(assembly_list)) as executor:
            futures = [
                executor.submit(
                    process_assembly,
                    assembly_name,
                    urdf_config,
                    robot_dir,
                    json_file_name,
                )
                for assembly_name in assembly_list
            ]

            # Optionally, wait for all futures to complete and handle exceptions
            for future in futures:
                try:
                    future.result()  # This will raise exceptions if any occurred within a thread
                except Exception as e:
                    print(f"Error processing assembly: {e}")
    else:
        # Process each assembly in series
        for assembly_name in assembly_list:
            process_assembly(assembly_name, urdf_config, robot_dir, json_file_name)


def main():
    parser = argparse.ArgumentParser(description="Process the urdf.")
    parser.add_argument(
        "--robot-name",
        type=str,
        required=True,
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--leg",
        type=str,
        default="",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Whether to run onshape-to-robot in parallel.",
    )
    args = parser.parse_args()

    if len(args.arm) == 0:
        args.arm = args.robot_name.split("arm")[0] + "arm"

    if len(args.leg) == 0:
        args.leg = args.robot_name.split("arm_")[1]

    run_onshape_to_robot(
        URDFConfig(robot_name=args.robot_name, arm_name=args.arm, leg_name=args.leg),
        args.parallel,
    )


if __name__ == "__main__":
    main()
