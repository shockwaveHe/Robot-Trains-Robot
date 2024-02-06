import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p

from toddleroid.sim.mujoco_sim import MujoCoSim
from toddleroid.sim.pybullet_sim import PyBulletSim
from toddleroid.sim.robot import HumanoidRobot
from toddleroid.utils.data_utils import round_floats


def parse_urdf_for_closing_links(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    closing_links = {}
    for link in root.findall(".//link"):
        link_name = link.get("name")
        if link_name.startswith("closing_"):
            base_name = "_".join(link_name.split("_")[:-1])
            if base_name in closing_links:
                closing_links[base_name] = (closing_links[base_name], link_name)
            else:
                closing_links[base_name] = link_name

    return closing_links


def get_link_idx_by_name(robot_id, link_name):
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[12].decode("UTF-8") == link_name:
            return i
    return -1  # Base link if not found


def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="base",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="pybullet",
        choices=["pybullet", "mujoco"],
        help="The simulator to use.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Time to sleep between steps.",
    )
    args = parser.parse_args()

    robot = HumanoidRobot(args.robot_name)
    if args.sim == "pybullet":
        sim = PyBulletSim(robot, fixed=True)
    elif args.sim == "mujoco":
        sim = MujoCoSim(robot)
    else:
        raise ValueError("Unknown simulator")

    robot_dir = os.path.join(
        os.path.join("toddleroid", "robot_descriptions"),
        args.robot_name,
        args.robot_name + ".urdf",
    )
    closing_links_dict = parse_urdf_for_closing_links(robot_dir)

    pivot = [0, 0, 0]
    constraint_dict = {}
    for base_name, (link_name_1, link_name_2) in closing_links_dict.items():
        parent_link_idx = get_link_idx_by_name(robot.id, link_name_1)
        child_link_idx = get_link_idx_by_name(robot.id, link_name_2)

        # Create a revolute joint constraint between the closing links
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot.id,
            parentLinkIndex=parent_link_idx,
            childBodyUniqueId=robot.id,
            childLinkIndex=child_link_idx,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=pivot,
            childFramePosition=pivot,
        )
        constraint_dict[constraint_id] = (parent_link_idx, child_link_idx)

    controls = {}
    for idx in range(p.getNumJoints(robot.id)):
        name = p.getJointInfo(robot.id, idx)[1].decode("UTF-8")
        qIndex = p.getJointInfo(robot.id, idx)[3]
        if not "passive" in name and qIndex > -1:
            low, high = p.getJointInfo(robot.id, idx)[8:10]
            print(f"{name} has limits ({low}, {high})")
            controls[name] = p.addUserDebugParameter(name, low, high, 0)

    sim_step_idx = 0

    # This function requires its parameters to be the same as its return values.
    def step_func(sim_step_idx):
        # for constraint_id in constraint_dict.keys():
        #     print(f"constraint_id: {p.getConstraintState(constraint_id)}")

        joint_angles = {}
        for name in controls.keys():
            joint_angles[name] = p.readUserDebugParameter(controls[name])

        # print(f"joint_angles: {round_floats(joint_angles, 6)}")
        sim.set_joint_angles(robot, joint_angles)

        sim_step_idx += 1

        return (sim_step_idx,)

    sim.simulate(step_func, (sim_step_idx,), args.sleep_time, vis_flags=[])


if __name__ == "__main__":
    main()
