import argparse
import os
import time
from typing import List

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import set_seed


def arrays_are_close(
    arr1: npt.NDArray[np.float32], arr2: npt.NDArray[np.float32], tol: float = 1e-6
) -> bool:
    """
    Check if two numpy arrays are element-wise equal within a tolerance.
    """
    return np.allclose(arr1, arr2, atol=tol)


def test_mass_properties(robot: Robot):
    sim = MuJoCoSim(robot)
    sim.forward()

    print(sim.get_mass())


def test_kinematics(robot: Robot) -> None:
    exp_name: str = "test_kinematics"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"
    os.makedirs(exp_folder_path, exist_ok=True)

    # from toddlerbot.visualization.vis_plot import plot_ankle_mapping, plot_waist_mapping

    # plot_waist_mapping(robot.config["joints"], robot.waist_ik, exp_folder_path)
    # plot_ankle_mapping(robot.config["joints"], robot.ankle_ik, exp_folder_path)

    sim = MuJoCoSim(robot, fixed_base=True)

    sim.set_motor_angles(robot.init_motor_angles)
    sim.step()

    mujoco_q = sim.get_observation().motor_pos
    init_q = np.array(list(robot.init_motor_angles.values()))

    assert arrays_are_close(mujoco_q, init_q, tol=1e-3)

    set_seed(0)
    failure_count = 0
    # TODO: There are certain upper body joint angles that may fail to reach the desired position
    # {'left_sho_roll': 0.0732, 'left_sho_yaw_driven': 0.0635, 'left_elbow_roll': 0.0447, 'left_elbow_yaw_driven': 0.0446, 'left_wrist_pitch': 0.0351}
    # {'right_sho_roll': 0.0556, 'right_sho_yaw_driven': 0.0308, 'right_elbow_roll': 0.0238}
    # {'left_sho_roll': 0.2791, 'left_sho_yaw_driven': 0.0623, 'left_elbow_roll': 0.0597, 'left_elbow_yaw_driven': 0.0317}
    # {'left_sho_pitch': 0.0201}
    for _ in tqdm(range(20)):
        random_motor_angles = robot.sample_motor_angles()
        random_joint_angles = robot.motor_to_joint_angles(random_motor_angles)
        random_motor_angles_copy = robot.joint_to_motor_angles(random_joint_angles)

        robot_act = np.array(list(random_motor_angles.values()))
        robot_act_copy = np.array(list(random_motor_angles_copy.values()))
        assert arrays_are_close(robot_act, robot_act_copy, tol=1e-3)

        sim.set_motor_angles(random_motor_angles)
        time.sleep(3.0)
        mujoco_q = sim.get_observation().motor_pos

        robot_q = np.array(list(random_motor_angles.values()))
        if not arrays_are_close(mujoco_q, robot_q, tol=2e-2):
            mask = np.abs(mujoco_q - robot_q) > 2e-2
            joint_names: List[str] = [
                robot.joint_ordering[index] for index in np.where(mask)[0]
            ]
            joint_diff = list(np.abs(mujoco_q - robot_q)[mask])
            print({name: round(diff, 4) for name, diff in zip(joint_names, joint_diff)})

        sim.set_motor_angles(robot.init_motor_angles)
        time.sleep(3.0)

    print(f"Failure count: {failure_count}")
    # sim.save_recording(exp_folder_path)

    sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)

    test_mass_properties(robot)
    # test_kinematics()
