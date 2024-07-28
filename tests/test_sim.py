import time
from typing import Dict

import numpy as np
import numpy.typing as npt

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import set_seed


def arrays_are_close(
    arr1: npt.NDArray[np.float32], arr2: npt.NDArray[np.float32], tol: float = 1e-6
):
    """
    Check if two numpy arrays are element-wise equal within a tolerance.
    """
    return np.allclose(arr1, arr2, atol=tol)


def test_mass_properties():
    robot = Robot("toddlerbot")
    sim = MuJoCoSim(robot)

    assert abs(sim.get_mass() - 2.53174268) < 1e-6
    assert arrays_are_close(
        sim.get_com(), np.array([-0.0020665, 0.00086725, 0.31932396])
    )


def test_kinematics():
    robot = Robot("toddlerbot")
    sim = MuJoCoSim(robot, fixed=True)
    sim.simulate()

    sim.set_motor_angles(robot.init_motor_angles)
    mujoco_q = sim.get_observation()["q"]
    init_q = np.array(list(robot.init_joint_angles.values()))

    assert arrays_are_close(mujoco_q, init_q, tol=1e-3)

    set_seed(0)
    random_motor_angles: Dict[str, float] = {}
    for motor_name in robot.motor_ordering:
        random_motor_angles[motor_name] = np.random.uniform(
            robot.config["joints"][motor_name]["lower_limit"],
            robot.config["joints"][motor_name]["upper_limit"],
        )

    sim.set_motor_angles(random_motor_angles)
    time.sleep(2.0)
    mujoco_q = sim.get_observation()["q"]
    random_joint_angles = robot.motor_to_joint_angles(random_motor_angles)
    robot_q = np.array(list(random_joint_angles.values()))

    assert arrays_are_close(mujoco_q, robot_q, tol=1e-3)


if __name__ == "__main__":
    test_kinematics()
    # import pytest
    # pytest.main()
