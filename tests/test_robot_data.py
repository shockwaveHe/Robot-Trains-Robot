import os

import numpy as np
import numpy.typing as npt
import pytest

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot


def arrays_are_close(
    arr1: npt.NDArray[np.float32], arr2: npt.NDArray[np.float32], tol: float = 1e-9
):
    """
    Check if two numpy arrays are element-wise equal within a tolerance.
    """
    return np.allclose(arr1, arr2, atol=tol)


def test_com():
    robot = Robot("toddlerbot")
    sim = MuJoCoSim(robot)

    assert arrays_are_close(sim.get_com(), robot.com)


# def test_ankle_IK():
#     model = mujoco.MjModel.from_xml_path(_XML.as_posix())


if __name__ == "__main__":
    pytest.main()
