import os
import time

import numpy as np
from tqdm import tqdm

from toddlerbot.motion_reference.walk_simple_ref import WalkSimpleReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot


def test_ref_motion():
    robot = Robot("toddlerbot")

    exp_name: str = "test_ref_motion"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"
    os.makedirs(exp_folder_path, exist_ok=True)

    sim = MuJoCoSim(robot, fixed_base=True, vis_type="render")
    # sim.simulate(vis_type="render")

    default_ctrl = sim.model.keyframe("home").ctrl  # type: ignore
    default_joint_angles = robot.motor_to_joint_angles(
        dict(zip(robot.motor_ordering, default_ctrl))  # type: ignore
    )

    walk_ref = WalkSimpleReference(
        robot,
        default_joint_pos=np.array(list(default_joint_angles.values())),  # type: ignore
    )

    duration = 10
    for phase in tqdm(np.arange(0, duration, sim.dt), desc="Running Ref Motion"):  # type: ignore
        state = walk_ref.get_state_ref(
            np.zeros(3),  # type: ignore
            np.array([1, 0, 0, 0]),
            phase=phase,
            command=np.zeros(6),  # type: ignore
        )
        sim.set_joint_angles(np.asarray(state[13 : 13 + len(robot.joint_ordering)]))  # type: ignore
        sim.step()

    sim.save_recording(exp_folder_path)

    sim.close()


if __name__ == "__main__":
    test_ref_motion()
