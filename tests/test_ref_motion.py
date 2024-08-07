import os
import time

import numpy as np
from tqdm import tqdm

from toddlerbot.reference_motion.walk_ref_motion import WalkRefMotionGenerator
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
    walk_ref = WalkRefMotionGenerator(robot, joint_pos_ref_scale=1.0)

    duration = 10
    for phase in tqdm(np.arange(0, duration, sim.dt), desc="Running Ref Motion"):  # type: ignore
        state = walk_ref.get_state(
            np.zeros(7, dtype=np.float32),
            phase=phase,
            command=np.zeros(6, dtype=np.float32),
        )
        sim.set_joint_angles(state[13 : 13 + len(robot.joint_ordering)])
        sim.step()

    sim.save_recording(exp_folder_path)

    sim.close()


if __name__ == "__main__":
    test_ref_motion()
