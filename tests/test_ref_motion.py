import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot

jax.config.update("jax_default_device", jax.devices("cpu")[0])  # type: ignore


def test_walk_simple_ref():
    robot = Robot("toddlerbot")

    exp_name: str = "test_ref_motion"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"
    os.makedirs(exp_folder_path, exist_ok=True)

    sim = MuJoCoSim(robot, fixed_base=True, vis_type="render")

    from toddlerbot.motion_reference.walk_simple_ref import WalkSimpleReference

    walk_ref = WalkSimpleReference(
        robot,
        default_joint_pos=np.array(list(robot.default_joint_angles.values())),  # type: ignore
    )

    duration = 10
    for phase in tqdm(
        np.arange(0, duration, sim.control_dt),  # type: ignore
        desc="Running Ref Motion",
    ):
        state = walk_ref.get_state_ref(
            np.zeros(3),  # type: ignore
            np.array([1, 0, 0, 0]),
            phase=phase,
            command=np.zeros(3),  # type: ignore
        )
        joint_angles = np.asarray(state[13 : 13 + len(robot.joint_ordering)])  # type: ignore
        motor_angles = robot.joint_to_motor_angles(
            dict(zip(robot.joint_ordering, joint_angles))
        )
        sim.set_motor_angles(motor_angles)
        sim.step()

    sim.save_recording(exp_folder_path, sim.control_dt, 2)

    sim.close()


def test_walk_zmp_ref():
    robot = Robot("toddlerbot")

    exp_name: str = "test_ref_motion"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"
    os.makedirs(exp_folder_path, exist_ok=True)

    sim = MuJoCoSim(robot, fixed_base=True, vis_type="render")
    # sim.simulate(vis_type="render")

    from toddlerbot.motion_reference.walk_zmp_ref import WalkZMPReference

    walk_ref = WalkZMPReference(
        robot,
        default_joint_pos=jnp.array(list(robot.default_joint_angles.values())),  # type: ignore
    )

    with jax.disable_jit():
        walk_ref.plan(
            jnp.zeros(3, dtype=np.float32),  # type: ignore
            jnp.array([1, 0, 0, 0], dtype=np.float32),  # type: ignore
            0,
            jnp.array([0.3, 0.0, np.pi / 12], dtype=np.float32),  # type: ignore
            duration=10,
        )

    duration = 10
    for phase in tqdm(np.arange(0, duration, sim.dt), desc="Running Ref Motion"):  # type: ignore
        state = walk_ref.get_state_ref(
            jnp.zeros(3),  # type: ignore
            jnp.array([1, 0, 0, 0]),  # type: ignore
            phase=phase,
            command=jnp.zeros(3),  # type: ignore
        )
        sim.set_joint_angles(np.asarray(state[13 : 13 + len(robot.joint_ordering)]))  # type: ignore
        sim.step()

    sim.save_recording(exp_folder_path, sim.control_dt, 2)  # type: ignore

    sim.close()


if __name__ == "__main__":
    # test_walk_simple_ref()
    test_walk_zmp_ref()
