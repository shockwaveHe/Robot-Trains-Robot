import argparse
import time

import mujoco
import numpy as np

from toddlerbot.motion.balance_pd_ref import BalancePDReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--vis",
        type=str,
        default="view",
        help="The visualization type.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)
    if args.sim == "mujoco":
        sim = MuJoCoSim(robot, vis_type=args.vis, fixed_base="fixed" in args.robot)

    motion_ref = BalancePDReference(robot, dt=0.02)

    neck_joint_pos = np.radians([0, 35])
    waist_joint_pos = np.radians([15, 45])
    left_leg_joint_pos = np.zeros(6)
    left_leg_joint_pos[0] = np.radians(-10)
    left_leg_joint_pos[1] = np.radians(3.75)
    left_leg_joint_pos[4] = np.radians(3.75)
    left_leg_joint_pos[5] = np.radians(-10)

    right_leg_joint_pos = np.zeros(6)
    right_leg_joint_pos[0] = np.radians(90)
    right_leg_joint_pos[5] = np.radians(-45)
    left_arm_joint_pos = np.zeros(7)
    left_arm_joint_pos[0] = np.radians(-135)
    right_arm_joint_pos = np.zeros(7)
    right_arm_joint_pos[0] = np.radians(-90)

    joint_pos_target = np.concatenate(
        [
            neck_joint_pos,
            waist_joint_pos,
            left_leg_joint_pos,
            right_leg_joint_pos,
            left_arm_joint_pos,
            right_arm_joint_pos,
        ]
    )

    motor_pos_target = np.concatenate(
        [
            motion_ref.neck_ik(neck_joint_pos),
            motion_ref.waist_ik(waist_joint_pos),
            motion_ref.leg_ik(
                np.concatenate([left_leg_joint_pos, right_leg_joint_pos])
            ),
            motion_ref.arm_ik(
                np.concatenate([left_arm_joint_pos, right_arm_joint_pos])
            ),
        ]
    )

    state_ref = np.concatenate(
        [
            np.zeros(3),
            np.array([1, 0, 0, 0]),
            np.zeros(6),
            motor_pos_target,
            joint_pos_target,
            np.zeros(2),
        ]
    )

    qpos_ref = motion_ref.get_qpos_ref(state_ref)
    sim.set_qpos(qpos_ref)
    sim.forward()

    left_foot_site_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_center"
    )
    kp = 5.0

    while True:
        obs = sim.get_observation()

        com_pos = np.array(sim.data.body(0).subtree_com, dtype=np.float32)

        left_foot_center = sim.data.site_xpos[left_foot_site_id]

        motor_angles = dict(zip(robot.motor_ordering, motor_pos_target))

        motor_angles["left_ank_roll"] += kp * (left_foot_center[1] - com_pos[1])

        print(f"com_pos: {com_pos}, left_foot_center: {left_foot_center}")

        sim.set_motor_target(motor_angles)
        sim.step()
        time.sleep(0.02)
