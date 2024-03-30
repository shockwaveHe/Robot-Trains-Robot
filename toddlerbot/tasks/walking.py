import argparse
import copy
import json
import pickle
import random
from typing import List, Optional, Tuple

import numpy as np

from toddlerbot.control.zmp_preview_control import *
from toddlerbot.planning.foot_step_planner import *
from toddlerbot.sim.mujoco_sim import MujoCoSim
from toddlerbot.sim.pybullet_sim import PyBulletSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.tasks.walking_configs import *
from toddlerbot.utils.file_utils import find_last_result_dir
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.vis_planning import *
from toddlerbot.utils.vis_plot import *

random.seed(0)


class Walking:
    """Class to handle the walking motion of a humanoid robot."""

    def __init__(
        self, robot: HumanoidRobot, config: WalkingConfig, joint_angles: List[float]
    ):
        self.robot = robot
        self.config = config
        self.fs_steps = round(config.plan_t_step / config.control_dt)
        self.x_offset_com_to_foot = robot.com[0]
        self.y_offset_com_to_foot = robot.offsets["y_offset_com_to_foot"]

        plan_params = FootStepPlanParameters(
            max_stride=np.array(config.plan_max_stride),
            t_step=config.plan_t_step,
            y_offset_com_to_foot=self.y_offset_com_to_foot,
        )
        self.fsp = FootStepPlanner(plan_params)

        control_params = ZMPPreviewControlParameters(
            com_z=robot.com[2] - config.squat_height,
            dt=config.control_dt,
            t_preview=config.control_t_preview,
            t_filter=config.control_t_filter,
            Q_val=config.control_cost_Q_val,
            R_val=config.control_cost_R_val,
            x_offset_com_to_foot=self.x_offset_com_to_foot,
            y_disp_zmp=config.y_offset_zmp - self.y_offset_com_to_foot,
        )
        self.pc = ZMPPreviewController(control_params)

        self.curr_pose = None
        self.com_curr = np.concatenate(
            [np.array(robot.com)[None, :2], np.zeros((2, 2))], axis=0
        )
        self.joint_angles = joint_angles

        self.idx = 0

        self.foot_steps = []
        self.com_ref_traj = []

        self.left_up = self.right_up = 0.0
        # Assume the initial state is the canonical pose
        self.theta_curr = 0.0
        self.left_pos = np.array([0.0, self.y_offset_com_to_foot, 0.0])
        self.right_pos = np.array([0.0, -self.y_offset_com_to_foot, 0.0])

    def plan_and_control(
        self,
        curr_pose: Optional[np.ndarray] = None,
        target_pose: Optional[np.ndarray] = None,
        last_robot_state_traj_data: Optional[dict] = None,
    ):
        if self.curr_pose is not None:
            curr_pose = self.curr_pose

        path, self.foot_steps = self.fsp.compute_steps(curr_pose, target_pose)
        zmp_ref_traj = self.pc.compute_zmp_ref_traj(self.foot_steps)
        if last_robot_state_traj_data is None:
            zmp_traj, self.com_ref_traj = self.pc.compute_com_traj(
                self.com_curr, zmp_ref_traj
            )
        else:
            zmp_traj, self.com_ref_traj = self.pc.filter_dynamics(
                zmp_ref_traj, last_robot_state_traj_data
            )

        self.com_curr = self.com_ref_traj[-1][0].copy()

        return path, self.foot_steps, zmp_ref_traj, zmp_traj, self.com_ref_traj.copy()

    def solve_joint_angles(self) -> Tuple[List[float], List[float], int]:
        if len(self.com_ref_traj) == 0:
            return "finished", self.joint_angles

        if abs(self.idx * self.config.control_dt - self.foot_steps[1].time) < 1e-6:
            self.foot_steps.pop(0)

            fs_curr = self.foot_steps[0]
            fs_next = self.foot_steps[1]
            self.theta_delta = (fs_next.position[2] - fs_curr.position[2]) / (
                self.fs_steps / 2
            )
            if fs_curr.support_leg == "left":
                if fs_next.support_leg == "right":
                    self.right_pos_target = fs_next.position
                else:
                    self.right_pos_target = np.array(
                        [
                            fs_next.position[0]
                            + np.sin(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[1]
                            - np.cos(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[2],
                        ]
                    )

                self.right_pos_delta = (self.right_pos_target - self.right_pos) / (
                    self.fs_steps / 2
                )
            elif fs_curr.support_leg == "right":
                if fs_next.support_leg == "left":
                    self.left_pos_target = fs_next.position
                else:
                    self.left_pos_target = np.array(
                        [
                            fs_next.position[0]
                            - np.sin(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[1]
                            + np.cos(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[2],
                        ]
                    )

                self.left_pos_delta = (self.left_pos_target - self.left_pos) / (
                    self.fs_steps / 2
                )

        self._compute_foot_offset()

        com_pos_ref = self.com_ref_traj.pop(0)
        self.joint_angles = self.robot.solve_ik(
            *self._compute_foot_pos(com_pos_ref),
            self.joint_angles,
        )
        self.idx += 1

        return "walking", self.joint_angles

    def _compute_foot_offset(self):
        # Need to update here
        idx_curr = self.idx % self.fs_steps
        up_start_idx = round(self.fs_steps / 4)
        up_end_idx = round(self.fs_steps / 2)
        up_period = up_end_idx - up_start_idx

        up_delta = self.config.foot_step_height / up_period
        support_leg = self.foot_steps[0].support_leg

        # Up or down foot movement
        if up_start_idx < idx_curr <= up_end_idx:
            if support_leg == "right":
                self.left_up += up_delta
            elif support_leg == "left":
                self.right_up += up_delta
        else:
            if support_leg == "right":
                self.left_up = max(self.left_up - up_delta, 0.0)
            elif support_leg == "left":
                self.right_up = max(self.right_up - up_delta, 0.0)

        # Move foot in the axes of x, y, theta
        if idx_curr > up_start_idx + up_period * 2:
            if support_leg == "right":
                self.left_pos = self.left_pos_target.copy()
            elif support_leg == "left":
                self.right_pos = self.right_pos_target.copy()
        elif idx_curr > up_start_idx:
            if support_leg == "right":
                self.left_pos += self.left_pos_delta
                self.theta_curr += self.theta_delta
            elif support_leg == "left":
                self.right_pos += self.right_pos_delta
                self.theta_curr += self.theta_delta

    def _compute_foot_pos(self, com_pos):
        if not self.config.rotate_torso:
            self.theta_curr = 0.0

        left_foot_theta = self.theta_curr - self.left_pos[2]
        right_foot_theta = self.theta_curr - self.right_pos[2]
        # print(
        #     f"theta_curr: {self.theta_curr}, left_foot_theta: {left_foot_theta}, right_foot_theta: {right_foot_theta}"
        # )
        left_foot_ori = [0, 0, left_foot_theta]
        right_foot_ori = [0, 0, right_foot_theta]

        com_x_offset = 0.0064
        com_y_offset = 0.034

        rotated_com_pos = np.array(
            [
                com_pos[0] + np.sin(self.theta_curr) * com_x_offset,
                com_pos[1] + (1 - np.cos(self.theta_curr)) * com_y_offset,
            ]
        )

        left_offset = self.left_pos[:2] - rotated_com_pos
        right_offset = self.right_pos[:2] - rotated_com_pos

        left_foot_pos = [
            left_offset[0]
            + np.cos(self.theta_curr) * self.x_offset_com_to_foot
            + np.sin(self.theta_curr) * self.y_offset_com_to_foot,
            left_offset[1]
            + np.sin(self.theta_curr) * self.x_offset_com_to_foot
            - np.cos(self.theta_curr) * self.y_offset_com_to_foot,
            self.config.squat_height + self.left_up,
        ]

        right_foot_pos = [
            right_offset[0]
            + np.cos(self.theta_curr) * self.x_offset_com_to_foot
            - np.sin(self.theta_curr) * self.y_offset_com_to_foot,
            right_offset[1]
            + np.sin(self.theta_curr) * self.x_offset_com_to_foot
            + np.cos(self.theta_curr) * self.y_offset_com_to_foot,
            self.config.squat_height + self.right_up,
        ]

        # if self.idx == 0:
        #     print("stop")

        return left_foot_pos, left_foot_ori, right_foot_pos, right_foot_ori


def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="sustaina_op",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="pybullet",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Time to sleep between steps.",
    )
    args = parser.parse_args()

    exp_name = f"walk_{args.robot_name}_{args.sim}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"

    config = walking_configs[f"{args.robot_name}_{args.sim}"]

    robot = HumanoidRobot(args.robot_name)

    joint_angles = robot.initialize_joint_angles()
    if robot.name == "robotis_op3":
        joint_angles["l_sho_roll"] = np.pi / 2
        joint_angles["r_sho_roll"] = -np.pi / 2
    elif robot.name == "base":
        joint_angles["left_sho_roll"] = -np.pi / 2
        joint_angles["right_sho_roll"] = np.pi / 2

    if args.sim == "pybullet":
        sim = PyBulletSim(robot)
    elif args.sim == "mujoco":
        sim = MujoCoSim(robot)
    elif args.sim == "real":
        sim = RealWorld(robot)
    else:
        raise ValueError("Unknown simulator")

    last_robot_state_traj_data = None
    if config.filter_dynamics:
        last_result_dir = find_last_result_dir("results", prefix=exp_name)
        if last_result_dir is not None:
            last_config_path = os.path.join(last_result_dir, "config.json")
            last_robot_state_traj_file_path = os.path.join(
                last_result_dir, "robot_state_traj_data.pkl"
            )
            if os.path.exists(last_config_path) and os.path.exists(
                last_robot_state_traj_file_path
            ):
                with open(last_config_path, "r") as f:
                    last_config = WalkingConfig(**json.load(f))
                    if last_config.filter_dynamics:
                        config = last_config  # Use the latest config to make sure the filter is valid
                        exp_folder_path = last_result_dir
                        with open(last_robot_state_traj_file_path, "rb") as f:
                            last_robot_state_traj_data = pickle.load(f)

    walking = Walking(robot, config, joint_angles)

    torso_pos_init, torso_mat_init = sim.get_torso_pose(robot)
    curr_pose = np.concatenate(
        [torso_pos_init[:2], [np.arctan2(torso_mat_init[1, 0], torso_mat_init[0, 0])]]
    )
    target_pose = np.array(config.target_pose_init)

    path, foot_steps, zmp_ref_traj, zmp_traj, com_ref_traj = walking.plan_and_control(
        curr_pose, target_pose, last_robot_state_traj_data
    )
    foot_steps_vis = copy.deepcopy(foot_steps)

    com_traj = []
    zmp_approx_traj = []

    time_start = time.time()
    time_seq_ref = []
    time_seq_dict = {}
    joint_angle_ref_dict = {}
    joint_angle_dict = {}

    # This function requires its parameters to be the same as its return values.
    def step_func(sim_step_idx, path, foot_steps, com_ref_traj, joint_angles):
        time_ref = time.time() - time_start
        time_seq_ref.append(time_ref)
        for name, angle in joint_angles.items():
            if name not in joint_angle_ref_dict:
                joint_angle_ref_dict[name] = []
            joint_angle_ref_dict[name].append(angle)

        joint_state_dict = sim.get_joint_state(robot)
        for name, joint_state in joint_state_dict.items():
            if name not in time_seq_dict:
                time_seq_dict[name] = []
                joint_angle_dict[name] = []

            time_seq_dict[name].append(joint_state.time - time_start)
            joint_angle_dict[name].append(joint_state.pos)

        if sim_step_idx >= config.actuator_steps:
            sim_step_idx = 0
            status, joint_angles = walking.solve_joint_angles()
            # print(f"joint_angles: {round_floats(list(joint_angles.values()), 6)}")

            torso_pos, torso_mat = sim.get_torso_pose(robot)
            torso_mat_delta = torso_mat @ torso_mat_init.T
            torso_theta = np.arctan2(torso_mat_delta[1, 0], torso_mat_delta[0, 0])
            # print(f"torso_pos: {torso_pos}, torso_theta: {torso_theta}")

            com_pos = [
                torso_pos[0] + np.cos(torso_theta) * walking.x_offset_com_to_foot,
                torso_pos[1] + np.sin(torso_theta) * walking.x_offset_com_to_foot,
            ]
            com_traj.append(com_pos)

            zmp_pos = sim.get_zmp(com_pos)
            zmp_approx_traj.append(zmp_pos)

            if status == "finished":
                tracking_error = np.array(target_pose) - np.array(
                    [*torso_pos[:2], torso_theta]
                )
                log(
                    f"Tracking error: {round_floats(tracking_error, 6)}",
                    header="Walking",
                )

        sim.set_joint_angles(robot, joint_angles)
        sim_step_idx += 1

        return sim_step_idx, path, foot_steps, com_ref_traj, joint_angles

    try:
        sim.simulate(
            step_func,
            (0, path, foot_steps, com_ref_traj, joint_angles),
            vis_flags=["foot_steps", "com_traj", "torso"],
            sleep_time=args.sleep_time,
        )
    finally:
        os.makedirs(exp_folder_path, exist_ok=True)

        with open(os.path.join(exp_folder_path, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=4)

        robot_state_traj_data = {
            "zmp_ref_traj": zmp_ref_traj,
            "zmp_traj": zmp_traj,
            "zmp_approx_traj": zmp_approx_traj,
            "com_ref_traj": com_ref_traj,
            "com_traj": com_traj,
        }

        file_suffix = ""
        if last_robot_state_traj_data is not None:
            file_suffix = "filter"

        if len(file_suffix) > 0:
            robot_state_traj_file_name = f"robot_state_traj_data_{file_suffix}.pkl"
        else:
            robot_state_traj_file_name = "robot_state_traj_data.pkl"

        with open(os.path.join(exp_folder_path, robot_state_traj_file_name), "wb") as f:
            pickle.dump(robot_state_traj_data, f)

        # plot_joint_tracking(
        #     time_seq_dict,
        #     time_seq_ref,
        #     joint_angle_dict,
        #     joint_angle_ref_dict,
        #     robot.config.motor_params,
        #     save_path=exp_folder_path,
        #     file_suffix=file_suffix,
        # )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect("equal")

        draw_footsteps(
            path,
            foot_steps_vis,
            robot.foot_size[:2],
            robot.offsets["y_offset_com_to_foot"],
            title=f"Footsteps Planning",
            save_path=exp_folder_path,
            file_suffix=file_suffix,
            ax=ax,
        )()

        plot_line_graph(
            [[record[1] for record in x] for x in robot_state_traj_data.values()],
            x=[[record[0] for record in x] for x in robot_state_traj_data.values()],
            title=f"Footsteps Planning",
            x_label="X",
            y_label="Y",
            save_config=True,
            save_path=exp_folder_path,
            file_suffix=file_suffix,
            legend_labels=list(robot_state_traj_data.keys()),
            ax=ax,
        )()


if __name__ == "__main__":
    main()
