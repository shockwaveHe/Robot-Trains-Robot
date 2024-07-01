import os
import sys
import threading
import time

import mediapy as media
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_rotate_inverse

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim
from toddlerbot.utils.constants import SIM_TIMESTEP
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import precise_sleep, set_seed

pass

import torch  # noqa: E402


class IsaacViewer:
    def __init__(self, env):
        self.env = env
        self.enable_viewer_sync = True

        self.viewer = env.gym.create_viewer(env.sim, gymapi.CameraProperties())
        env.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        env.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
        )

        cam_pos = gymapi.Vec3(-1, -0.5, 0.5)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.3)
        env.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def visualize(self):
        # update viewer
        # check for window closed
        if self.env.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        # check for keyboard events
        for evt in self.env.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        if self.enable_viewer_sync:
            self.env.gym.step_graphics(self.sim)
            self.env.gym.draw_viewer(self.viewer, self.sim, False)
            self.env.gym.sync_frame_time(self.sim)
        else:
            self.env.gym.poll_viewer_events(self.viewer)

    def close(self):
        self.env.gym.destroy_viewer(self.viewer)


class IssacRenderer:
    def __init__(self, env, height=720, width=1280, frame_rate=24):
        self.env = env
        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.video_frames = []

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = width
        camera_properties.height = height
        self.camera_tensor = env.gym.create_camera_sensor(env.env, camera_properties)

        self.attach_to_actor()

    def attach_to_actor(self):
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135)
        )
        actor_handle = self.env.gym.get_actor_handle(self.env.env, 0)
        body_handle = self.env.gym.get_actor_rigid_body_handle(
            self.env.env, actor_handle, 0
        )
        self.env.gym.attach_camera_to_body(
            self.camera_tensor,
            self.env.env,
            body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION,
        )

    def visualize(self):
        time_curr = self.env.gym.get_sim_time(self.env.sim)
        if len(self.video_frames) < time_curr * self.frame_rate:
            self.env.gym.fetch_results(self.env.sim, True)
            self.env.gym.step_graphics(self.env.sim)
            self.env.gym.render_all_camera_sensors(self.env.sim)
            img = self.env.gym.get_camera_image(
                self.env.sim, self.env.env, self.camera_tensor, gymapi.IMAGE_COLOR
            )
            img = np.reshape(img, (self.height, self.width, 4))
            self.video_frames.append(img[..., :3])

    def save_recording(self, exp_folder_path):
        video_path = os.path.join(exp_folder_path, "isaac.mp4")
        media.write_video(video_path, self.video_frames, fps=self.frame_rate)

    def anim_pose_callback(self):
        # TODO: implement for blender rendering
        pass

    def close(self):
        pass


class IsaacSim(BaseSim):
    def __init__(self, robot, urdf_path=None, fixed=False, custom_parameters=[]):
        super().__init__()
        self.name = "isaac"
        self.robot = robot

        set_seed(3407)
        # initialize gym
        self.gym = gymapi.acquire_gym()
        # parse arguments
        args = gymutil.parse_arguments(
            description="Asset and Environment Information",
            custom_parameters=custom_parameters,
        )

        # create simulation context
        sim_params = gymapi.SimParams()
        sim_params.dt = SIM_TIMESTEP
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim = self.gym.create_sim(
            args.compute_device_id,
            args.graphics_device_id,
            args.physics_engine,
            sim_params,
        )

        if urdf_path is None:
            urdf_path = find_robot_file_path(robot.name, suffix="_isaac.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fixed  # fixe the base of the robot
        asset_options.default_dof_drive_mode = 3
        # merge bodies connected by fixed joints.
        asset_options.collapse_fixed_joints = True
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 1000.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        robot_asset = self.gym.load_asset(
            self.sim,
            os.path.dirname(urdf_path),
            os.path.basename(urdf_path),
            asset_options,
        )

        # print_asset_info(robot_asset, robot.name)

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.static_friction = 0.6
        plane_params.dynamic_friction = 0.6
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

        # Setup environment spacing
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Create one environment
        self.env = self.gym.create_env(self.sim, lower, upper, 1)
        # Add actors to environment
        start_pose = gymapi.Transform()
        start_pos_z = 0.2 if fixed else 0.0
        start_pose.p = gymapi.Vec3(0.0, 0.0, start_pos_z)
        self.gym.create_actor(self.env, robot_asset, start_pose, robot.name, 0, 0, 0)

        # print("=== Environment info: ================================================")

        self.actor_handle = self.gym.get_actor_handle(self.env, 0)
        # print_actor_info(gym, self.env, self.actor_handle)

        self._init_buffers()

        self.last_command = None
        self.thread = None
        self.stop_event = threading.Event()

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_state = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_names = self.gym.get_actor_dof_names(self.env, self.actor_handle)

    def reset_dof_state(self, q, dq=None):
        q_tensor = torch.tensor(q, dtype=torch.float)

        self.dof_state[:, 0] = q_tensor
        self.dof_state[:, 1] = (
            torch.zeros_like(q_tensor)
            if dq is None
            else torch.tensor(dq, dtype=torch.float)
        )
        self.root_state[:, 2] = -0.005

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_state)
        )

    def get_torso_pose(self):
        return np.array([0, 0, self.robot.com[-1]]), np.eye(3)

    def get_joint_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)

        joint_state_dict = {}
        time_curr = time.time()
        for name in self.robot.config["joints"]:
            dof_idx = self.dof_names.index(name)
            joint_state_dict[name] = JointState(
                time=time_curr,
                pos=self.dof_state[dof_idx, 0].item(),
                vel=self.dof_state[dof_idx, 0].item(),
            )

        return joint_state_dict

    def get_observation(self):
        joint_state_dict = self.get_joint_state()

        self.gym.refresh_actor_root_state_tensor(self.sim)

        root_state = {}
        root_state["quaternion"] = np.array(
            [self.root_state[0, 6], *self.root_state[0, 3:6]]
        )
        ang_vel_tensor = quat_rotate_inverse(
            self.root_state[:, 3:7], self.root_state[:, 10:13]
        )
        root_state["angular_velocity"] = ang_vel_tensor.detach().cpu().numpy().squeeze()

        return joint_state_dict, root_state

    def set_joint_angles(self, joint_ctrls, ctrl_type="position"):
        self.last_command = (joint_ctrls, ctrl_type)

    def step_control(self):
        if self.last_command is None:
            return  # Do nothing if no command has been set yet

        joint_ctrls, ctrl_type = self.last_command

        if ctrl_type == "position":
            joint_state_dict = self.get_joint_state()
            joint_ctrls_tensor = torch.zeros(len(joint_ctrls), dtype=torch.float)
            for i, (name, state) in enumerate(joint_state_dict.items()):
                joint_ctrls_tensor[i] = self.robot.config.motor_params[name].kp * (
                    joint_ctrls[name] - state.pos
                )
        elif ctrl_type == "torque":
            joint_ctrls_tensor = torch.zeros(len(joint_ctrls), dtype=torch.float)
            for i, (name, state) in enumerate(joint_state_dict.items()):
                joint_ctrls_tensor[i] = joint_ctrls[name]
        else:
            raise ValueError(f"Unknown control type: {ctrl_type}")

        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(joint_ctrls_tensor)
        )

    def rollout(self, joint_angles_traj):
        joint_state_dict = self.get_joint_state()
        joint_state_traj = []
        for i, joint_angles in enumerate(joint_angles_traj):
            self.set_joint_angles(joint_angles, ctrl_type="position")

            self.step_control()

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)

            joint_state_dict = self.get_joint_state()

            for joint_name in joint_state_dict:
                joint_state_dict[joint_name].time = i * SIM_TIMESTEP

            joint_state_traj.append(joint_state_dict)

        return joint_state_traj

    def run_simulation(self, headless=True):
        self.thread = threading.Thread(target=self.simulate, args=(headless,))
        self.thread.start()

    def simulate(self, headless):
        if headless:
            self.visualizer = IssacRenderer(self)
        else:
            self.visualizer = IsaacViewer(self)

        # simulation loop
        while not self.stop_event.is_set():
            step_start = time.time()

            self.step_control()

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)

            self.visualizer.visualize()

            time_until_next_step = SIM_TIMESTEP - (time.time() - step_start)
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

        self.visualizer.close()

    def close(self):
        if self.thread is not None and threading.current_thread() is not self.thread:
            # Wait for the thread to finish if it's not the current thread
            self.stop_event.set()
            self.thread.join()

        # Cleanup the simulator
        self.gym.destroy_sim(self.sim)
