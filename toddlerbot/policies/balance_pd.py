import platform
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.motion.balance_pd_ref import BalancePDReference
from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.math_utils import interpolate_action, mat2euler, euler2mat

# from toddlerbot.utils.misc_utils import profile

# from toddlerbot.utils.misc_utils import profile

SYS_NAME = platform.system()


class BalancePDPolicy(BasePolicy, policy_name="balance_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
        cameras: Optional[List[Camera]] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        ip: str = "",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        use_torso_pd: bool = True,
    ):
        super().__init__(name, robot, init_motor_pos)

        self.balance_ref = BalancePDReference(
            robot, self.control_dt, arm_playback_speed=0.0
        )
        cfg = get_env_config("balance")
        self.command_range = np.array(cfg.commands.command_range, dtype=np.float32)
        self.num_commands = len(self.command_range)

        self.zero_command = np.zeros(self.num_commands, dtype=np.float32)
        self.fixed_command = (
            self.zero_command if fixed_command is None else fixed_command
        )
        self.use_torso_pd = use_torso_pd

        self.state_ref: Optional[npt.NDArray[np.float32]] = None

        self.joystick = joystick
        if joystick is None:
            try:
                self.joystick = Joystick()
            except Exception:
                pass

        self.zmq_receiver = None
        if zmq_receiver is not None:
            self.zmq_receiver = zmq_receiver
        elif SYS_NAME != "Darwin":
            self.zmq_receiver = ZMQNode(type="receiver")

        self.zmq_sender = None
        if zmq_sender is not None:
            self.zmq_sender = zmq_sender
        elif SYS_NAME != "Darwin":
            self.zmq_sender = ZMQNode(type="sender", ip=ip)

        self.left_eye = None
        self.right_eye = None
        if cameras is not None:
            self.left_eye = cameras[0]
            self.right_eye = cameras[1]
        elif SYS_NAME != "Darwin":
            try:
                self.left_eye = Camera("left")
                self.right_eye = Camera("right")
            except Exception:
                pass

        self.capture_frame = False

        self.msg = None
        self.is_running = False
        self.is_button_pressed = False
        self.is_ended = False
        self.last_control_inputs: Dict[str, float] = {}
        self.step_curr = 0

        self.arm_motor_pos = None
        self.fsr = np.zeros(2, dtype=np.float32)
        self.camera_frame = None

        self.last_arm_motor_pos = None
        self.arm_delta_max = 0.2
        self.last_gripper_pos = np.zeros(2, dtype=np.float32)
        self.gripper_delta_max = 0.5

        self.desired_torso_pitch = 0.0
        self.desired_torso_roll = 0.0
        self.last_torso_pitch = 0.0
        self.last_torso_roll = 0.0
        self.torso_kp = 0.5
        self.torso_kd = 0.01
        self.hip_pitch_indices = np.array(
            [
                robot.motor_ordering.index("left_hip_pitch"),
                robot.motor_ordering.index("right_hip_pitch"),
            ]
        )
        self.hip_pitch_signs = np.array([1.0, -1.0], dtype=np.float32)
        self.hip_roll_indices = np.array(
            [
                robot.motor_ordering.index("left_hip_roll"),
                robot.motor_ordering.index("right_hip_roll"),
            ]
        )
        self.hip_roll_signs = np.array([1.0, -1.0], dtype=np.float32)

        self.is_prepared = False
        self.prep_duration = 7.0

        self.is_ready = False
        self.manip_duration = 0.0
        self.manip_motor_pos = self.default_motor_pos.copy()

    def reset(self):
        if self.state_ref is not None:
            self.state_ref[:3] = np.zeros(3, dtype=np.float32)
            self.state_ref[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.state_ref[13 : 13 + self.robot.nu] = self.manip_motor_pos.copy()
            self.state_ref[13 + self.robot.nu : 13 + 2 * self.robot.nu] = (
                self.default_joint_pos.copy()
            )

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        command = np.zeros(len(self.command_range), dtype=np.float32)
        for task, input in control_inputs.items():
            if task in self.name:
                if abs(input) > 0.5:
                    # Button is pressed
                    if not self.is_button_pressed:
                        self.is_button_pressed = True  # Mark the button as pressed
                        self.is_running = not self.is_running  # Toggle logging

                        if not self.is_running:
                            self.is_ended = True

                        print(
                            f"\nLogging is now {'enabled' if self.is_running else 'disabled'}."
                        )
                else:
                    # Button is released
                    self.is_button_pressed = False  # Reset button pressed state

            elif task == "look_left" and input > 0:
                command[0] = input * self.command_range[0][1]
            elif task == "look_right" and input > 0:
                command[0] = input * self.command_range[0][0]
            elif task == "look_up" and input > 0:
                command[1] = input * self.command_range[1][1]
            elif task == "look_down" and input > 0:
                command[1] = input * self.command_range[1][0]
            elif task == "lean_left" and input > 0:
                command[3] = input * self.command_range[3][0]
            elif task == "lean_right" and input > 0:
                command[3] = input * self.command_range[3][1]
            elif task == "twist_left" and input > 0:
                command[4] = input * self.command_range[4][0]
            elif task == "twist_right" and input > 0:
                command[4] = input * self.command_range[4][1]
            elif task == "squat":
                command[5] = np.interp(
                    input,
                    [-1, 0, 1],
                    [self.command_range[5][1], 0.0, self.command_range[5][0]],
                )

        return command

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        return self.manip_motor_pos[self.arm_motor_indices]

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        if not self.is_prepared:
            self.is_prepared = True
            if not is_real:
                self.prep_duration -= 5.0

            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                end_time=5.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        if not self.is_ready:
            self.is_ready = True

            self.manip_time, self.manip_action = self.move(
                -self.control_dt,
                self.default_motor_pos,
                self.manip_motor_pos,
                self.manip_duration,
            )

        if obs.time - self.prep_duration < self.manip_duration:
            action = np.asarray(
                interpolate_action(
                    obs.time - self.prep_duration, self.manip_time, self.manip_action
                )
            )
            return {}, action

        msg = None
        if self.msg is not None:
            msg = self.msg
        elif self.zmq_receiver is not None:
            msg = self.zmq_receiver.get_msg()

        # print(f"msg: {msg}")

        if msg is not None:
            # print(f"latency: {abs(time.time() - msg.time) * 1000:.2f} ms")
            if abs(time.time() - msg.time) < 1:
                self.fsr = msg.fsr
                self.arm_motor_pos = msg.action
                if self.last_arm_motor_pos is not None:
                    self.arm_motor_pos = np.clip(
                        self.arm_motor_pos,
                        self.last_arm_motor_pos - self.arm_delta_max,
                        self.last_arm_motor_pos + self.arm_delta_max,
                    )
                self.last_arm_motor_pos = self.arm_motor_pos

                if (
                    self.robot.has_gripper
                    and self.arm_motor_pos is not None
                    and self.fsr is not None
                ):
                    gripper_pos = self.fsr / 100 * self.motor_limits[-2:, 1]
                    gripper_pos = np.clip(
                        gripper_pos,
                        self.last_gripper_pos - self.gripper_delta_max,
                        self.last_gripper_pos + self.gripper_delta_max,
                    )
                    self.arm_motor_pos = np.concatenate(
                        [self.arm_motor_pos, gripper_pos]
                    )
                    self.last_gripper_pos = gripper_pos

                if self.arm_motor_pos is not None:
                    self.arm_motor_pos = np.clip(
                        self.arm_motor_pos,
                        self.motor_limits[self.arm_motor_indices, 0],
                        self.motor_limits[self.arm_motor_indices, 1],
                    )
            else:
                print("\nstale message received, discarding")

        if self.left_eye is not None and self.capture_frame:
            jpeg_frame, self.camera_frame = self.left_eye.get_jpeg()
        else:
            jpeg_frame = None

        if self.zmq_sender is not None:
            send_msg = ZMQMessage(time=time.time(), camera_frame=jpeg_frame)
            self.zmq_sender.send_msg(send_msg)

        control_inputs = self.last_control_inputs
        if self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()
        elif msg is not None and msg.control_inputs is not None:
            control_inputs = msg.control_inputs

        self.last_control_inputs = control_inputs

        if control_inputs is None:
            command = self.fixed_command
        else:
            command = self.get_command(control_inputs)

        time_curr = self.step_curr * self.control_dt
        arm_motor_pos = self.get_arm_motor_pos(obs)
        arm_joint_pos = self.balance_ref.arm_fk(arm_motor_pos)

        if self.state_ref is None:
            manip_joint_pos = np.array(
                list(
                    self.robot.motor_to_joint_angles(
                        dict(zip(self.robot.motor_ordering, self.manip_motor_pos))
                    ).values()
                ),
                dtype=np.float32,
            )
            state_ref = np.concatenate(
                [
                    np.zeros(3, dtype=np.float32),  # Position
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Quaternion
                    np.zeros(3, dtype=np.float32),  # Linear velocity
                    np.zeros(3, dtype=np.float32),  # Angular velocity
                    self.manip_motor_pos,  # Motor positions
                    manip_joint_pos,
                    np.ones(2, dtype=np.float32),  # Stance mask
                ]
            )
            self.state_ref = np.asarray(
                self.balance_ref.get_state_ref(state_ref, 0.0, command)
            )

        self.state_ref[13 + self.arm_motor_indices] = arm_motor_pos
        self.state_ref[13 + self.robot.nu + self.arm_joint_indices] = arm_joint_pos
        self.state_ref = np.asarray(
            self.balance_ref.get_state_ref(self.state_ref, time_curr, command)
        )

        motor_target = self.state_ref[13 : 13 + self.robot.nu]

        if self.use_torso_pd:
            # Apply PD control based on torso pitch angle
            waist_roll, waist_yaw = self.robot.waist_fk(obs.motor_pos[self.waist_motor_indices])
            waist_mat = euler2mat([-waist_roll, 0.0, -waist_yaw])
            torso_mat = euler2mat(obs.euler) @ waist_mat.T
            torso_euler = mat2euler(torso_mat)

            # print(f"waist_roll: {waist_roll:.2f}, waist_yaw: {waist_yaw:.2f}")
            # print(f"torso_euler: {torso_euler}")
            
            current_roll = torso_euler[0]
            current_pitch = torso_euler[1]
            roll_error = self.desired_torso_roll - current_roll
            roll_vel = (current_roll - self.last_torso_roll) / self.control_dt
            pitch_error = self.desired_torso_pitch - current_pitch
            pitch_vel = (current_pitch - self.last_torso_pitch) / self.control_dt
            roll_pd_output = (
                self.torso_kp * roll_error - self.torso_kd * roll_vel
            )
            pitch_pd_output = (
                self.torso_kp * pitch_error - self.torso_kd * pitch_vel
            )
            # Adjust hip pitch motor target
            # print(f"current_roll: {current_roll:.2f}, roll_vel: {roll_vel:.2f}, roll_pd_output: {roll_pd_output:.2f}")
            # print(f"current_pitch: {current_pitch:.2f}, pitch_vel: {pitch_vel:.2f}, pitch_pd_output: {pitch_pd_output:.2f}")

            motor_target[self.hip_roll_indices] += roll_pd_output * self.hip_roll_signs
            motor_target[self.hip_pitch_indices] += pitch_pd_output * self.hip_pitch_signs

            self.last_torso_roll = current_roll
            self.last_torso_pitch = current_pitch

        # Override motor target with reference motion or teleop motion
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.step_curr += 1
        self.last_torso_pitch = current_pitch

        return control_inputs, motor_target
