import platform
import time
from typing import Dict, Optional, Tuple

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
from toddlerbot.utils.math_utils import interpolate_action

SYS_NAME = platform.system()


class BalancePDPolicy(BasePolicy, policy_name="balance_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
        camera: Optional[Camera] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        ip: str = "127.0.0.1",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
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

        state_ref = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),  # Position
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Quaternion
                np.zeros(3, dtype=np.float32),  # Linear velocity
                np.zeros(3, dtype=np.float32),  # Angular velocity
                self.default_motor_pos,  # Motor positions
                self.default_joint_pos,  # Joint positions
                np.ones(2, dtype=np.float32),  # Stance mask
            ]
        )
        self.state_ref = self.balance_ref.get_state_ref(
            state_ref, 0.0, self.fixed_command
        )

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

        self.camera = None
        if camera is not None:
            self.camera = camera
        elif SYS_NAME != "Darwin":
            try:
                self.camera = Camera()
            except Exception:
                pass

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

        self.is_prepared = False

    def reset(self):
        self.state_ref[:3] = np.zeros(3, dtype=np.float32)
        self.state_ref[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.state_ref[13 : 13 + self.robot.nu] = self.default_motor_pos.copy()
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
        return self.default_motor_pos[self.arm_motor_indices]

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 7.0 if is_real else 2.0
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

        if self.camera is not None:
            jpeg_frame, self.camera_frame = self.camera.get_jpeg()
        else:
            jpeg_frame = None

        if self.zmq_sender is not None:
            send_msg = ZMQMessage(time=time.time(), camera_frame=jpeg_frame)
            self.zmq_sender.send_msg(send_msg)

        control_inputs = self.last_control_inputs
        if self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()
        elif msg is not None:
            control_inputs = msg.control_inputs

        self.last_control_inputs = control_inputs

        if control_inputs is None:
            command = self.fixed_command
        else:
            command = self.get_command(control_inputs)

        time_curr = self.step_curr * self.control_dt
        arm_motor_pos = self.get_arm_motor_pos(obs)
        arm_joint_pos = self.balance_ref.arm_fk(arm_motor_pos)
        self.state_ref[13 + self.arm_motor_indices] = arm_motor_pos
        self.state_ref[13 + self.robot.nu + self.arm_joint_indices] = arm_joint_pos
        self.state_ref = self.balance_ref.get_state_ref(
            self.state_ref, time_curr, command
        )

        motor_target = self.state_ref[13 : 13 + self.robot.nu]
        # Override motor target with reference motion or teleop motion
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.step_curr += 1

        return control_inputs, motor_target
