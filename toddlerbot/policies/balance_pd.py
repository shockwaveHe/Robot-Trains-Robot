import platform
import subprocess
import time
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.balance_env import BalanceCfg
from toddlerbot.policies import BasePolicy
from toddlerbot.ref_motion.balance_pd_ref import BalancePDReference
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

        # Run the command
        result = subprocess.run(
            f"sudo ntpdate -u {ip}",
            shell=True,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        print(result.stdout.strip())

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.motor_limits = np.array(
            [robot.joint_limits[name] for name in robot.motor_ordering]
        )

        self.balance_ref = BalancePDReference(
            robot, self.control_dt, arm_playback_speed=0.0
        )
        cfg = BalanceCfg()
        self.command_range = np.array(cfg.commands.command_range, dtype=np.float32)
        self.num_commands = len(self.command_range)

        if fixed_command is None:
            self.fixed_command = np.zeros(self.num_commands, dtype=np.float32)
        else:
            self.fixed_command = fixed_command

        self.state_ref = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),  # Position
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Quaternion
                np.zeros(3, dtype=np.float32),  # Linear velocity
                np.zeros(3, dtype=np.float32),  # Angular velocity
                self.default_joint_pos,  # Joint positions
                np.zeros_like(self.default_joint_pos),  # Joint velocities
                np.ones(2, dtype=np.float32),  # Stance mask
            ]
        )
        self.arm_motor_indices = np.array(self.balance_ref.arm_motor_indices)
        self.arm_joint_indices = np.array(self.balance_ref.arm_joint_indices)
        self.arm_gear_ratio = np.asarray(
            self.balance_ref.arm_gear_ratio, dtype=np.float32
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
        self.is_logging = False
        self.is_button_pressed = False
        self.n_logs = 1
        self.last_control_inputs: Dict[str, float] | None = None
        self.step_curr = 0

        self.fsr = np.zeros(2, dtype=np.float32)
        self.camera_frame = None

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            end_time=0.0,
        )

        print('\nBy default, logging is disabled. Press "menu" to toggle logging.')

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        command = np.zeros(len(self.command_range), dtype=np.float32)
        for task, input in control_inputs.items():
            if task == "manipulate":
                if abs(input) > 0.5:
                    # Button is pressed
                    if not self.is_button_pressed:
                        self.is_button_pressed = True  # Mark the button as pressed
                        self.is_logging = not self.is_logging  # Toggle logging

                        print(
                            f"\nLogging is now {'enabled' if self.is_logging else 'disabled'}."
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

        return command

    def get_arm_motor_pos(self) -> npt.NDArray[np.float32]:
        return self.default_motor_pos[self.arm_motor_indices]

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        # Preparation phase
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

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
        arm_joint_pos = self.get_arm_motor_pos() * self.arm_gear_ratio
        self.state_ref[13 + self.arm_joint_indices] = arm_joint_pos
        self.state_ref = self.balance_ref.get_state_ref(
            self.state_ref, time_curr, command
        )

        joint_angles = dict(
            zip(self.robot.joint_ordering, self.state_ref[13 : 13 + self.robot.nu])
        )
        # Convert joint positions to motor angles
        motor_target = np.array(
            list(self.robot.joint_to_motor_angles(joint_angles).values()),
            dtype=np.float32,
        )
        # Override motor target with reference motion or teleop motion
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.step_curr += 1

        return motor_target
