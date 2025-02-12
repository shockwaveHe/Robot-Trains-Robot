import platform
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.reference.balance_pd_ref import BalancePDReference
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.math_utils import euler2mat, interpolate_action

# from toddlerbot.utils.misc_utils import profile


class BalancePDPolicy(BasePolicy, policy_name="balance_pd"):
    """Policy for balancing the robot using a PD controller."""

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
        """Initializes the control system for a robot, setting up various components such as joystick, cameras, and ZeroMQ nodes for communication.

        Args:
            name (str): The name of the control system.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial positions of the robot's motors.
            joystick (Optional[Joystick], optional): Joystick for manual control. Defaults to None.
            cameras (Optional[List[Camera]], optional): List of camera objects for visual input. Defaults to None.
            zmq_receiver (Optional[ZMQNode], optional): ZeroMQ node for receiving data. Defaults to None.
            zmq_sender (Optional[ZMQNode], optional): ZeroMQ node for sending data. Defaults to None.
            ip (str, optional): IP address for ZeroMQ communication. Defaults to an empty string.
            fixed_command (Optional[npt.NDArray[np.float32]], optional): Fixed command array for the robot. Defaults to None.
            use_torso_pd (bool, optional): Flag to use proportional-derivative control for the torso. Defaults to True.
        """
        super().__init__(name, robot, init_motor_pos)

        self.balance_ref = BalancePDReference(
            robot, self.control_dt, arm_playback_speed=0.0
        )
        self.command_range = np.array(
            [
                [-1.5, 1.5],
                [-1.5, 1.5],
                [0.0, 0.5],
                [-0.3, 0.3],
                [-1.5, 1.5],
                [-0.1, 0.1],
            ],
            dtype=np.float32,
        )
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

        sys_name = platform.system()
        self.zmq_receiver = None
        if zmq_receiver is not None:
            self.zmq_receiver = zmq_receiver
        elif sys_name != "Darwin":
            self.zmq_receiver = ZMQNode(type="receiver")

        self.zmq_sender = None
        if zmq_sender is not None:
            self.zmq_sender = zmq_sender
        elif sys_name != "Darwin":
            self.zmq_sender = ZMQNode(type="sender", ip=ip)

        self.left_eye = None
        self.right_eye = None
        if cameras is not None:
            self.left_eye = cameras[0]
            self.right_eye = cameras[1]
        elif sys_name != "Darwin":
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
        self.camera_frame: npt.NDArray[np.uint8] | None = None
        self.camera_time_list: List[float] = []
        self.camera_frame_list: List[npt.NDArray[np.uint8]] = []

        self.last_arm_motor_pos = None
        self.arm_delta_max = 0.2
        self.last_gripper_pos = np.zeros(2, dtype=np.float32)
        self.gripper_delta_max = 0.5

        self.left_sho_pitch_idx = robot.motor_ordering.index("left_sho_pitch")
        self.right_sho_pitch_idx = robot.motor_ordering.index("right_sho_pitch")
        self.left_sho_roll_idx = robot.motor_ordering.index("left_sho_roll")
        self.right_sho_roll_idx = robot.motor_ordering.index("right_sho_roll")

        self.desired_torso_pitch = -0.2  # -0.7 for the payload test
        self.desired_torso_roll = 0.0
        self.last_torso_pitch = 0.0
        self.last_torso_roll = 0.0
        self.torso_roll_kp = 0.2
        self.torso_roll_kd = 0.0
        self.torso_pitch_kp = 0.2
        self.torso_pitch_kd = 0.01
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
        self.hip_roll_signs = np.array([-1.0, 1.0], dtype=np.float32)
        self.ank_roll_indices = np.array(
            [
                robot.motor_ordering.index("left_ank_roll"),
                robot.motor_ordering.index("right_ank_roll"),
            ]
        )
        self.ank_roll_signs = np.array([-1.0, -1.0], dtype=np.float32)

        self.is_prepared = False
        self.prep_duration = 7.0

        self.time_start = self.prep_duration

        self.is_ready = False
        self.manip_duration = 0.0
        self.manip_motor_pos = self.default_motor_pos.copy()

    def reset(self):
        """Resets the state reference of the robot to its initial configuration.

        This method sets the first three elements of the state reference to zero, representing the initial position. It then sets the next four elements to represent the initial orientation as a quaternion. The method also updates the state reference with the current manipulator motor positions and their corresponding joint angles.
        """
        if self.state_ref is not None:
            self.state_ref[:3] = np.zeros(3, dtype=np.float32)
            self.state_ref[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.state_ref[13 : 13 + self.robot.nu] = self.manip_motor_pos.copy()
            self.state_ref[13 + self.robot.nu : 13 + 2 * self.robot.nu] = np.array(
                list(
                    self.robot.motor_to_joint_angles(
                        dict(zip(self.robot.motor_ordering, self.manip_motor_pos))
                    ).values()
                ),
                dtype=np.float32,
            )

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        """Generates a command array based on control inputs for various tasks.

        This function processes a dictionary of control inputs, mapping each task to a corresponding command value within a predefined range. It also manages the state of a button press to toggle logging.

        Args:
            control_inputs (Dict[str, float]): A dictionary where keys are task names and values are the corresponding input values.

        Returns:
            npt.NDArray[np.float32]: An array of command values, each corresponding to a task, with values adjusted according to the input and predefined command ranges.
        """
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
        """Retrieve the positions of the arm motors from the observation data.

        Args:
            obs (Obs): The observation data containing motor positions.

        Returns:
            npt.NDArray[np.float32]: An array of the arm motor positions.
        """
        return self.manip_motor_pos[self.arm_motor_indices]

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Processes an observation to compute the next action and control inputs for a robotic system.

        Args:
            obs (Obs): The current observation containing sensor data and time information.
            is_real (bool, optional): Indicates if the operation is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing control inputs and the target motor positions for the next step.
        """
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

            if hasattr(self, "task") and self.task == "hug":
                manip_motor_pos_1 = self.manip_motor_pos.copy()
                manip_motor_pos_1[self.left_sho_pitch_idx] = self.default_motor_pos[
                    self.left_sho_pitch_idx
                ]
                manip_motor_pos_1[self.right_sho_pitch_idx] = self.default_motor_pos[
                    self.right_sho_pitch_idx
                ]
                manip_motor_pos_1[self.left_sho_roll_idx] = -1.4
                manip_motor_pos_1[self.right_sho_roll_idx] = -1.4

                manip_time_1, manip_action_1 = self.move(
                    -self.control_dt,
                    self.default_motor_pos,
                    manip_motor_pos_1,
                    self.manip_duration / 2,
                )
                manip_time_2, manip_action_2 = self.move(
                    manip_time_1[-1],
                    manip_motor_pos_1,
                    self.manip_motor_pos,
                    self.manip_duration / 2,
                )
                self.manip_time = np.concatenate([manip_time_1, manip_time_2])
                self.manip_action = np.concatenate([manip_action_1, manip_action_2])
            else:
                self.manip_time, self.manip_action = self.move(
                    -self.control_dt,
                    self.default_motor_pos,
                    self.manip_motor_pos,
                    self.manip_duration,
                )

        if obs.time - self.time_start < self.manip_duration:
            action = np.asarray(
                interpolate_action(
                    obs.time - self.time_start, self.manip_time, self.manip_action
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
                    and msg.fsr is not None
                ):
                    gripper_pos = msg.fsr / 100 * self.motor_limits[-2:, 1]
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
            assert self.camera_frame is not None
            self.camera_time_list.append(time.time())
            self.camera_frame_list.append(self.camera_frame)
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
            current_roll = obs.euler[0].item()
            current_pitch = obs.euler[1].item()
            roll_error = self.desired_torso_roll - current_roll
            roll_vel = (current_roll - self.last_torso_roll) / self.control_dt
            pitch_error = self.desired_torso_pitch - current_pitch
            pitch_vel = (current_pitch - self.last_torso_pitch) / self.control_dt

            roll_pd_output = (
                self.torso_roll_kp * roll_error - self.torso_roll_kd * roll_vel
            )
            pitch_pd_output = (
                self.torso_pitch_kp * pitch_error - self.torso_roll_kd * pitch_vel
            )

            pd_output = np.array([roll_pd_output, pitch_pd_output, 0], dtype=np.float32)

            # Apply PD control based on torso pitch angle
            waist_roll, waist_yaw = self.robot.waist_fk(
                obs.motor_pos[self.waist_motor_indices]
            )
            waist_mat = euler2mat(np.array([waist_roll, 0.0, waist_yaw]))
            pd_output_rotated = waist_mat.T @ pd_output

            # print(f"waist_roll: {waist_roll:.2f}, waist_yaw: {waist_yaw:.2f}")
            # print(f"pd_output_rotated: {pd_output_rotated}")

            motor_target[self.hip_roll_indices] += (
                pd_output_rotated[0] * self.hip_roll_signs
            )
            motor_target[self.ank_roll_indices] += (
                pd_output_rotated[0] * self.ank_roll_signs
            )
            motor_target[self.hip_pitch_indices] += (
                pd_output_rotated[1] * self.hip_pitch_signs
            )

            self.last_torso_roll = current_roll
            self.last_torso_pitch = current_pitch

        # Override motor target with reference motion or teleop motion
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.step_curr += 1
        self.last_torso_pitch = current_pitch

        return control_inputs, motor_target
