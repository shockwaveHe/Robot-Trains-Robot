import platform
import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.ref_motion.balance_pd_ref import BalancePDReference
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode

SYS_NAME = platform.system()


class TeleopFollowerPDPolicy(BalancePDPolicy, policy_name="teleop_follower_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
        camera: Optional[Camera] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        squat_speed=0.03,
    ):
        super().__init__(name, robot, init_motor_pos)

        self.arm_joint_slice = slice(
            robot.joint_ordering.index("left_sho_pitch"),
            robot.joint_ordering.index("right_wrist_roll") + 1,
        )

        arm_gear_ratio_list = []
        for i, motor_name in enumerate(robot.motor_ordering):
            if robot.joint_groups[motor_name] == "arm":
                motor_config = robot.config["joints"][motor_name]
                if (
                    motor_config["transmission"] == "gear"
                    or motor_config["transmission"] == "rack_and_pinion"
                ):
                    arm_gear_ratio_list.append(-motor_config["gear_ratio"])
                else:
                    arm_gear_ratio_list.append(1.0)

        self.arm_gear_ratio = np.array(arm_gear_ratio_list, dtype=np.float32)

        self.neck_yaw_idx = robot.joint_ordering.index("neck_yaw_driven")
        self.neck_pitch_idx = robot.joint_ordering.index("neck_pitch_driven")
        self.neck_yaw_limits = robot.joint_limits["neck_yaw_driven"]
        self.neck_pitch_limits = robot.joint_limits["neck_pitch_driven"]

        self.neck_yaw_target = 0.0
        self.neck_pitch_target = 0.0

        self.squat_speed = squat_speed
        self.balance_ref = BalancePDReference(robot, self.control_dt)

        self.joystick = joystick
        if joystick is None:
            try:
                self.joystick = Joystick()
            except Exception:
                pass

        # Initialize sensors

        self.zmq_receiver = None
        if zmq_receiver is not None:
            self.zmq_receiver = zmq_receiver
        elif SYS_NAME != "Darwin":
            self.zmq_receiver = ZMQNode(type="receiver")

        self.zmq_sender = None
        if zmq_sender is not None:
            self.zmq_sender = zmq_sender
        elif SYS_NAME != "Darwin":
            self.zmq_sender = ZMQNode(type="sender")

        self.camera = None
        if camera is not None:
            self.camera = camera
        elif SYS_NAME != "Darwin":
            try:
                self.camera = Camera()
                self.zmq_sender = ZMQNode(type="sender", ip="192.168.46")
            except Exception:
                pass

        self.msg = None
        self.last_control_inputs = None
        self.last_arm_joint_pos = self.default_joint_pos[self.arm_joint_slice].copy()

    def reset(self):
        super().reset()
        self.neck_yaw_target = 0.0
        self.neck_pitch_target = 0.0
        self.last_arm_joint_pos = self.default_joint_pos[self.arm_joint_slice].copy()

    def plan(self) -> npt.NDArray[np.float32]:
        # Get the motor target from the teleop node
        msg = None
        if self.msg is not None:
            msg = self.msg
        elif self.zmq_receiver is not None:
            msg = self.zmq_receiver.get_msg()

        # print(f"msg: {msg}")

        control_inputs = self.last_control_inputs
        if self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()
        elif msg is not None:
            control_inputs = msg.control_inputs

        self.last_control_inputs = control_inputs

        look_command = np.zeros(2, dtype=np.float32)
        squat_command = np.zeros(1, dtype=np.float32)

        if control_inputs is not None:
            for task, input in control_inputs.items():
                if task == "look_up" and input > 0:
                    look_command[1] = input
                elif task == "look_down" and input > 0:
                    look_command[1] = -input
                elif task == "look_left" and input > 0:
                    look_command[0] = input
                elif task == "look_right" and input > 0:
                    look_command[0] = -input
                if task == "squat":
                    squat_command[0] = -input * self.squat_speed

        time_curr = self.step_curr * self.control_dt
        state_ref = self.balance_ref.get_state_ref(
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            time_curr,
            squat_command,
        )

        joint_target = np.asarray(state_ref[13 : 13 + self.robot.nu])

        self.neck_yaw_target = np.clip(
            self.neck_yaw_target + look_command[0] * self.control_dt,
            self.neck_yaw_limits[0],
            self.neck_yaw_limits[1],
        )
        self.neck_pitch_target = np.clip(
            self.neck_pitch_target + look_command[1] * self.control_dt,
            self.neck_pitch_limits[0],
            self.neck_pitch_limits[1],
        )
        joint_target[self.neck_yaw_idx] = self.neck_yaw_target
        joint_target[self.neck_pitch_idx] = self.neck_pitch_target

        joint_target[self.arm_joint_slice] = self.last_arm_joint_pos

        if msg is not None:
            # print(f"latency: {abs(time.time() - msg.time) * 1000:.2f} ms")
            if abs(time.time() - msg.time) < 0.1:
                arm_motor_pos = msg.action
                arm_joint_pos = arm_motor_pos * self.arm_gear_ratio
                joint_target[self.arm_joint_slice] = arm_joint_pos
                self.last_arm_joint_pos = arm_joint_pos
            else:
                print("stale message received, discarding")

        return joint_target

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if self.camera is not None:
            t1 = time.time()
            jpeg_frame, raw_frame = self.camera.get_jpeg()
            camera_frame = jpeg_frame
            t2 = time.time()
            print(raw_frame.shape)
        else:
            camera_frame = None

        if self.zmq_sender is not None:
            self.zmq_sender.send_msg(
                {"time": time.time(), "camera_frame": camera_frame}
            )

        return super().step(obs, is_real)
