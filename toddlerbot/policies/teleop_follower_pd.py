import time

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQNode


class TeleopFollowerPDPolicy(BalancePDPolicy, policy_name="teleop_follower_pd"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
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

        self.zmq_node = ZMQNode(type="receiver")
        self.last_arm_joint_pos = self.default_joint_pos[self.arm_joint_slice].copy()

    def reset(self):
        super().reset()
        self.last_arm_joint_pos = self.default_joint_pos[self.arm_joint_slice].copy()

    def get_joint_target(self, obs: Obs, time_curr: float) -> npt.NDArray[np.float32]:
        # Still override even if no message received, so that it won't suddenly go to default pose
        # TODO: Maintain the squatting pose
        joint_target = self.default_joint_pos.copy()
        joint_angles = self.robot.motor_to_joint_angles(
            dict(zip(self.robot.motor_ordering, obs.motor_pos))
        )
        joint_target[self.neck_yaw_idx] = joint_angles["neck_yaw_driven"]
        joint_target[self.neck_pitch_idx] = joint_angles["neck_pitch_driven"]
        joint_target[self.arm_joint_slice] = self.last_arm_joint_pos
        # Get the motor target from the teleop node
        msg = self.zmq_node.get_msg()
        if msg is not None:
            if abs(time.time() - msg.time) < 0.1:
                arm_motor_pos = msg.action
                arm_joint_pos = arm_motor_pos * self.arm_gear_ratio
                joint_target[self.arm_joint_slice] = arm_joint_pos
                self.last_arm_joint_pos = arm_joint_pos
            else:
                print("stale message received, discarding")

        return joint_target
