import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy, default_pose
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQNode


class TeleopFollowerPDPolicy(BalancePDPolicy, policy_name="teleop_follower_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(name, robot, init_motor_pos, fixed_command)

        self.zmq_node = ZMQNode(type="receiver")
        self.last_pose = default_pose

    def override_motor_target(self, motor_target):
        # Still override even if no message received, so that it won't suddenly go to default pose
        motor_target[16:30] = self.last_pose
        # Get the motor target from the teleop node
        remote_state_dict = self.zmq_node.get_msg()
        if remote_state_dict is not None:
            # print(remote_state_dict["time"], time.time())
            # print(remote_state_dict["sim_action"])
            # print((time.time() - remote_state_dict["time"]) * 1000)
            # print("last ", remote_state_dict["test"])
            if abs(time.time() - remote_state_dict["time"]) < 0.1:
                motor_target[16:30] = remote_state_dict["sim_action"]
                self.last_pose = remote_state_dict["sim_action"]
            else:
                print("stale message received, discarding")

        return motor_target
