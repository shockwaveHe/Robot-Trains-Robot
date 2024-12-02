from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.algorithms.apf import APF
from toddlerbot.policies.walk import WalkPolicy
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class WalkAPFPolicy(WalkPolicy, policy_name="walk_apf"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str = "",
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(name, robot, init_motor_pos, ckpt, joystick, fixed_command)

        self.control_inputs = {"table_x": 1.0, "table_y": 0.0}

        # Table boundaries (rectangle)
        table_pos = [0.5, 0.0, 0.08]
        table_size = [0.15, 0.26, 0.16]
        table_bounds = [
            table_pos[0] - table_size[0] / 2,
            table_pos[1] - table_size[1] / 2,
            table_pos[0] + table_size[0] / 2,
            table_pos[1] + table_size[1] / 2,
        ]
        # Create an instance of APF
        self.apf = APF(table_bounds, dt=self.control_dt)

        # Start and goal positions
        x_start, y_start = 0.0, 0.0
        x_goal, y_goal = self.control_inputs["table_x"], self.control_inputs["table_y"]

        # Plan the path
        self.apf.plan_path(x_start, y_start, x_goal, y_goal)

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        command = np.array(self.apf.velocities[self.step_curr], dtype=np.float32)
        return command
