from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.walk_env import WalkCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.teleop.joystick import Joystick


class WalkPolicy(MJXPolicy, policy_name="walk"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = WalkCfg()
        motion_ref = WalkZMPReference(
            robot,
            env_cfg.commands.command_list,
            env_cfg.action.cycle_time,
            env_cfg.sim.timestep * env_cfg.action.n_frames,
        )

        self.joystick = None
        try:
            self.joystick = Joystick()
        except Exception:
            pass

        super().__init__(
            name, robot, init_motor_pos, ckpt, fixed_command, env_cfg, motion_ref
        )

    def get_command(self) -> npt.NDArray[np.float32]:
        if self.joystick is None:
            command_arr = self.fixed_command
        else:
            task_commands = self.joystick.get_controller_input()
            command_arr = np.zeros(2, dtype=np.float32)
            for task, command in task_commands.items():
                if task == "walk_vertical":
                    command_arr[0] = command
                elif task == "walk_horizontal":
                    command_arr[1] = command

        return command_arr
