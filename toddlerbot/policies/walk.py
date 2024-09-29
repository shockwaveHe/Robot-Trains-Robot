from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.walk_env import WalkCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


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
            env_cfg.action.cycle_time,
            env_cfg.sim.timestep * env_cfg.action.n_frames,
        )

        self.command_range = env_cfg.commands.command_range

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
            command = self.fixed_command
        else:
            control_inputs = self.joystick.get_controller_input()
            command = np.zeros_like(self.fixed_command)
            for task, input in control_inputs.items():
                if task == "walk_vertical":
                    command[0] = np.interp(
                        input,
                        [-1, 0, 1],
                        [self.command_range[0][1], 0.0, self.command_range[0][0]],
                    )

                elif task == "walk_horizontal":
                    command[1] = np.interp(
                        input,
                        [-1, 0, 1],
                        [self.command_range[1][1], 0.0, self.command_range[1][0]],
                    )

                elif task == "turn":
                    command[2] = input

        return command
