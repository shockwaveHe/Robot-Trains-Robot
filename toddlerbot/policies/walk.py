from typing import Dict, Optional

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
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = WalkCfg()
        motion_ref = WalkZMPReference(
            robot,
            env_cfg.action.cycle_time,
            env_cfg.sim.timestep * env_cfg.action.n_frames,
        )

        self.command_range = env_cfg.commands.command_range

        self.joystick = joystick
        if joystick is None:
            try:
                self.joystick = Joystick()
            except Exception:
                pass

        super().__init__(
            name,
            robot,
            init_motor_pos,
            ckpt,
            joystick,
            fixed_command,
            env_cfg,
            motion_ref,
        )

    def get_command(
        self, control_inputs: Optional[Dict[str, float]] = None
    ) -> npt.NDArray[np.float32]:
        # TODO: Remove the fixed command
        command = np.zeros_like(self.fixed_command)
        if control_inputs is None:
            control_inputs = self.joystick.get_controller_input()

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

        return command
