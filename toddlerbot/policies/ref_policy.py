from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot

from toddlerbot.utils.math_utils import interpolate_action
from toddlerbot.utils.misc_utils import set_seed


class RefPolicy(BasePolicy):
    def __init__(self, robot: Robot, init_motor_pos: npt.NDArray[np.float32], ref_motion: str):
        super().__init__(robot, init_motor_pos)
        self.name = f"ref_policy_{ref_motion}"

        set_seed(0)

        prep_duration = 3.0
        warm_up_duration = 2.0
       
        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []
        self.time_mark_dict: Dict[str, float] = {}

        prep_time, prep_action = self.reset(
            -self.control_dt,
            init_motor_pos,
            np.zeros_like(init_motor_pos),
            prep_duration,
        )

        time_list.append(prep_time)
        action_list.append(prep_action)


        warm_up_act = np.zeros_like(init_motor_pos)
        # warm_up_angles={
        #     "left_sho_roll": -np.pi / 6,
        #     "right_sho_roll": -np.pi / 6,
        # }
        # for name, angle in warm_up_angles.items():
        #     warm_up_act[robot.joint_ordering.index(name)] = angle

        # if not np.allclose(warm_up_act, action_list[-1][-1], 1e-6):
        #     warm_up_time, warm_up_action = self.reset(
        #         time_list[-1][-1],
        #         action_list[-1][-1],
        #         warm_up_act,
        #         warm_up_duration,
        #     )

        #     time_list.append(warm_up_time)
        #     action_list.append(warm_up_action)
    
        ref_data = np.load(f"toddlerbot/ref_motion/{ref_motion}.npz", allow_pickle=True)
        ref_time = ref_data["time"] + time_list[-1][-1] + self.control_dt
        ref_action = ref_data["action"] + warm_up_act
        time_list.append(ref_time)
        action_list.append(ref_action)
        self.time_arr = np.concatenate(time_list)  # type: ignore
        self.action_arr = np.concatenate(action_list)  # type: ignore
        self.num_total_steps = len(self.time_arr)

    def step(self, obs: Obs) -> npt.NDArray[np.float32]:
        action = np.asarray(
            interpolate_action(obs.time, self.time_arr, self.action_arr)
        )
        return action
