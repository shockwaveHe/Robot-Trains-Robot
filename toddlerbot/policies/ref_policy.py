from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot

from toddlerbot.utils.math_utils import interpolate_action
from toddlerbot.utils.misc_utils import set_seed


class RefPolicy(BasePolicy, policy_name="ref"):
    def __init__(self, name:str, robot: Robot, init_motor_pos: npt.NDArray[np.float32], ref_motion: str):
        super().__init__(f"{name}_{ref_motion}", robot, init_motor_pos)
        set_seed(0)

        prep_duration = 3.0
        warm_up_duration = 2.0
       
        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []
        self.time_mark_dict: Dict[str, float] = {}

        # prep_act = np.array(list(robot.init_motor_angles.values()), dtype=np.float32)
        # prep_time, prep_action = self.move(-self.control_dt,
        #     init_motor_pos, prep_act, prep_duration)
        # time_list.append(prep_time)
        # action_list.append(prep_action)


        warm_up_act = np.zeros_like(init_motor_pos)
        warm_up_angles={
            "left_sho_roll": -np.pi / 6,
            "right_sho_roll": -np.pi / 6,
        }
        for name, angle in warm_up_angles.items():
            warm_up_act[robot.joint_ordering.index(name)] = angle

        # if not np.allclose(warm_up_act, action_list[-1][-1], 1e-6):
        #     warm_up_time, warm_up_action = self.move(time_list[-1][-1],
        #             action_list[-1][-1], warm_up_act, warm_up_duration)

        #     time_list.append(warm_up_time)
        #     action_list.append(warm_up_action)
    
        ref_data = np.load(f"toddlerbot/ref_motion/{ref_motion}.npz", allow_pickle=True)
        ref_time = ref_data["time"] + self.control_dt
        ref_action = ref_data["action"]
        time_list.append(ref_time)
        action_list.append(ref_action)
        self.time_arr = np.concatenate(time_list)  # type: ignore
        self.action_arr = np.concatenate(action_list)  # type: ignore
        self.n_steps_total = len(self.time_arr)
        self.step_curr = 0

        self.torso_pos = None
        self.torso_pos = np.array([
            0.0, 0.0, 0.15, 1.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17, -1.41, 1.4, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17, 1.41, -1.4, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.42, 0.0, 0.0, -1.57, 1.46, 0.0, -0.0, -0.32, 0.0, 0.0, 1.42, 0.0, 0.0, 1.57, 1.46, 0.0, 0.0, 0.32, 0.0, -0.0, 
        ], dtype=np.float32)

    def step(self, obs: Obs, is_real: bool=False) -> npt.NDArray[np.float32]:
        action = np.asarray(
            interpolate_action(obs.time, self.time_arr, self.action_arr)
        )
        self.step_curr += 1
        return action
