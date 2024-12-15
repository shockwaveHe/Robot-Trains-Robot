from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
import numpy.typing as npt

from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate
from toddlerbot.utils.misc_utils import snake2camel

# Global registry to store policy names and their corresponding classes
policy_registry: Dict[str, Type["BasePolicy"]] = {}


def get_policy_class(policy_name: str) -> Type["BasePolicy"]:
    if policy_name not in policy_registry:
        raise ValueError(f"Unknown policy: {policy_name}")

    return policy_registry[policy_name]


def get_policy_names() -> List[str]:
    policy_names: List[str] = []
    for key in policy_registry.keys():
        policy_names.append(key)
        policy_names.append(key + "_fixed")

    return policy_names


class BasePolicy(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        control_dt: float = 0.02,
        prep_duration: float = 2.0,
        n_steps_total: float = float("inf"),
    ):
        self.name = name
        self.robot = robot
        self.init_motor_pos = init_motor_pos
        self.control_dt = control_dt
        self.prep_duration = prep_duration
        self.n_steps_total = n_steps_total

        self.header_name = snake2camel(name)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.motor_limits = np.array(
            [robot.joint_limits[name] for name in robot.motor_ordering]
        )
        indices = np.arange(robot.nu)
        motor_groups = np.array(
            [robot.joint_groups[name] for name in robot.motor_ordering]
        )
        joint_groups = np.array(
            [robot.joint_groups[name] for name in robot.joint_ordering]
        )
        self.leg_motor_indices = indices[motor_groups == "leg"]
        self.leg_joint_indices = indices[joint_groups == "leg"]
        self.arm_motor_indices = indices[motor_groups == "arm"]
        self.arm_joint_indices = indices[joint_groups == "arm"]
        self.neck_motor_indices = indices[motor_groups == "neck"]
        self.neck_joint_indices = indices[joint_groups == "neck"]
        self.waist_motor_indices = indices[motor_groups == "waist"]
        self.waist_joint_indices = indices[joint_groups == "waist"]

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt, init_motor_pos, self.default_motor_pos, self.prep_duration
        )

    # Automatic registration of subclasses
    def __init_subclass__(cls, policy_name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if len(policy_name) > 0:
            policy_registry[policy_name] = cls

    @abstractmethod
    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        pass

    def is_done(self, obs: Obs) -> bool:
        return False
    
    # duration: total length of the motion
    # end_time: when motion should end, end time < time < duration will be static
    def move(
        self,
        time_curr: float,
        action_curr: npt.NDArray[np.float32],
        action_next: npt.NDArray[np.float32],
        duration: float,
        end_time: float = 0.0,
    ):
        reset_time = np.linspace(
            0,
            duration,
            int(duration / self.control_dt),
            endpoint=False,
            dtype=np.float32,
        )

        reset_pos = np.zeros((len(reset_time), action_curr.shape[0]), dtype=np.float32)
        for i, t in enumerate(reset_time):
            if t < duration - end_time:
                pos = interpolate(
                    action_curr,
                    action_next,
                    duration - end_time,
                    t,
                )
            else:
                pos = action_next

            reset_pos[i] = pos

        reset_time += time_curr + self.control_dt

        return reset_time, reset_pos
