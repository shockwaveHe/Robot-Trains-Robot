from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
import numpy.typing as npt

import pkgutil
import importlib
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate
from toddlerbot.utils.misc_utils import snake2camel, log

# Global registry to store policy names and their corresponding classes
policy_registry: Dict[str, Type["BasePolicy"]] = {}


def get_policy_class(policy_name: str) -> Type["BasePolicy"]:
    """Retrieves the policy class associated with the given policy name.

    Args:
        policy_name (str): The name of the policy to retrieve.

    Returns:
        Type[BasePolicy]: The class of the policy corresponding to the given name.

    Raises:
        ValueError: If the policy name is not found in the policy registry.
    """
    if policy_name not in policy_registry:
        raise ValueError(f"Unknown policy: {policy_name}")

    return policy_registry[policy_name]


def get_policy_names() -> List[str]:
    """Retrieves a list of policy names from the policy registry.

    This function iterates over the keys in the policy registry and generates a list
    of policy names. For each key, it adds the key itself and a modified version of
    the key with the suffix '_fixed' to the list.

    Returns:
        List[str]: A list containing the original and modified policy names.
    """
    policy_names: List[str] = []
    for key in policy_registry.keys():
        policy_names.append(key)
        policy_names.append(key + "_fixed")

    return policy_names


def dynamic_import_policies(policy_package: str):
    """Dynamically imports all modules within a specified package.

    This function attempts to import each module found in the given package directory. If a module cannot be imported, a log message is generated.

    Args:
        policy_package (str): The name of the package containing the modules to be imported.
    """
    package = importlib.import_module(policy_package)
    package_path = package.__path__

    # Iterate over all modules in the given package directory
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{policy_package}.{module_name}"
        try:
            importlib.import_module(full_module_name)
        except Exception:
            log(f"Could not import {full_module_name}", header="Dynamic Import")


class BasePolicy(ABC):
    """Base class for all policies."""

    @abstractmethod
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        init_arm_pos: npt.NDArray[np.float32] = None,
        control_dt: float = 0.02,
        prep_duration: float = 2.0,
        n_steps_total: float = float("inf"),
        exp_folder: str = "",
    ):
        """Initializes the class with robot configuration and control parameters.

        Args:
            name (str): The name of the robot or component.
            robot (Robot): An instance of the Robot class containing robot specifications.
            init_motor_pos (npt.NDArray[np.float32]): Initial positions of the robot's motors.
            control_dt (float, optional): Time interval for control updates. Defaults to 0.02.
            prep_duration (float, optional): Duration for preparation phase. Defaults to 2.0.
            n_steps_total (float, optional): Total number of control steps. Defaults to infinity.
        """
        self.name = name
        self.robot = robot
        self.exp_folder = exp_folder
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
        """Initializes a subclass and registers it with a policy name.

        Args:
            policy_name (str): The name of the policy to register the subclass under. If not provided, the subclass will not be registered.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
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
    
    def is_truncated(self) -> bool:
        return False

    def reset(self) -> Obs:
        pass
    
    def close(self) -> None:
        pass
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
        """Calculates the trajectory of an action over a specified duration, interpolating between current and next actions.

        Args:
            time_curr (float): The current time from which the trajectory starts.
            action_curr (npt.NDArray[np.float32]): The current action state as a NumPy array.
            action_next (npt.NDArray[np.float32]): The next action state as a NumPy array.
            duration (float): The total duration over which the action should be interpolated.
            end_time (float, optional): The time at the end of the duration where the action should remain constant. Defaults to 0.0.

        Returns:
            Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: A tuple containing the time steps and the corresponding interpolated positions.
        """
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
