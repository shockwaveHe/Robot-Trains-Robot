from abc import ABC, abstractmethod
from typing import Optional

import numpy

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class MotionReference(ABC):
    def __init__(self, name: str, motion_type: str, robot: Robot):
        self.name = name
        self.motion_type = motion_type
        self.robot = robot

        self.default_joint_pos = np.array(  # type: ignore
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_joint_vel = np.zeros_like(self.default_joint_pos)  # type: ignore

        self.default_motor_pos = np.array(  # type: ignore
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        motor_indices = np.arange(robot.nu)  # type:ignore
        motor_groups = numpy.array(  # type:ignore
            [robot.joint_groups[name] for name in robot.motor_ordering]
        )
        self.leg_motor_indices = motor_indices[motor_groups == "leg"]
        self.arm_motor_indices = motor_indices[motor_groups == "arm"]
        self.neck_motor_indices = motor_indices[motor_groups == "neck"]
        self.waist_motor_indices = motor_indices[motor_groups == "waist"]

    @abstractmethod
    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        pass

    @abstractmethod
    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        time_curr: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> ArrayType:
        pass

    @abstractmethod
    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        pass
