from abc import ABC, abstractmethod


class BaseSim(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_joint_state(self):
        pass

    @abstractmethod
    def set_joint_angles(self, joint_angles: dict):
        pass

    @abstractmethod
    def close(self):
        pass
