from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple


class BaseController(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def connect_to_client(self):
        pass

    @abstractmethod
    def initialize_motors(self):
        pass

    @abstractmethod
    def set_pos(self, pos):
        pass

    @abstractmethod
    def read_state(self):
        pass

    @abstractmethod
    def close_motors(self):
        pass

    # @abstractmethod
    # def set_vel(self, vel):
    #     pass
