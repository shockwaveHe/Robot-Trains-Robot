from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple


class BaseController(ABC):
    @abstractmethod
    def __init__(self, port):
        pass

    @abstractmethod
    def send_command(self, command):
        pass
