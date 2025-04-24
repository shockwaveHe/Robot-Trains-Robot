import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from toddlerbot.actuation import JointState
from toddlerbot.actuation.dynamixel_control import DynamixelConfig, DynamixelController


class DynamixelMCHController:
    def __init__(self, controller_configs: List[Tuple[DynamixelConfig, List[int]]]):
        """
        Initializes multiple DynamixelController instances.

        Args:
            controller_configs (List[Tuple[DynamixelConfig, List[int]]]):
                A list where each element is a tuple of (DynamixelConfig, motor_ids)
                for one controller.
        """
        self.executor = ThreadPoolExecutor()

        future_controllers = [
            self.executor.submit(DynamixelController, config, motor_ids)
            for i, (config, motor_ids) in enumerate(controller_configs[:4])
        ]
        self.controllers: List[DynamixelController] = []
        for future in as_completed(future_controllers):
            for f in future_controllers:
                if f is future:
                    # end_time = time.time()
                    self.controllers.append(future.result())
                    # log(f"Time taken for {key}: {end_time - start_times[key]}", header=snake2camel(self.name), level="debug")
                    break

    def get_motor_state(self) -> Dict[int, JointState]:
        """
        Queries each controller concurrently and aggregates the results.
        Returns:
            Dict[int, JointState]: A dictionary of motor states keyed by motor ID.
        """
        aggregated_state = {}
        start_time = time.time()
        futures = [
            self.executor.submit(controller.get_motor_state)
            for controller in self.controllers
        ]
        for future in as_completed(futures):
            try:
                state = future.result()
                aggregated_state.update(state)
                end_time = time.time()
                print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
            except Exception as e:
                print(f"Error retrieving motor state: {e}")

        return aggregated_state

    def close_motors(self):
        """Closes all controllers."""
        self.controllers[0].close_motors()
