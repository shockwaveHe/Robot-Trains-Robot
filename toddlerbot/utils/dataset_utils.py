import os
import shutil
from dataclasses import dataclass, fields
from typing import Optional

import joblib
import numpy as np
import numpy.typing as npt


@dataclass
class Data:
    """Data class for logging teleoperation data during an episode."""

    time: float
    action: npt.NDArray[np.float32]
    motor_pos: npt.NDArray[np.float32]
    image: Optional[npt.NDArray[np.uint8]] = None


class DatasetLogger:
    """A class for logging teleoperation data during an episode and saving it to disk."""

    def __init__(self):
        """Initializes the object with an empty data list and sets the episode count to zero."""
        self.data_list = []
        self.n_episodes = 0

    def log_entry(self, data: Data):
        """Adds a data entry to the internal list.

        Args:
            data (Data): The data entry to be added to the list.
        """
        self.data_list.append(data)

    def save(self):
        """Saves the current data list to a compressed file and resets the data list.

        This method converts the attributes of each data object in `self.data_list` into a dictionary of numpy arrays, adds a start time, and saves the dictionary to a file in LZ4 compressed format. The file is named using the current number of episodes. After saving, the episode count is incremented, and the data list is cleared.

        Attributes:
            data_list (list): A list of data objects to be saved.
            n_episodes (int): The current number of episodes logged.

        Side Effects:
            Increments `self.n_episodes`.
            Clears `self.data_list`.
            Prints a log message indicating the number of episodes logged and their length.
        """
        data_dict = {
            field.name: np.array(
                [getattr(data, field.name) for data in self.data_list],
            )
            for field in fields(Data)
        }
        data_dict["start_time"] = self.data_list[0].time

        # dump to lz4 format
        joblib.dump(data_dict, f"/tmp/toddlerbot_{self.n_episodes}.lz4", compress="lz4")

        self.n_episodes += 1

        print(
            f"\nLogged {self.n_episodes} episodes. Episode length: {len(self.data_list)}"
        )

        self.data_list = []

    def move_files_to_exp_folder(self, exp_folder_path: str):
        """Moves files with a specific naming pattern from the temporary directory to a specified experiment folder.

        Args:
            exp_folder_path (str): The destination directory path where the files will be moved.
        """
        lz4_files = [
            f
            for f in os.listdir("/tmp")
            if f.startswith("toddlerbot_") and f.endswith(".lz4")
        ]

        # Move each file to the exp_folder
        for file_name in lz4_files:
            source = os.path.join("/tmp", file_name)
            destination = os.path.join(exp_folder_path, file_name)
            shutil.move(source, destination)

        print(f"Moved {len(lz4_files)} files to {exp_folder_path}")
