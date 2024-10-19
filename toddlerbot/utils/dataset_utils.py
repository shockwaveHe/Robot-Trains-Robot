import os
import shutil
from dataclasses import dataclass, fields
from typing import Optional

import joblib
import numpy as np
import numpy.typing as npt

"""
Dataset format:
Dict: {"state_array":[n,1+16+1], "images":[n,h,w,3]}
state_array: [time(1), joint_angles(14), fsrL(1), fsrR(1), camera_frame_idx(1)]
images: [n,h,w,3], RGB images uint8
"""


@dataclass
class Data:
    time: float
    arm_motor_pos: npt.NDArray[np.float32]
    fsr_data: npt.NDArray[np.float32]
    image: Optional[npt.NDArray[np.uint8]] = None


class DatasetLogger:
    def __init__(self):
        self.data_list = []
        self.n_episodes = 0

    def log_entry(self, data: Data):
        self.data_list.append(data)

    # episode end index is the index of the last state entry in the episode +1
    def save(self):
        # watchout for saving time in float32, it will get truncated to 100s accuracy
        # Assuming self.data_list is a list of Data instances
        print(
            f"\nLogged {self.n_episodes} episodes. Episode length: {len(self.data_list)}"
        )
        data_dict = {
            field.name: np.array(
                [getattr(data, field.name) for data in self.data_list],
            )
            for field in fields(Data)
        }
        data_dict["start_time"] = self.data_list[0].time

        # dump to lz4 format
        joblib.dump(data_dict, f"/tmp/toddlerbot_{self.n_episodes}.lz4", compress="lz4")

        self.data_list = []
        self.n_episodes += 1

    def move_files_to_exp_folder(self, exp_folder_path: str):
        # Find all files that match the pattern
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
