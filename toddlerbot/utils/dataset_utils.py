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
        self.episode_ends = []

    def log_entry(self, data: Data):
        self.data_list.append(data)

    # episode end index is the index of the last state entry in the episode +1
    def log_episode_end(self):
        self.episode_ends.append(len(self.data_list))

    def maintain_log(self):
        if len(self.episode_ends) > 0:
            len_dataset = self.episode_ends[-1]
            self.data_list = self.data_list[:len_dataset]

    # TODO: Implement this function
    def _set_to_rate(self, rate: float):
        # state_array = self.data_dict["state_array"]
        # images = self.data_dict["images"]

        # ds_state = []
        # ds_images = []
        # for i in range(0, len(state_array)):
        #     if state
        #         ds_state.append(state_array[i])
        #         ds_images.append(images[i])

        # self.data_dict["state_array"] = state_array.tolist()
        # self.data_dict["images"] = images.tolist()
        pass

    def save(self, path: str):
        # watchout for saving time in float32, it will get truncated to 100s accuracy
        # Assuming self.data_list is a list of Data instances
        data_dict = {
            field.name: np.array(
                [getattr(data, field.name) for data in self.data_list],
            )
            for field in fields(Data)
        }
        data_dict["start_time"] = self.data_list[0].time

        # downsample everything to 10hz
        self._set_to_rate(10)

        # dump to lz4 format
        joblib.dump(data_dict, path, compress="lz4")
