import joblib
import numpy as np

"""
Dataset format:
Dict: {"state_array":[n,1+16+1], "images":[n,h,w,3]}
state_array: [time(1), joint_angles(14), fsrL(1), fsrR(1), camera_frame_idx(1)]
images: [n,h,w,3], RGB images uint8
"""


class DatasetLogger:
    def __init__(self):
        self.data_dict = {"state_array": [], "images": [], "episode_ends": []}

    def log_entry(self, time, joint_angles, fsr_data, camera_frame):
        camera_frame_idx = len(self.data_dict["images"])
        state_entry = [time] + list(joint_angles) + fsr_data + [camera_frame_idx]

        self.data_dict["state_array"].append(state_entry)
        if camera_frame is not None:
            self.data_dict["images"].append(camera_frame)
        else:
            self.data_dict["images"].append(np.zeros((1, 1, 3), dtype=np.uint8))

    # episode end index is the index of the last state entry in the episode +1
    def log_episode_end(self):
        self.data_dict["episode_ends"].append(len(self.data_dict["state_array"]))

    def maintain_log(self):
        if len(self.data_dict["episode_ends"]) > 0:
            len_dataset = self.data_dict["episode_ends"][-1]
            self.data_dict["state_array"] = self.data_dict["state_array"][:len_dataset]
            self.data_dict["images"] = self.data_dict["images"][:len_dataset]

    # TODO: Implement this function
    def _set_to_rate(self, rate):
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
        self.data_dict["state_array"] = np.array(self.data_dict["state_array"])
        self.data_dict["start_time"] = self.data_dict["state_array"][0, 0]

        # convert list to np array, image to uint8 to save space (4x)
        self.data_dict["images"] = np.array(self.data_dict["images"], dtype=np.uint8)

        # downsample everything to 10hz
        self._set_to_rate(10)

        # dump to lz4 format
        joblib.dump(self.data_dict, path, compress="lz4")
