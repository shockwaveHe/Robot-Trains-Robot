"""
This script processes the raw data collected in teleop to create a dataset for dp training.
Raw dataset entry:
state_array: [time(1), motor_angles(14), fsrL(1), fsrR(1), camera_frame_idx(1)]

It creates an output dataset of the form:
image: (N, 96, 96,3) - RGB
episode_ends: (N,) 
state: (N, ns)
action: (N, na)
"""

import matplotlib.pyplot as plt
import joblib
import numpy as np
import cv2


result_dir = "/Users/weizhuo2/Documents/gits/toddleroid/results/"
dataset_path = "toddlerbot_arms_teleop_fixed_mujoco_20240909_204445/dataset.lz4"
raw_data = joblib.load(result_dir + dataset_path)
output_dataset = dict()
print(raw_data.keys())

# convert from 30hz to 10hz
low_freq_state_array = np.array([])
low_freq_epi_ends = []
last_epi_end_idx = 0
for epi_end_idx in raw_data['episode_ends']:
    epi_state_array = raw_data['state_array'][last_epi_end_idx:epi_end_idx]
    epi_state_array = epi_state_array[::3]
    low_freq_state_array = np.vstack([low_freq_state_array, epi_state_array]) if low_freq_state_array.size else epi_state_array
    low_freq_epi_ends.append(low_freq_state_array.shape[0])
    last_epi_end_idx = epi_end_idx

raw_data['state_array'] = low_freq_state_array
raw_data['episode_ends'] = low_freq_epi_ends
raw_data['images'] = raw_data['images'][low_freq_state_array[:, -1].astype(int)]

# convert images to 171x96 resolution
resized_images = [cv2.resize(image, (171,96)) for image in raw_data['images']]
output_dataset['images'] = np.array(resized_images, dtype=np.float32)[:,:96,38:134]

# assign state and action
output_dataset['agent_pos'] = raw_data['state_array'][:, 1:17].astype(np.float32)
output_dataset['action'] = []
last_idx = 0
offset = 2
for idx in raw_data['episode_ends']:
    shifted_state = raw_data['state_array'][last_idx+offset:idx, 1:17]
    repeated_last_rows = np.tile(shifted_state[-1], (offset, 1))  # Create n copies of the last row
    shifted_state = np.vstack([shifted_state, repeated_last_rows])
    output_dataset['action'].append(shifted_state)
    last_idx = idx
output_dataset['action'] = np.vstack(output_dataset['action']).astype(np.float32)
# output_dataset['action'] = raw_data['state_array'][:, 1:17]
output_dataset['episode_ends'] = raw_data['episode_ends']

# save the dataset
output_path = "/Users/weizhuo2/Documents/gits/diffusion_policy_minimal/teleop_data/teleop_dataset.lz4"
joblib.dump(output_dataset, output_path)