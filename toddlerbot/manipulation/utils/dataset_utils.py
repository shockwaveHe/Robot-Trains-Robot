import os

import cv2
import numpy as np


def create_video_grid(
    image_data: np.ndarray,
    episode_ends: np.ndarray,
    save_path: str,
    file_name: str,
    num_cols: int = 10,
    fps: int = 10,
):
    """Creates a grid video from a sequence of image data, organizing episodes into a specified number of columns and saving the result to a file.

    Args:
        image_data (np.ndarray): A 4D numpy array of shape (T, C, H, W) containing the image data for all episodes, where T is the total number of frames, C is the number of channels, H is the height, and W is the width.
        episode_ends (np.ndarray): A 1D numpy array indicating the end frame indices for each episode.
        save_path (str): The directory path where the video file will be saved.
        file_name (str): The name of the video file to be created.
        num_cols (int, optional): The number of columns in the video grid. Defaults to 10.
        fps (int, optional): The frames per second for the output video. Defaults to 10.
    """
    video_path = os.path.join(save_path, file_name)

    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    episode_list = []
    for e_idx in range(len(episode_ends)):
        start_idx = episode_starts[e_idx]
        end_idx = episode_ends[e_idx]
        # Extract the joint trajectory for this episode
        episode_list.append(image_data[start_idx:end_idx])

    episode_lengths = [epi.shape[0] for epi in episode_list]
    max_length = max(episode_lengths)

    E = len(episode_list)
    num_rows = (E + num_cols - 1) // num_cols  # ceil division

    C, H, W = image_data.shape[1:]
    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(video_path, fourcc, fps, (W * num_cols, H * num_rows))

    for t in range(max_length):
        # Create a canvas for this frame (in RGB)
        big_frame = np.zeros((H * num_rows, W * num_cols, C), dtype=np.uint8)

        for e_idx in range(E):
            row = e_idx // num_cols
            col = e_idx % num_cols

            # If t is beyond the episode length, use the last frame
            if t < episode_list[e_idx].shape[0]:
                frame_rgb = episode_list[e_idx][t].copy()  # shape (3, H, W)
            else:
                # Use the last frame of the episode
                frame_rgb = episode_list[e_idx][-1].copy()

            # Transpose to (H, W, C)
            frame = np.transpose(frame_rgb, (1, 2, 0))

            # Compute where to place this frame
            start_y = row * H
            start_x = col * W

            big_frame[start_y : start_y + H, start_x : start_x + W] = frame

        # Convert RGB to BGR for OpenCV
        big_frame_bgr = cv2.cvtColor(big_frame, cv2.COLOR_RGB2BGR)
        out.write(big_frame_bgr)

    out.release()
    print(f"Saved grid video to {video_path}")


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    """Generates sample indices for sequences within episodes, considering padding.

    Args:
        episode_ends (np.ndarray): An array of indices indicating the end of each episode.
        sequence_length (int): The length of the sequence to sample.
        pad_before (int, optional): Number of padding steps to allow before the start of a sequence. Defaults to 0.
        pad_after (int, optional): Number of padding steps to allow after the end of a sequence. Defaults to 0.

    Returns:
        np.ndarray: An array of shape (n, 4) where each row contains the start and end indices of the buffer and the start and end indices of the sample within the sequence.
    """
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices_arr = np.array(indices)
    return indices_arr


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    """Generates a sequence sample from the given training data with specified indices and sequence length.

    Args:
        train_data (dict): A dictionary where keys are identifiers and values are numpy arrays representing the data.
        sequence_length (int): The desired length of the output sequence.
        buffer_start_idx (int): The starting index for slicing the input arrays.
        buffer_end_idx (int): The ending index for slicing the input arrays.
        sample_start_idx (int): The starting index for placing the sliced data in the output sequence.
        sample_end_idx (int): The ending index for placing the sliced data in the output sequence.

    Returns:
        dict: A dictionary with the same keys as `train_data`, where each value is a numpy array of the specified sequence length, containing the sampled data.
    """
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    """Calculates the minimum and maximum values for each feature in the given dataset.

    Args:
        data (np.ndarray): A multi-dimensional numpy array representing the dataset.

    Returns:
        dict: A dictionary containing the minimum and maximum values for each feature, with keys 'min' and 'max'.
    """
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    """Normalizes the input data to a range of [-1, 1] based on provided statistics.

    This function adjusts the input data by first normalizing it to a [0, 1] range using the minimum and maximum values from the `stats` dictionary. It then scales the normalized data to a [-1, 1] range. Indices where the range is zero are set to zero in the output.

    Args:
        data (numpy.ndarray): The input data to be normalized.
        stats (dict): A dictionary containing 'min' and 'max' keys with corresponding numpy arrays for minimum and maximum values.

    Returns:
        numpy.ndarray: The normalized data with values scaled to the range [-1, 1].
    """
    # Calculate the range and create a mask where the range is zero
    range_vals = stats["max"] - stats["min"]
    zero_range_mask = range_vals == 0

    # Initialize normalized data with zeros for indices with zero range
    ndata = np.zeros_like(data)

    # Normalize to [0, 1] at non-zero range indices
    ndata[:, ~zero_range_mask] = (
        data[:, ~zero_range_mask] - stats["min"][~zero_range_mask]
    ) / range_vals[~zero_range_mask]

    # Scale to [-1, 1] only at non-zero range indices
    ndata[:, ~zero_range_mask] = ndata[:, ~zero_range_mask] * 2 - 1

    return ndata


def unnormalize_data(ndata, stats):
    """Converts normalized data back to its original scale using provided statistics.

    Args:
        ndata (float or array-like): The normalized data, assumed to be in the range [0, 1].
        stats (dict): A dictionary containing 'max' and 'min' keys with corresponding values
            representing the maximum and minimum of the original data.

    Returns:
        float or array-like: The data rescaled to its original range.
    """
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data
