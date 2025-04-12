import os
import numpy as np
import pickle


def load_compressed_buffer(path):
    if os.path.exists(os.path.join(path, "buffer_old.npz")):
        buffer_path = os.path.join(path, "buffer_old.npz")
    else:
        buffer_path = os.path.join(path, "buffer.npz")

    data = np.load(buffer_path)
    with open(os.path.join(path, "raw_obs.pkl"), "rb") as f:
        raw_obs = pickle.load(f)
    return data, raw_obs


def pad_to_match(arrays, axis=-1):
    """
    Pads all arrays in a list along `axis` to match the max shape.
    """
    max_dim = max(arr.shape[-1] for arr in arrays)
    padded = []
    for arr in arrays:
        if arr.shape[-1] < max_dim:
            pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, max_dim - arr.shape[-1])]
            arr = np.pad(arr, pad_width, mode="constant")
        padded.append(arr)
    return padded


def mix_and_save_buffers(buffer_dirs, save_path):
    fields_to_stack = [
        "observations",
        "privileged_obs",
        "actions",
        "rewards",
        "terminals",
        "truncated",
        "returns",
        "advantage",
    ]

    # Initialize empty list per field
    stacked_data = {key: [] for key in fields_to_stack}
    all_raw_obs = []

    total_size = 0

    for buf_dir in buffer_dirs:
        data, raw_obs = load_compressed_buffer(buf_dir)
        dsize = int(data["size"])
        print(f"Loaded buffer from {buf_dir}, size: {dsize}")
        total_size += dsize
        for key in fields_to_stack:
            stacked_data[key].append(data[key][:dsize])  # slice to buffer size

        all_raw_obs.extend(raw_obs)

    # Pad and concatenate all data
    final_data = {}
    for key in fields_to_stack:
        try:
            final_data[key] = np.concatenate(stacked_data[key], axis=0)
        except ValueError:
            print("Shape mismatch for key:", key)
            # Shape mismatch → pad arrays to match
            padded = pad_to_match(stacked_data[key])
            final_data[key] = np.concatenate(padded, axis=0)

    final_data["size"] = total_size

    for k, v in final_data.items():
        if isinstance(v, np.ndarray):
            print(f"Final shape for {k}: {v.shape}")

    # # Concatenate all data
    # final_data = {
    #     key: np.concatenate(stacked_data[key], axis=0) for key in fields_to_stack
    # }
    # final_data["size"] = total_size

    # Save the mixed data
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(os.path.join(save_path, "buffer.npz")) and not os.path.exists(
        os.path.join(save_path, "buffer_old.npz")
    ):
        os.rename(
            os.path.join(save_path, "buffer.npz"),
            os.path.join(save_path, "buffer_old.npz"),
        )

    np.savez_compressed(os.path.join(save_path, "buffer.npz"), **final_data)

    # with open(os.path.join(save_path, "raw_obs.pkl"), "wb") as f:
    #     pickle.dump(all_raw_obs, f)

    print(f"✅ Saved mixed buffer to: {save_path}, total size: {total_size}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mix and save buffers.")
    parser.add_argument(
        "--buffer_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of directories containing the buffer files.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory to save the mixed buffer.",
    )

    args = parser.parse_args()

    mix_and_save_buffers(args.buffer_dirs, args.save_path)
