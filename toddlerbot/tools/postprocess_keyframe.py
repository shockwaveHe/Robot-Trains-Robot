import argparse
import copy
import os
import pickle
import time
import tkinter as tk
from dataclasses import asdict, dataclass
from tkinter import font, ttk

import numpy as np
from tqdm import tqdm

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action, mat2quat
from toddlerbot.utils.misc_utils import find_latest_file_with_time_str


@dataclass
class Keyframe:
    name: str
    index: int
    motor_pos: np.ndarray
    joint_pos: np.ndarray | None = None
    qpos: np.ndarray | None = None


class MuJoCoApp:
    def __init__(
        self, root, sim: MuJoCoSim, robot: Robot, task_name: str, run_name: str
    ):
        self.root = root
        self.sim = sim
        self.robot = robot
        self.task_name = task_name

        self.root.title("MuJoCo Keyframe Manager")

        time_str = time.strftime("%Y%m%d_%H%M%S")
        if len(run_name) > 0:
            self.data_path = find_latest_file_with_time_str(
                os.path.join("results", run_name), "result"
            )
            if self.data_path is None:
                self.data_path = os.path.join("results", run_name, "keyframes.pkl")

            self.result_path = os.path.join(
                os.path.dirname(self.data_path), f"result_{time_str}.pkl"
            )
        else:
            self.data_path = ""
            result_dir = os.path.join(
                "results", f"{robot.name}_keyframe_{sim.name}_{time_str}"
            )
            os.makedirs(result_dir, exist_ok=True)
            self.result_path = os.path.join(result_dir, f"result_{time_str}.pkl")

        self.mirror_joint_signs = {
            "left_hip_pitch": -1,
            "left_hip_roll": 1,
            "left_hip_yaw_driven": -1,
            "left_knee": -1,
            "left_ank_pitch": -1,
            "left_ank_roll": -1,
            "left_sho_pitch": -1,
            "left_sho_roll": 1,
            "left_sho_yaw_driven": -1,
            "left_elbow_roll": 1,
            "left_elbow_yaw_driven": -1,
            "left_wrist_pitch_driven": -1,
            "left_wrist_roll": 1,
            "left_gripper_pinion": 1,
        }

        self.paused = True
        self.slider_columns = 4
        self.qpos_offset = 7

        self.create_widgets()

        self.load()

    def create_widgets(self):
        self.root.geometry("3600x1200")  # Initial window size
        large_font = font.Font(family="Helvetica", size=14)
        self.root.option_add("*Font", large_font)

        style = ttk.Style()
        style.configure(".", font=("Helvetica", 14))

        # Configure the root window's rows and columns to be resizable
        self.root.rowconfigure(0, weight=1)  # For button_frame
        self.root.rowconfigure(1, weight=2)  # For horizontal_frame
        self.root.rowconfigure(2, weight=4)  # For joint_sliders_frame
        self.root.columnconfigure(0, weight=1)  # Allow horizontal resizing

        # Button Frame
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        # Configure grid inside button_frame
        for i in range(8):  # Number of columns in button_frame
            self.button_frame.columnconfigure(i, weight=1)
        for i in range(3):  # Number of rows in button_frame
            self.button_frame.rowconfigure(i, weight=1)

        self.add_button = tk.Button(
            self.button_frame,
            text="Add Keyframe",
            command=self.add_keyframe,
            font=large_font,
        )
        self.add_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.remove_button = tk.Button(
            self.button_frame,
            text="Remove Keyframe",
            command=self.remove_keyframe,
            font=large_font,
        )
        self.remove_button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.load_button = tk.Button(
            self.button_frame,
            text="Load Keyframe",
            command=self.load_keyframe,
            font=large_font,
        )
        self.load_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        self.update_button = tk.Button(
            self.button_frame,
            text="Update Keyframe",
            command=self.update_keyframe,
            font=large_font,
        )
        self.update_button.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

        self.test_button = tk.Button(
            self.button_frame,
            text="Test Keyframe",
            command=self.test_keyframe,
            font=large_font,
        )
        self.test_button.grid(row=0, column=4, padx=5, pady=5, sticky="nsew")

        self.ground_button = tk.Button(
            self.button_frame,
            text="Put Feet on Gournd",
            command=self.put_feet_on_ground,
            font=large_font,
        )
        self.ground_button.grid(row=0, column=5, padx=5, pady=5, sticky="nsew")

        self.mirror_checked = tk.BooleanVar(value=True)
        self.mirror_checkbox = tk.Checkbutton(
            self.button_frame,
            text="Mirror",
            variable=self.mirror_checked,
            font=large_font,
        )
        self.mirror_checkbox.bind(
            "<Button-1>", lambda event: self.rev_mirror_checked.set(False)
        )
        self.mirror_checkbox.grid(row=0, column=6, padx=5, pady=5, sticky="nsew")

        self.rev_mirror_checked = tk.BooleanVar()
        self.rev_mirror_checkbox = tk.Checkbutton(
            self.button_frame,
            text="Rev. Mirror",
            variable=self.rev_mirror_checked,
            font=large_font,
        )
        self.rev_mirror_checkbox.bind(
            "<Button-1>", lambda event: self.mirror_checked.set(False)
        )
        self.rev_mirror_checkbox.grid(row=0, column=7, padx=5, pady=5, sticky="nsew")

        self.add_to_sequence_button = tk.Button(
            self.button_frame,
            text="Add to Sequence",
            command=self.add_to_sequence,
            font=large_font,
        )
        self.add_to_sequence_button.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.remove_from_sequence_button = tk.Button(
            self.button_frame,
            text="Remove from Sequence",
            command=self.remove_from_sequence,
            font=large_font,
        )
        self.remove_from_sequence_button.grid(
            row=1, column=1, padx=5, pady=5, sticky="nsew"
        )

        self.update_arrival_time_button = tk.Button(
            self.button_frame,
            text="Update Arrival Time",
            command=self.update_arrival_time,
            font=large_font,
        )
        self.update_arrival_time_button.grid(
            row=1, column=2, padx=5, pady=5, sticky="nsew"
        )

        self.move_up_button = tk.Button(
            self.button_frame, text="Move Up", command=self.move_up, font=large_font
        )
        self.move_up_button.grid(row=1, column=3, padx=5, pady=5, sticky="nsew")

        self.move_down_button = tk.Button(
            self.button_frame, text="Move Down", command=self.move_down, font=large_font
        )
        self.move_down_button.grid(row=1, column=4, padx=5, pady=5, sticky="nsew")

        self.display_trajectory_button = tk.Button(
            self.button_frame,
            text="Display Trajectory",
            command=self.display_trajectory,
            font=large_font,
        )
        self.display_trajectory_button.grid(
            row=1, column=5, padx=5, pady=5, sticky="nsew"
        )

        self.physics_enabled = tk.BooleanVar(value=True)
        self.physics_checkbox = tk.Checkbutton(
            self.button_frame,
            text="Enable Physics",
            variable=self.physics_enabled,
            font=large_font,
        )
        self.physics_checkbox.grid(row=1, column=6, padx=5, pady=5, sticky="nsew")

        self.arrival_time_label = ttk.Label(self.button_frame, text="Arrival Time:")
        self.arrival_time_label.grid(row=2, column=0, padx=3, pady=5)
        self.arrival_time_entry = ttk.Entry(self.button_frame, width=5)
        self.arrival_time_entry.grid(row=2, column=1, padx=3, pady=5)
        self.arrival_time_entry.insert(0, "0")

        self.dt_label = ttk.Label(self.button_frame, text="Interpolation Delta Time:")
        self.dt_label.grid(row=2, column=2, padx=3, pady=5)
        self.dt_entry = ttk.Entry(self.button_frame, width=5)
        self.dt_entry.grid(row=2, column=3, padx=3, pady=5)
        self.dt_entry.insert(0, "0.02")

        self.motion_name_entry = ttk.Entry(self.button_frame, width=20)
        self.motion_name_entry.grid(row=2, column=4, padx=3, pady=5)
        self.motion_name_entry.insert(0, self.task_name)

        self.save_button = tk.Button(
            self.button_frame, text="Save", command=self.save, font=large_font
        )
        self.save_button.grid(row=2, column=6, padx=5, pady=5, sticky="nsew")

        self.exit_button = tk.Button(
            self.button_frame, text="Exit", command=self.root.quit, font=large_font
        )
        self.exit_button.grid(row=2, column=7, padx=5, pady=5, sticky="nsew")

        # Horizontal Frame for lists
        self.horizontal_frame = ttk.Frame(self.root)
        self.horizontal_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self.horizontal_frame.columnconfigure(0, weight=1)
        self.horizontal_frame.columnconfigure(1, weight=1)

        # Keyframe Listbox
        self.keyframe_listbox = tk.Listbox(self.horizontal_frame, height=10)
        self.keyframe_listbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.keyframe_listbox.bind("<<ListboxSelect>>", self.on_keyframe_select)

        # Sequence Listbox
        self.sequence_listbox = tk.Listbox(self.horizontal_frame, height=10)
        self.sequence_listbox.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.sequence_listbox.bind("<<ListboxSelect>>", self.on_sequence_select)

        # Joint Sliders Frame
        self.joint_sliders_frame = ttk.Frame(self.root)
        self.joint_sliders_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Configure rows and columns for better layout
        num_columns = 12  # Number of slider columns
        for i in range(num_columns):
            self.joint_sliders_frame.columnconfigure(i, weight=1)

        for i in range(8):  # Allow room for rows
            self.joint_sliders_frame.rowconfigure(i, weight=1)

        joint_slider_names = []
        for joint_name in self.robot.joint_ordering:
            # If the joint is "left", add it and its "right" counterpart (if exists)
            if "left" in joint_name:
                joint_slider_names.append(joint_name)
                counterpart = joint_name.replace("left", "right")
                if counterpart and counterpart in self.robot.joint_ordering:
                    joint_slider_names.append(counterpart)
            elif "right" in joint_name:
                continue
            else:
                # Add other joints (neither "left" nor "right")
                joint_slider_names.append(joint_name)

        self.joint_sliders = {}  # Clear the slider dictionary
        self.num_sliders = 0
        # Create sliders for each joint in three columns
        for joint_name in joint_slider_names:
            joint_range = self.robot.joint_limits[joint_name]

            # Determine column and row for grid layout
            column = self.num_sliders % self.slider_columns
            row = self.num_sliders // self.slider_columns

            # Label
            label = ttk.Label(self.joint_sliders_frame, text=joint_name)
            label.grid(
                row=row,
                column=column * 3,
                sticky="nsew",
                padx=10,
                pady=10,
            )

            # Scale value label (to display the current value)
            value_label = ttk.Label(self.joint_sliders_frame, text="0.00")
            value_label.grid(
                row=row, column=column * 3 + 2, sticky="s", padx=10, pady=10
            )

            # Slider
            slider = ttk.Scale(
                self.joint_sliders_frame,
                from_=joint_range[0],
                to=joint_range[1],
                orient=tk.HORIZONTAL,
                length=150,
                command=lambda val,
                name=joint_name,
                label=value_label: self.update_joint_pos(name, val, label),
            )
            slider.grid(
                row=row,
                column=column * 3 + 1,
                sticky="nsew",
                padx=10,
                pady=10,
            )

            # Store the slider in the dictionary
            self.joint_sliders[joint_name] = slider

            # Double-click event binding to reset joint to initial position
            slider.bind(
                "<Button-3>",
                lambda event, name=joint_name, label=value_label: self.reset_joint_pos(
                    name, label
                ),
            )

            # Initial value
            slider.set(self.robot.default_joint_angles[joint_name])
            self.num_sliders += 1

        # import ipdb; ipdb.set_trace()
        # Start periodic slider update
        self.update_sliders_periodically()

    def on_keyframe_select(self, event):
        selected_index = self.keyframe_listbox.curselection()
        if selected_index:
            self.selected_keyframe = selected_index[0]

    def on_sequence_select(self, event):
        selected_index = self.sequence_listbox.curselection()
        if selected_index:
            self.selected_sequence = selected_index[0]

    def update_keyframe_listbox(self):
        self.keyframe_listbox.delete(0, tk.END)
        for keyframe in self.keyframes:
            self.keyframe_listbox.insert(tk.END, f"{keyframe.name} {keyframe.index}")

    def update_sequence_listbox(self):
        self.sequence_listbox.delete(0, tk.END)
        for name, arrival_time in self.sequence_list:
            self.sequence_listbox.insert(tk.END, f"{name} {arrival_time}")

    def update_sliders_periodically(self):
        for joint_name, slider in self.joint_sliders.items():
            current_value = self.sim.data.joint(joint_name).qpos
            if slider.get() != current_value:
                slider.set(float(current_value))

        # Continue updating every 100 ms
        self.root.after(100, self.update_sliders_periodically)

    def load(self):
        self.keyframes = []
        self.keyframe_listbox.delete(0, tk.END)
        self.sequence_list = []

        self.sim.data.qpos = self.sim.default_qpos.copy()
        self.keyframes.append(
            Keyframe(
                "default",
                0,
                self.sim.get_motor_angles(type="array"),
                self.sim.get_joint_angles(type="array"),
                self.sim.data.qpos.copy(),
            )
        )
        self.keyframe_listbox.insert(tk.END, "default 0")

        if len(self.data_path) > 0:
            with open(self.data_path, "rb") as f:
                print(f"Loading inputs from {self.data_path}")
                data = pickle.load(f)

            if isinstance(data, dict):
                keyframes = [Keyframe(**k) for k in data.get("keyframes", [])]
                self.sequence_list = data.get("sequence", [])
                self.traj_times = data.get("time", [])
                self.action_traj = data.get("action_traj", [])
                self.update_sequence_listbox()
            else:
                keyframes = data

            for i, key in enumerate(keyframes):
                if isinstance(key, Keyframe):
                    keyframe = key
                else:
                    self.sim.set_motor_angles(key)
                    keyframe = Keyframe(
                        "plank",
                        i,
                        self.sim.get_motor_angles(type="array"),
                        self.sim.get_joint_angles(type="array"),
                        self.sim.data.qpos.copy(),
                    )
                    self.sim.data.qpos = self.sim.default_qpos.copy()

                self.keyframes.append(keyframe)
                self.keyframe_listbox.insert(
                    tk.END, f"{keyframe.name} {keyframe.index}"
                )

    def save(self):
        result_dict = {}
        saved_keyframes = []
        for keyframe in self.keyframes:
            if "default" not in keyframe.name:
                saved_keyframes.append(asdict(keyframe))

        result_dict["keyframes"] = saved_keyframes
        result_dict["sequence"] = self.sequence_list
        result_dict["time"] = self.traj_times
        result_dict["action_traj"] = self.action_traj
        result_dict["ee_traj"] = self.ee_traj
        result_dict["root_traj"] = self.root_traj

        with open(self.result_path, "wb") as f:
            print(f"Saving the results to {self.result_path}")
            pickle.dump(result_dict, f)

        motion_name = self.motion_name_entry.get()
        motion_file_path = os.path.join("toddlerbot", "motion", f"{motion_name}.pkl")
        with open(motion_file_path, "wb") as f:
            print(f"Saving the results to {motion_file_path}")
            pickle.dump(result_dict, f)

    def update_joint_pos(self, joint_name, value, label):
        joint_pos = float(value)
        joint_angles = self.sim.get_joint_angles()
        joint_angles[joint_name] = joint_pos

        label.config(text=str(round(joint_pos, 3)))
        mirror_checked, rev_mirror_checked = (
            self.mirror_checked.get(),
            self.rev_mirror_checked.get(),
        )
        if mirror_checked or rev_mirror_checked:
            if "left" in joint_name or "right" in joint_name:
                mirrored_joint_name = (
                    joint_name.replace("left", "right")
                    if "left" in joint_name
                    else joint_name.replace("right", "left")
                )
                mirror_sign = (
                    self.mirror_joint_signs[joint_name]
                    if "left" in joint_name
                    else self.mirror_joint_signs[mirrored_joint_name]
                )
                joint_angles[mirrored_joint_name] = (
                    mirror_checked * joint_pos * mirror_sign
                    - rev_mirror_checked * joint_pos * mirror_sign
                )

        self.sim.set_joint_angles(joint_angles)
        self.sim.forward()

    def reset_joint_pos(self, joint_name, label):
        joint_pos = self.sim.model.joint(joint_name).qpos0
        self.sim.data.joint(joint_name).qpos = joint_pos
        label.config(text=str(joint_pos))
        self.sim.forward()

    def add_keyframe(self):
        selected_index = self.keyframe_listbox.curselection()
        idx = -1
        if selected_index:
            idx = selected_index[0]

        new_keyframe = copy.deepcopy(self.keyframes[idx])
        motion_name = self.motion_name_entry.get()
        if "default" in new_keyframe.name:
            new_keyframe.name = motion_name

        unique_index = 1
        keyframe_indices = []
        for keyframe in self.keyframes:
            if keyframe.name == new_keyframe.name:
                keyframe_indices.append(keyframe.index)

        # Find the minimum unique index
        while unique_index in keyframe_indices:
            unique_index += 1

        new_keyframe.index = unique_index
        self.keyframes.append(new_keyframe)
        self.keyframe_listbox.insert(
            tk.END, f"{new_keyframe.name} {new_keyframe.index}"
        )

    def remove_keyframe(self):
        if hasattr(self, "selected_keyframe"):
            keyframe = self.keyframes[self.selected_keyframe]
            for name, arrival_time in self.sequence_list:
                if name == f"{keyframe.name} {keyframe.index}":
                    self.sequence_list.remove((name, arrival_time))
                    self.update_sequence_listbox()

            self.keyframes.pop(self.selected_keyframe)
            self.update_keyframe_listbox()

    def load_keyframe(self):
        if hasattr(self, "selected_keyframe"):
            keyframe = self.keyframes[self.selected_keyframe]
            self.sim.data.qpos = keyframe.qpos.copy()
            self.sim.forward()

    def update_keyframe(self):
        if hasattr(self, "selected_keyframe"):
            self.keyframes[
                self.selected_keyframe
            ].motor_pos = self.sim.get_motor_angles(type="array")
            self.keyframes[
                self.selected_keyframe
            ].joint_pos = self.sim.get_joint_angles(type="array")
            self.keyframes[self.selected_keyframe].qpos = self.sim.data.qpos.copy()

    def put_feet_on_ground(self):
        left_foot_transform = self.sim.get_body_transofrm(self.sim.left_foot_name)
        right_foot_transform = self.sim.get_body_transofrm(self.sim.right_foot_name)
        torso_curr_transform = self.sim.get_body_transofrm("torso")

        # Select the foot with the smaller z-coordinate
        if left_foot_transform[2, 3] < right_foot_transform[2, 3]:
            aligned_torso_transform = (
                self.sim.left_foot_transform
                @ np.linalg.inv(left_foot_transform)
                @ torso_curr_transform
            )
        else:
            aligned_torso_transform = (
                self.sim.right_foot_transform
                @ np.linalg.inv(right_foot_transform)
                @ torso_curr_transform
            )

        # Update the simulation with the new torso position and orientation
        self.sim.data.qpos[:3] = aligned_torso_transform[:3, 3]  # Update position
        self.sim.data.qpos[3:7] = mat2quat(aligned_torso_transform[:3, :3])
        self.sim.forward()

    def test_keyframe(self):
        dt = float(self.dt_entry.get())
        enabled = self.physics_enabled.get()

        motor_target = self.sim.get_motor_angles()
        for i in tqdm(range(100), desc="Testing Keyframe"):
            if not enabled:
                self.sim.set_motor_angles(motor_target)
                self.sim.forward()
            else:
                self.sim.set_motor_target(motor_target)
                self.sim.step()

            self.root.after(int(dt * 1000))

    def add_to_sequence(self):
        selected_index = self.keyframe_listbox.curselection()
        if selected_index:
            keyframe_name = self.keyframe_listbox.get(selected_index)
            arrival_time = float(self.arrival_time_entry.get())

            self.sequence_list.append((keyframe_name, arrival_time))
            self.update_sequence_listbox()

    def remove_from_sequence(self):
        selected_index = self.sequence_listbox.curselection()
        if selected_index:
            self.sequence_list.pop(selected_index[0])
            self.update_sequence_listbox()

    def update_arrival_time(self):
        if hasattr(self, "selected_sequence"):
            arrival_time = self.sequence_list[self.selected_sequence][1]
            arrival_time_new = float(self.arrival_time_entry.get())
            for i in range(self.selected_sequence, len(self.sequence_list)):
                self.sequence_list[i] = (
                    self.sequence_list[i][0],
                    self.sequence_list[i][1] + arrival_time_new - arrival_time,
                )

            self.update_sequence_listbox()

    def move_up(self):
        if hasattr(self, "selected_keyframe") and self.keyframe_listbox.curselection():
            index = self.selected_keyframe
            self.keyframes[index - 1], self.keyframes[index] = (
                self.keyframes[index],
                self.keyframes[index - 1],
            )
            self.update_keyframe_listbox()
            self.keyframe_listbox.select_set(index - 1)
            self.selected_keyframe = index - 1
        elif (
            hasattr(self, "selected_sequence") and self.sequence_listbox.curselection()
        ):
            print("move up sequence")
            index = self.selected_sequence
            self.sequence_list[index - 1], self.sequence_list[index] = (
                self.sequence_list[index],
                self.sequence_list[index - 1],
            )
            self.update_sequence_listbox()
            self.sequence_listbox.select_set(index - 1)
            self.selected_sequence = index - 1

    def move_down(self):
        if hasattr(self, "selected_keyframe") and self.keyframe_listbox.curselection():
            index = self.selected_keyframe
            self.keyframes[index + 1], self.keyframes[index] = (
                self.keyframes[index],
                self.keyframes[index + 1],
            )
            self.update_keyframe_listbox()
            self.keyframe_listbox.select_set(index + 1)
            self.selected_keyframe = index + 1
        elif (
            hasattr(self, "selected_sequence") and self.sequence_listbox.curselection()
        ):
            print("move down sequence")
            index = self.selected_sequence
            self.sequence_list[index + 1], self.sequence_list[index] = (
                self.sequence_list[index],
                self.sequence_list[index + 1],
            )
            self.update_sequence_listbox()
            self.sequence_listbox.select_set(index + 1)
            self.selected_sequence = index + 1

    def display_trajectory(self):
        # Extract positions and arrival times from the sequence
        start_idx = 0

        selected_index = self.sequence_listbox.curselection()
        if selected_index:
            start_idx = selected_index[0]

        action_list = []
        qpos_list = []
        times = []
        for keyframe_name, arrival_time in self.sequence_list:
            for keyframe in self.keyframes:
                if keyframe_name in f"{keyframe.name} {keyframe.index}":
                    action_list.append(keyframe.motor_pos)
                    qpos_list.append(keyframe.qpos)
                    times.append(arrival_time)
                    break

        action_arr = np.array(action_list)
        times = np.array(times) - times[0]

        dt = float(self.dt_entry.get())
        enabled = self.physics_enabled.get()
        self.traj_times = np.array([t for t in np.arange(0, times[-1], dt)])
        self.action_traj = []
        for t in self.traj_times:
            if t < times[-1]:
                motor_pos = interpolate_action(t, times, action_arr)
            else:
                motor_pos = action_arr[-1]

            self.action_traj.append(motor_pos)

        self.sim.set_qpos(qpos_list[start_idx])
        self.sim.forward()

        traj_start = int(np.searchsorted(self.traj_times, times[start_idx]))

        self.ee_traj = []
        self.root_traj = []
        for i, motor_target in enumerate(
            tqdm(self.action_traj[traj_start:], desc="Displaying Trajectory")
        ):
            t1 = time.time()
            if not enabled:
                self.sim.set_motor_angles(motor_target)
                self.sim.forward()
            else:
                self.sim.set_motor_target(motor_target)
                self.sim.step()

            ee_pose_combined = []
            for side in ["left", "right"]:
                ee_pos = self.sim.data.site(f"{side}_ee_center").xpos.copy()
                ee_quat = mat2quat(
                    self.sim.data.site(f"{side}_ee_center").xmat.reshape(3, 3).copy()
                )
                ee_pose_combined.extend(ee_pos)
                ee_pose_combined.extend(ee_quat)

            self.ee_traj.append(np.array(ee_pose_combined, dtype=np.float32))
            self.root_traj.append(self.sim.data.qpos[:7])

            t2 = time.time()

            if dt - (t2 - t1) > 0:
                self.root.after(int((dt - (t2 - t1)) * 1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="push_up",
        help="The name of the task",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="The path of the keyframes",
    )
    args = parser.parse_args()

    root = tk.Tk()
    robot = Robot(args.robot)
    sim = MuJoCoSim(robot, vis_type="view", fixed_base="fixed" in args.robot)

    app = MuJoCoApp(root, sim, robot, args.task, args.run_name)

    root.mainloop()
