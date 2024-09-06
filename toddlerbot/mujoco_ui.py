import tkinter as tk
from tkinter import simpledialog, messagebox
from typing import Dict
import mujoco
import mujoco.viewer as viewer
import os
from scipy.interpolate import CubicSpline
import numpy as np
from toddlerbot.sim.robot import Robot
# TODO: fix repload and simulation unstable issue.
class MuJoCoApp:
    def __init__(self, root, robot: Robot):
        self.root = root
        self.robot = robot
        self.root.title("MuJoCo Keyframe Manager")
        
        self.model = None
        self.data = None
        self.viewer = None
        
        self.paused = True
        self.keyframes = []
        self.current_index = 0
        self.slider_columns = 4
        
        self.key_frame_dir = "toddlerbot/key_frames"
        self.qpos_offset = 6
        self.create_widgets()
        self.load_keyframes()
        self.joint_sliders = {}
        self.load_xml()
    

    def create_widgets(self):
        # Load XML button, Keyframe input, and Record button in one row
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=5)

        self.load_button = tk.Button(top_frame, text="Load XML", command=self.load_xml)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.keyframe_name_entry = tk.Entry(top_frame)
        self.keyframe_name_entry.pack(side=tk.LEFT, padx=5)

        self.record_button = tk.Button(top_frame, text="Record Keyframe", command=self.record_keyframe)
        self.record_button.pack(side=tk.LEFT, padx=5)

        self.trajectory_name_entry = tk.Entry(top_frame)
        self.trajectory_name_entry.pack(side=tk.LEFT, padx=5)

        self.store_trajectory_button = tk.Button(top_frame, text="Store Trajectory", command=self.store_trajectory)
        self.store_trajectory_button.pack(side=tk.LEFT, padx=5)

        # Keyframe management buttons arranged in a grid
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)

        self.load_button = tk.Button(self.button_frame, text="Load Keyframe", command=self.load_keyframe)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.rename_button = tk.Button(self.button_frame, text="Rename Keyframe", command=self.rename_keyframe)
        self.rename_button.grid(row=0, column=1, padx=5, pady=5)

        self.delete_button = tk.Button(self.button_frame, text="Delete Keyframe", command=self.delete_keyframe)
        self.delete_button.grid(row=0, column=2, padx=5, pady=5)

        self.update_button = tk.Button(self.button_frame, text="Update Keyframe", command=self.update_keyframe)
        self.update_button.grid(row=0, column=3, padx=5, pady=5)

        self.move_up_button = tk.Button(self.button_frame, text="Move Up", command=self.move_up)
        self.move_up_button.grid(row=1, column=1, padx=5, pady=5)

        self.move_down_button = tk.Button(self.button_frame, text="Move Down", command=self.move_down)
        self.move_down_button.grid(row=1, column=2, padx=5, pady=5)

        self.pause_button = tk.Button(self.button_frame, text="Start Simulation", command=self.toggle_simulation)
        self.pause_button.grid(row=0, column=4, padx=5, pady=5)

        self.mirror_checked = tk.BooleanVar()
        self.mirror_checkbox = tk.Checkbutton(self.button_frame, text="Mirror", variable=self.mirror_checked)
        self.mirror_checkbox.bind("<Button-1>", lambda event: self.rev_mirror_checked.set(False))
        self.mirror_checkbox.grid(row=0, column=5, padx=5, pady=5)

        self.rev_mirror_checked = tk.BooleanVar()
        self.rev_mirror_checkbox = tk.Checkbutton(self.button_frame, text="Rev. Mirror", variable=self.rev_mirror_checked)
        self.rev_mirror_checkbox.bind("<Button-1>", lambda event: self.mirror_checked.set(False))
        self.rev_mirror_checkbox.grid(row=1, column=5, padx=5, pady=5)
        # Gravity toggle button
        # self.gravity_button = tk.Button(self.button_frame, text="Toggle Gravity", command=self.toggle_gravity)
        # self.gravity_button.grid(row=1, column=5, padx=5, pady=5)

        self.sequence_list = []
        # Horizontal frame to hold both the keyframe list and sequence list
        self.horizontal_frame = tk.Frame(self.root)
        self.horizontal_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        # Keyframe listbox
        self.keyframe_listbox = tk.Listbox(self.horizontal_frame, height=10)
        self.keyframe_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.keyframe_listbox.bind('<<ListboxSelect>>', self.on_keyframe_select)

        # Sequence list frame to hold keyframes and their arrival times
        self.sequence_listbox = tk.Listbox(self.horizontal_frame, height=10)
        self.sequence_listbox.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.sequence_listbox.bind('<<ListboxSelect>>', self.on_sequence_select)


        # Sliders for joints in three columns
        self.joint_sliders_frame = tk.Frame(self.root)
        self.joint_sliders_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    
        self.add_to_sequence_button = tk.Button(self.button_frame, text="Add to Sequence", command=self.add_to_sequence)
        self.add_to_sequence_button.grid(row=1, column=0, padx=5, pady=5)

        self.update_arrival_time_button = tk.Button(self.button_frame, text="Update Arrival Time", command=self.update_arrival_time)
        self.update_arrival_time_button.grid(row=1, column=3, padx=5, pady=5)

        self.display_trajectory_button = tk.Button(self.button_frame, text="Display Trajectory", command=self.display_trajectory)
        self.display_trajectory_button.grid(row=1, column=4, padx=5, pady=5)
        

        # Total time , Arrival Time, and Interpolation Steps input
        self.total_time_label = tk.Label(self.button_frame, text="Total time:")
        self.total_time_label.grid(row=2, column=2, padx=3, pady=5)
        self.total_time_entry = tk.Entry(self.button_frame, width=5)
        self.total_time_entry.grid(row=2, column=3, padx=3, pady=5)
        self.total_time_entry.insert(0, "5")

        self.arrival_time_label = tk.Label(self.button_frame, text="Arrival Time:")
        self.arrival_time_label.grid(row=2, column=0, padx=3, pady=5)
        self.arrival_time_entry = tk.Entry(self.button_frame, width=5)
        self.arrival_time_entry.grid(row=2, column=1, padx=3, pady=5)
        self.arrival_time_entry.insert(0, "0")

        self.interpolation_steps_label = tk.Label(self.button_frame, text="Interpolation Steps:")
        self.interpolation_steps_label.grid(row=2, column=4, padx=3, pady=5)
        self.interpolation_steps_entry = tk.Entry(self.button_frame, width=5)
        self.interpolation_steps_entry.grid(row=2, column=5, padx=3, pady=5)
        self.interpolation_steps_entry.insert(0, "500")

        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.root.quit)
        self.exit_button.grid(row=2, column=6, padx=5, pady=5)
    
    def load_xml(self):
        # xml_file = filedialog.askopenfilename(initialdir = "toddlerbot/robot_descriptions/toddlerbot", filetypes=[("XML files", "*.xml")])
        xml_file = "toddlerbot/robot_descriptions/toddlerbot/toddlerbot_scene.xml"
        # xml_file = "toddlerbot/robot_descriptions/franka_sim/franka_panda.xml"
        if xml_file:
            print(xml_file)
            self.model = mujoco.MjModel.from_xml_path(xml_file)
            self.data = mujoco.MjData(self.model)
            self.viewer = viewer.launch_passive(self.model, self.data)
            self.create_joint_sliders()
            self.root.after(100, self.update_viewer)
    
    def load_joint_positions(self, joint_positions: Dict[str, float]):
        for joint_name, joint_pos in joint_positions.items():
            self.data.joint(joint_name).qpos = joint_pos

        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()
        self.update_sliders()

    def update_viewer(self):
        if self.viewer is not None:
            if not self.paused:
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            self.root.after(5, self.update_viewer)
    
    def update_sliders(self):
        for i in range(self.model.njnt):
            joint_slider = self.joint_sliders_frame.grid_slaves(row=i//self.slider_columns, column=(i % self.slider_columns) * (self.slider_columns - 1) + 1)
            if joint_slider:
                slider = joint_slider[0]
                slider.set(self.data.qpos[i + self.qpos_offset])

    def create_joint_sliders(self):
        # Clear existing sliders
        for widget in self.joint_sliders_frame.winfo_children():
            widget.destroy()

        if self.model is None:
            return

        self.joint_sliders.clear()  # Clear the slider dictionary
        self.num_sliders = 0
        # Create sliders for each joint in three columns
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None or 'act' in joint_name or 'rev' in joint_name or ('drive' in joint_name and 'driven' not in joint_name):
                continue # Skip actuator and driven joints
            
            joint_limited = self.model.jnt_limited[i]
            joint_range = self.model.jnt_range[i] if joint_limited else (-3.14, 3.14)

            # Determine column and row for grid layout
            column = self.num_sliders % self.slider_columns
            row = self.num_sliders // self.slider_columns

            # Label
            label = tk.Label(self.joint_sliders_frame, text=joint_name)
            label.grid(row=row, column=column*(self.slider_columns - 1), sticky="w")

            # Slider
            slider = tk.Scale(self.joint_sliders_frame, from_=joint_range[0], to=joint_range[1],
                            orient=tk.HORIZONTAL, resolution=0.01, length=150,
                            command=lambda val, idx=i: self.update_joint_position(idx, val, self.qpos_offset))
            slider.grid(row=row, column=column * (self.slider_columns - 1) + 1, sticky="ew")

            # Store the slider in the dictionary
            self.joint_sliders[i + self.qpos_offset] = slider

            # Double-click event binding to reset joint to initial position
            slider.bind("<Double-Button-1>", lambda event, idx=i + self.qpos_offset: self.reset_joint(idx))

            # Initial value
            slider.set(self.data.qpos[i + self.qpos_offset])
            self.num_sliders += 1

        # add whole body pitch z sliders
        self.create_joint_slider(2, (0, 0.5), "torso_z")
        self.create_joint_slider(5, (-3.14, 3.14), "torso_pitch")

        # import ipdb; ipdb.set_trace()
        # Start periodic slider update
        self.update_sliders_periodically()

    def create_joint_slider(self, qpos_index, joint_range, joint_name):
        # create joint slider mannually
        label = tk.Label(self.joint_sliders_frame, text=joint_name)
        # Determine column and row for grid layout
        column = self.num_sliders % self.slider_columns
        row = self.num_sliders // self.slider_columns
        label.grid(row=row, column=column*(self.slider_columns - 1), sticky="w")

        # Slider
        slider = tk.Scale(self.joint_sliders_frame, from_=joint_range[0], to=joint_range[1],
                        orient=tk.HORIZONTAL, resolution=0.01, length=150,
                        command=lambda val, idx=qpos_index: self.update_joint_position(idx, val))
        slider.grid(row=row, column=column * (self.slider_columns - 1) + 1, sticky="ew")

        # Store the slider in the dictionary
        self.joint_sliders[qpos_index] = slider

        # Double-click event binding to reset joint to initial position
        slider.bind("<Double-Button-1>", lambda event, idx=qpos_index: self.reset_joint(idx))

        # Initial value
        slider.set(self.data.qpos[qpos_index])
        self.num_sliders += 1

    def update_sliders_periodically(self):
        for i, slider in self.joint_sliders.items():
            current_value = self.data.qpos[i]
            if slider.get() != current_value:
                slider.set(current_value)
        
        # Continue updating every 100 ms
        self.root.after(100, self.update_sliders_periodically)
   
    def reset_joint(self, joint_index):
        initial_position = self.model.qpos0[joint_index]
        self.data.qpos[joint_index] = initial_position
        # self.data.qvel[joint_index + self.qpos_offset] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def update_joint_position(self, joint_index, value, qpos_offset=0):
        self.data.qpos[joint_index + qpos_offset] = float(value)
        mirror_checked, rev_mirror_checked = self.mirror_checked.get(), self.rev_mirror_checked.get()
        if mirror_checked or rev_mirror_checked:
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if 'left' in joint_name or 'right' in joint_name:
                mirrored_joint_name = joint_name.replace("left", "right") if "left" in joint_name else joint_name.replace("right", "left")
                mirrored_joint_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mirrored_joint_name)
                self.data.qpos[mirrored_joint_index + self.qpos_offset] = mirror_checked * float(value) - rev_mirror_checked * float(value)
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def load_keyframes(self):
        self.keyframes = []
        self.keyframe_listbox.delete(0, tk.END)
        
        if not os.path.exists(self.key_frame_dir):
            return
        
        keyframe_files = sorted(os.listdir(self.key_frame_dir))
        
        for filename in keyframe_files:
            if filename.endswith(".txt"):
                index_str, keyframe_name = filename[:-4].split("_", 1)
                index = int(index_str)
                keyframe_path = os.path.join(self.key_frame_dir, filename)
                with open(keyframe_path, "r") as f:
                    qpos = [float(line.strip()) for line in f.readlines()]
                    self.keyframes.append((index, keyframe_name, qpos))
                    self.keyframe_listbox.insert(tk.END, keyframe_name)
        
        if self.keyframes:
            self.current_index = max(index + 1 for index, _, _ in self.keyframes)
        else:
            self.current_index = 0

    def record_keyframe(self):
        if self.data is None:
            messagebox.showerror("Error", "Load a MuJoCo model first.")
            return
        
        keyframe_name = self.keyframe_name_entry.get() or f"Keyframe {self.current_index}"
        keyframe = (self.current_index, keyframe_name, self.data.qpos.copy()[self.qpos_offset:])
        self.keyframes.append(keyframe)
        self.keyframe_listbox.insert(tk.END, keyframe_name)

        # Save keyframe to file
        keyframe_path = os.path.join(self.key_frame_dir, f"{self.current_index}_{keyframe_name}.txt")
        with open(keyframe_path, "w") as f:
            for value in self.data.qpos:
                f.write(f"{value}\n")
        self.current_index += 1
    
    def on_keyframe_select(self, event):
        selected_index = self.keyframe_listbox.curselection()
        if selected_index:
            self.selected_keyframe = selected_index[0]
    
    def load_keyframe(self):
        if hasattr(self, 'selected_keyframe'):
            index, keyframe_name, qpos = self.keyframes[self.selected_keyframe]
            self.data.qpos = qpos
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            # import ipdb; ipdb.set_trace()
            self.update_sliders()
    
    def rename_keyframe(self):
        if hasattr(self, 'selected_keyframe'):
            index, old_name, qpos = self.keyframes[self.selected_keyframe]
            new_name = simpledialog.askstring("Rename Keyframe", "Enter new name:")
            if new_name:
                old_file_path = os.path.join(self.key_frame_dir, f"{index}_{old_name}.txt")
                new_file_path = os.path.join(self.key_frame_dir, f"{index}_{new_name}.txt")
                
                os.rename(old_file_path, new_file_path)
                self.keyframes[self.selected_keyframe] = (index, new_name, qpos)
                self.update_keyframe_listbox()
    
    def delete_keyframe(self):
        # TODO: debug this part
        if hasattr(self, 'selected_keyframe') and self.keyframe_listbox.curselection():
            index, keyframe_name, _ = self.keyframes[self.selected_keyframe]
            keyframe_file = f"{index}_{keyframe_name}.txt"
            
            # Delete the keyframe file
            os.remove(os.path.join(self.key_frame_dir, keyframe_file))
            
            # Remove the keyframe from the list
            del self.keyframes[self.selected_keyframe]
            self.keyframe_listbox.delete(self.selected_keyframe)
            
            # Update indices and filenames of subsequent keyframes
            for i in range(self.selected_keyframe, len(self.keyframes)):
                new_index = i
                old_index, old_name, qpos = self.keyframes[i]
                new_name = f"{new_index}_{old_name}.txt"
                old_file_path = os.path.join(self.key_frame_dir, f"{old_index}_{old_name}.txt")
                new_file_path = os.path.join(self.key_frame_dir, new_name)
                
                os.rename(old_file_path, new_file_path)
                self.keyframes[i] = (new_index, old_name, qpos)
            
            self.update_keyframe_listbox()
            self.current_index -= 1
        
        elif hasattr(self, 'selected_sequence') and self.sequence_listbox.curselection():
            del self.sequence_list[self.selected_sequence]
            self.update_sequence_listbox()
    
    def update_keyframe(self):
        if hasattr(self, 'selected_keyframe'):
            index, keyframe_name, _ = self.keyframes[self.selected_keyframe]
            self.keyframes[self.selected_keyframe] = (index, keyframe_name, self.data.qpos.copy()[self.qpos_offset:])
            
            # Update the keyframe file
            keyframe_path = os.path.join(self.key_frame_dir, f"{index}_{keyframe_name}.txt")
            with open(keyframe_path, "w") as f:
                for value in self.data.qpos[self.qpos_offset:]:
                    f.write(f"{value}\n")

    def move_up(self):
        if hasattr(self, 'selected_keyframe') and self.keyframe_listbox.curselection():
            index = self.selected_keyframe
            self.keyframes[index - 1], self.keyframes[index] = self.keyframes[index], self.keyframes[index - 1]
            self.update_keyframe_listbox()
            self.keyframe_listbox.select_set(index - 1)
            self.selected_keyframe = index - 1
        elif hasattr(self, 'selected_sequence') and self.sequence_listbox.curselection():
            print('move up sequence')
            index = self.selected_sequence
            self.sequence_list[index - 1], self.sequence_list[index] = self.sequence_list[index], self.sequence_list[index - 1]
            self.update_sequence_listbox()
            self.sequence_listbox.select_set(index - 1)
            self.selected_sequence = index - 1
    
    def move_down(self):
        if hasattr(self, 'selected_keyframe') and self.keyframe_listbox.curselection():
            index = self.selected_keyframe
            self.keyframes[index + 1], self.keyframes[index] = self.keyframes[index], self.keyframes[index + 1]
            self.update_keyframe_listbox()
            self.keyframe_listbox.select_set(index + 1)
            self.selected_keyframe = index + 1
        elif hasattr(self, 'selected_sequence') and self.sequence_listbox.curselection():
            print('move down sequence')
            index = self.selected_sequence
            self.sequence_list[index + 1], self.sequence_list[index] = self.sequence_list[index], self.sequence_list[index + 1]
            self.update_sequence_listbox()
            self.sequence_listbox.select_set(index + 1)
            self.selected_sequence = index + 1
    
    def update_keyframe_listbox(self):
        # TODO: rename the files
        self.keyframe_listbox.delete(0, tk.END)
        for index, name, _ in self.keyframes:
            self.keyframe_listbox.insert(tk.END, f"{index} {name}")
    
    def update_sequence_listbox(self):
        self.sequence_listbox.delete(0, tk.END)
        for name, _ in self.sequence_list:
            self.sequence_listbox.insert(tk.END, name)
    
    def toggle_simulation(self):
        if self.data is not None:
            self.paused = not self.paused
            if self.paused:
                self.pause_button.config(text="Resume Simulation")
            else:
                self.pause_button.config(text="Pause Simulation")

    def toggle_gravity(self):
        if self.data is not None:
            # Check current gravity and toggle
            if self.model.opt.gravity[2] == -9.8:
                self.model.opt.gravity[:] = 0.0
                print("Gravity off")
            else:
                self.model.opt.gravity[:] = [0.0, 0.0, -9.8]
                print("Gravity on")

            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            
    def on_sequence_select(self, event):
        selected_index = self.sequence_listbox.curselection()
        if selected_index:
            self.selected_sequence = selected_index[0]

    def add_to_sequence(self):
        selected_index = self.keyframe_listbox.curselection()
        if selected_index:
            keyframe_name = self.keyframe_listbox.get(selected_index)
            arrival_time = float(self.arrival_time_entry.get())
            keyframe_name += f" {arrival_time}"

            self.sequence_list.append((keyframe_name, float(self.arrival_time_entry.get())))
            self.update_sequence_listbox()
    
    def update_arrival_time(self):
        if hasattr(self, 'selected_sequence'):
            keyframe_name = self.sequence_list[self.selected_sequence][0].rsplit(" ", 1)[0]
            arrival_time = float(self.arrival_time_entry.get())
            self.sequence_list[self.selected_sequence] = (f"{keyframe_name} {arrival_time}", arrival_time)
            self.update_sequence_listbox()
        
    def generate_trajectory(self):
        # Extract positions and arrival times from the sequence
        positions = []
        times = []
        for keyframe_name, arrival_time in self.sequence_list:
            keyframe_name = keyframe_name.rsplit(" ", 1)[0]

            index = next(i for i, (i, name, _) in enumerate(self.keyframes) if name == keyframe_name)
            positions.append(self.keyframes[index][2])
            times.append(arrival_time)
        
        positions = np.array(positions)
        times = np.array(times) * float(self.total_time_entry.get())
        # Interpolation using cubic spline
        spline = CubicSpline(times, positions, bc_type='clamped')
        self.trajectory_times = np.array([t for t in np.linspace(0, float(self.total_time_entry.get()), int(self.interpolation_steps_entry.get()))])
        self.trajectory = [spline(t) for t in self.trajectory_times]

    def store_trajectory(self):
        assert hasattr(self, 'trajectory'), "Please generate a trajectory first"
        joint_indexs = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in self.robot.joint_ordering]
        joint_action = np.array(self.trajectory)[:, self.qpos_offset:][:, joint_indexs]
        motor_action = np.zeros_like(joint_action)
        for i, joint_pos in enumerate(joint_action):
            motor_action[i] = np.array(
                list(
                    robot.joint_to_motor_angles(
                        dict(zip(robot.joint_ordering, joint_pos))
                    ).values()
                ),
                dtype=np.float32,
            )
        trajectory_name = self.trajectory_name_entry.get()
        assert trajectory_name, "Please enter a trajectory name"
        np.savez(f"toddlerbot/ref_motion/{trajectory_name}", time=self.trajectory_times, action=motor_action)

    def display_trajectory(self):
        self.generate_trajectory()

        for qpos in self.trajectory:
            self.data.qpos = qpos
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            self.root.after(int(float(self.total_time_entry.get()) * 1000 / len(self.trajectory)))


if __name__ == "__main__":
    root = tk.Tk()
    robot = Robot("toddlerbot")
    app = MuJoCoApp(root, robot)
    root.mainloop()
