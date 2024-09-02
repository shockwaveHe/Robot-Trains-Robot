import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import mujoco
import mujoco.viewer as viewer
import os

class MuJoCoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MuJoCo Keyframe Manager")
        
        self.model = None
        self.data = None
        self.viewer = None
        
        self.paused = True
        self.keyframes = []
        self.current_index = 0
        self.slider_columns = 4
        
        self.key_frame_dir = "toddlerbot/key_frames"
        self.create_widgets()
        self.load_keyframes()
        self.joint_sliders = {}
    
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

        # self.move_up_button = tk.Button(self.button_frame, text="Move Up", command=self.move_up)
        # self.move_up_button.grid(row=1, column=0, padx=5, pady=5)

        # self.move_down_button = tk.Button(self.button_frame, text="Move Down", command=self.move_down)
        # self.move_down_button.grid(row=1, column=1, padx=5, pady=5)

        self.pause_button = tk.Button(self.button_frame, text="Start Simulation", command=self.toggle_simulation)
        self.pause_button.grid(row=0, column=4, padx=5, pady=5)

        # # Gravity toggle button
        # self.gravity_button = tk.Button(self.button_frame, text="Toggle Gravity", command=self.toggle_gravity)
        # self.gravity_button.grid(row=1, column=3, padx=5, pady=5)

        # Keyframe listbox
        self.keyframe_listbox = tk.Listbox(self.root)
        self.keyframe_listbox.pack(pady=5, fill=tk.BOTH, expand=True)
        self.keyframe_listbox.bind('<<ListboxSelect>>', self.on_keyframe_select)

        # Sliders for joints in three columns
        self.joint_sliders_frame = tk.Frame(self.root)
        self.joint_sliders_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    
    def load_xml(self):
        # xml_file = filedialog.askopenfilename(initialdir = "toddlerbot/robot_descriptions/toddlerbot", filetypes=[("XML files", "*.xml")])
        xml_file = "toddlerbot/robot_descriptions/toddlerbot/toddlerbot_scene.xml"
        if xml_file:
            self.model = mujoco.MjModel.from_xml_path(xml_file)
            self.data = mujoco.MjData(self.model)
            self.viewer = viewer.launch_passive(self.model, self.data)
            self.create_joint_sliders()
            self.root.after(100, self.update_viewer)
    
    def update_viewer(self):
        if self.viewer is not None:
            if not self.paused:
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            self.root.after(100, self.update_viewer)
    
    def update_sliders(self):
        for i in range(self.model.njnt):
            joint_slider = self.joint_sliders_frame.grid_slaves(row=i//self.slider_columns, column=(i % self.slider_columns) * (self.slider_columns - 1) + 1)
            if joint_slider:
                slider = joint_slider[0]
                slider.set(self.data.qpos[i])

    def create_joint_sliders(self):
        # Clear existing sliders
        for widget in self.joint_sliders_frame.winfo_children():
            widget.destroy()

        if self.model is None:
            return

        self.joint_sliders.clear()  # Clear the slider dictionary

        # Create sliders for each joint in three columns
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None:
                continue
            
            joint_limited = self.model.jnt_limited[i]
            joint_range = self.model.jnt_range[i] if joint_limited else (-3.14, 3.14)

            # Determine column and row for grid layout
            column = i % self.slider_columns
            row = i // self.slider_columns

            # Label
            label = tk.Label(self.joint_sliders_frame, text=joint_name)
            label.grid(row=row, column=column*(self.slider_columns - 1), sticky="w")

            # Slider
            slider = tk.Scale(self.joint_sliders_frame, from_=joint_range[0], to=joint_range[1],
                            orient=tk.HORIZONTAL, resolution=0.01, length=150,
                            command=lambda val, idx=i: self.update_joint_position(idx, val))
            slider.grid(row=row, column=column*(self.slider_columns - 1)+1, sticky="ew")

            # Store the slider in the dictionary
            self.joint_sliders[i] = slider

            # Double-click event binding to reset joint to initial position
            slider.bind("<Double-Button-1>", lambda event, idx=i: self.reset_joint(idx))

            # Initial value
            slider.set(self.data.qpos[i])

        # Start periodic slider update
        self.update_sliders_periodically()

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
        self.data.qvel[joint_index] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    # The slider update will be handled by the periodic callback


    def update_joint_position(self, joint_index, value):
        self.data.qpos[joint_index] = float(value)
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
                    self.keyframe_listbox.insert(tk.END, filename[:-4])
        
        if self.keyframes:
            self.current_index = max(index + 1 for index, _, _ in self.keyframes)
        else:
            self.current_index = 0

    def record_keyframe(self):
        if self.data is None:
            messagebox.showerror("Error", "Load a MuJoCo model first.")
            return
        
        keyframe_name = self.keyframe_name_entry.get() or f"Keyframe {self.current_index}"
        keyframe = (keyframe_name, self.data.qpos.copy())
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
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
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
        if hasattr(self, 'selected_keyframe'):
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
    
    def update_keyframe(self):
        if hasattr(self, 'selected_keyframe'):
            index, keyframe_name, _ = self.keyframes[self.selected_keyframe]
            self.keyframes[self.selected_keyframe] = (index, keyframe_name, self.data.qpos.copy())
            
            # Update the keyframe file
            keyframe_path = os.path.join(self.key_frame_dir, f"{index}_{keyframe_name}.txt")
            with open(keyframe_path, "w") as f:
                for value in self.data.qpos:
                    f.write(f"{value}\n")

    def move_up(self):
        if hasattr(self, 'selected_keyframe') and self.selected_keyframe > 0:
            index = self.selected_keyframe
            self.keyframes[index - 1], self.keyframes[index] = self.keyframes[index], self.keyframes[index - 1]
            self.update_keyframe_listbox()
            self.keyframe_listbox.select_set(index - 1)
    
    def move_down(self):
        if hasattr(self, 'selected_keyframe') and self.selected_keyframe < len(self.keyframes) - 1:
            index = self.selected_keyframe
            self.keyframes[index + 1], self.keyframes[index] = self.keyframes[index], self.keyframes[index + 1]
            self.update_keyframe_listbox()
            self.keyframe_listbox.select_set(index + 1)
    
    def update_keyframe_listbox(self):
        self.keyframe_listbox.delete(0, tk.END)
        for name, _ in self.keyframes:
            self.keyframe_listbox.insert(tk.END, name)
    
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

if __name__ == "__main__":
    root = tk.Tk()
    app = MuJoCoApp(root)
    root.mainloop()
