import argparse
import copy
import os
import pickle
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from functools import partial
from typing import List

import mujoco
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action, mat2quat
from toddlerbot.utils.misc_utils import find_latest_file_with_time_str

# This script is a GUI application that allows the user to interact with MuJoCo simulations in real-time
# and create keyframes for a given task. The keyframes can be tested and saved as a sequence
# for later use. The user can also visualize the keyframes and the sequence in MuJoCo.
# This script is highly inspired by the following code snippets: https://gist.github.com/JeanElsner/755d0feb49864ecadab4ef00fd49a22b

format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(2)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2, 0)
# Deprecated
# format.setColorSpace(format.sRGBColorSpace)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)


@dataclass
class Keyframe:
    """Dataclass for storing keyframe information."""

    name: str
    index: int
    motor_pos: np.ndarray
    joint_pos: np.ndarray | None = None
    qpos: np.ndarray | None = None


class Viewport(QOpenGLWindow):
    """Class for rendering the MuJoCo simulation in a Qt window."""

    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn, mutex) -> None:
        """Initializes an instance of the class with the given parameters and sets up a timer for periodic updates.

        Args:
            model: The model object to be used within the class.
            data: The data object associated with the model.
            cam: The camera object for capturing or processing images.
            opt: Options or configurations for the model or process.
            scn: The scene object related to the model or visualization.
            mutex: A threading lock to ensure thread-safe operations.

        Attributes:
            width (int): The width of the processing area, initialized to 0.
            height (int): The height of the processing area, initialized to 0.
            __last_pos: The last known position, initially set to None.
            runtime (collections.deque): A deque to store runtime data with a maximum length of 1000.
            timer (QTimer): A QTimer object set to trigger updates at approximately 60 frames per second.
        """
        super().__init__()

        self.model = model
        self.data = data

        self.cam = cam
        self.opt = opt
        self.scn = scn

        self.width = 0
        self.height = 0
        self.__last_pos = None

        self.mutex = mutex

        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1 / 60 * 1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def mousePressEvent(self, event):
        """Handles the mouse press event by updating the last known position of the mouse.

        Args:
            event (QMouseEvent): The mouse event containing information about the mouse press, including the position.
        """
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        """Handles mouse movement events to perform camera actions in a MuJoCo simulation.

        This method interprets mouse movements and translates them into camera actions
        such as move, rotate, or zoom based on the mouse button pressed. It updates the
        camera view accordingly using the MuJoCo API.

        Args:
            event (QMouseEvent): The mouse event containing information about the
                mouse position and button states.

        Returns:
            None
        """
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(
            self.model, action, dx / self.height, dy / self.height, self.scn, self.cam
        )
        self.__last_pos = pos

    def wheelEvent(self, event):
        """Handles the mouse wheel event to zoom the camera in or out in the MuJoCo simulation.

        This method is triggered when the mouse wheel is scrolled. It adjusts the camera's zoom level based on the scroll direction and magnitude.

        Args:
            event: The QWheelEvent object containing information about the wheel event, such as the scroll delta.
        """
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.0005 * event.angleDelta().y(),
            self.scn,
            self.cam,
        )

    def initializeGL(self):
        """Initializes the OpenGL context for rendering.

        This method sets up the OpenGL context using the MuJoCo library, which is necessary for rendering the simulation. It creates a new `MjrContext` object associated with the model and specifies the font scale to be used in the rendering context.

        Attributes:
            con (mujoco.MjrContext): The OpenGL context for rendering, initialized with the specified model and font scale.
        """
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        """Resize the OpenGL viewport to the specified width and height.

        This method updates the internal width and height attributes of the
        OpenGL context to match the new dimensions provided.

        Args:
            w (int): The new width of the OpenGL viewport.
            h (int): The new height of the OpenGL viewport.
        """
        self.width = w
        self.height = h

    def paintGL(self) -> None:
        """Renders the current scene using the MuJoCo physics engine and updates the runtime.

        This method updates the scene with the current model and data, renders it to a viewport,
        and calculates the time taken for rendering. The average runtime is then emitted through
        a signal for further processing or display.

        """
        t = time.time()

        with QMutexLocker(self.mutex):
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scn,
            )

        viewport = mujoco.MjrRect(0, 0, self.width * 2, self.height * 2)
        mujoco.mjr_render(viewport, self.scn, self.con)

        self.runtime.append(time.time() - t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):
    """Thread for updating the simulation in real-time and handling keyframe and trajectory testing."""

    updated = Signal()
    stateDataCurr = Signal(np.ndarray, np.ndarray, np.ndarray)
    trajDataCurr = Signal(list, list)

    def __init__(
        self, sim: MuJoCoSim, robot: Robot, mutex: QMutex, parent=None
    ) -> None:
        """Initializes the simulation control class with the given simulation, robot, and mutex.

        Args:
            sim (MuJoCoSim): The MuJoCo simulation instance to be controlled.
            robot (Robot): The robot instance containing default joint angles.
            mutex (QMutex): A mutex for synchronizing access to shared resources.
            parent (optional): The parent object, if any, for the class instance.

        Attributes:
            sim (MuJoCoSim): Stores the simulation instance.
            mutex (QMutex): Stores the mutex for synchronization.
            running (bool): Indicates if the simulation control is active.
            is_testing (bool): Indicates if the system is in testing mode.
            update_joint_angles_requested (bool): Flag to request joint angle updates.
            joint_angles_to_update (list): List of joint angles to update.
            update_qpos_requested (bool): Flag to request qpos updates.
            qpos_to_update (list): List of qpos values to update.
            keyframe_test_counter (int): Counter for keyframe testing.
            keyframe_test_dt (int): Time delta for keyframe testing.
            traj_test_counter (int): Counter for trajectory testing.
            action_traj (optional): Stores the action trajectory for testing.
            traj_test_dt (int): Time delta for trajectory testing.
            traj_physics_enabled (bool): Indicates if trajectory physics is enabled.

        """
        super().__init__(parent)
        self.sim = sim
        self.mutex = mutex
        self.running = True
        self.is_testing = False

        self.update_joint_angles_requested = False
        self.joint_angles_to_update = robot.default_joint_angles.copy()

        self.update_qpos_requested = False
        self.qpos_to_update = sim.model.qpos0.copy()

        self.keyframe_test_counter = -1
        self.keyframe_test_dt = 0

        self.traj_test_counter = -1
        self.action_traj = None
        self.traj_test_dt = 0
        self.traj_physics_enabled = False

    @Slot()
    def update_joint_angles(self, joint_angles_to_update):
        """Updates the joint angles for the robotic system.

        This method sets a flag indicating that a joint angle update has been requested and stores a copy of the new joint angles to be updated. It also prints a confirmation message to the console.

        Args:
            joint_angles_to_update (list or array-like): A list or array containing the new joint angles to be updated.
        """
        self.update_joint_angles_requested = True
        self.joint_angles_to_update = joint_angles_to_update.copy()

        print("Joint angles update requested!")

    @Slot()
    def update_qpos(self, qpos):
        """Request an update to the current position configuration.

        This method sets a flag indicating that an update to the position
        configuration (`qpos`) is requested. It stores a copy of the new
        position configuration to be updated later.

        Args:
            qpos (list or array-like): The new position configuration to be
                updated. It should be a list or array-like structure containing
                the desired position values.
        """
        self.update_qpos_requested = True
        self.qpos_to_update = qpos.copy()

        print("Qpos update requested!")

    @Slot()
    def request_feet_on_ground(self):
        """Aligns the torso of the simulated model with the ground based on foot positions.

        This method checks the z-coordinates of the left and right foot to determine which foot is closer to the ground. It then adjusts the torso's position and orientation in the simulation to align with the selected foot, ensuring the model's feet are on the ground. The simulation is updated with the new torso transformation.

        If the `is_testing` attribute is set to True, the function does nothing.

        Attributes:
            is_testing (bool): A flag indicating whether the function should perform its operations.
            sim (object): The simulation object containing methods and data for body transformations.
            mutex (QMutex): A mutex for thread-safe operations on the simulation data.

        """
        if not self.is_testing:
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

            with QMutexLocker(self.mutex):
                # Update the simulation with the new torso position and orientation
                self.sim.data.qpos[:3] = aligned_torso_transform[
                    :3, 3
                ]  # Update position
                self.sim.data.qpos[3:7] = mat2quat(aligned_torso_transform[:3, :3])
                self.sim.forward()

            print("Feet aligned with the ground!")

    @Slot()
    def request_state_data(self):
        """Retrieve and emit current state data from the simulation.

        This method fetches the current motor and joint angles from the simulation
        environment, along with the position data (`qpos`). It then emits these
        values using the `stateDataCurr` signal. This function is only executed
        when the system is not in testing mode.

        """
        if not self.is_testing:
            """Retrieve joint angles from the simulation and emit the signal."""
            motor_angles = self.sim.get_motor_angles(type="array")
            joint_angles = self.sim.get_joint_angles(type="array")
            qpos = self.sim.data.qpos.copy()
            self.stateDataCurr.emit(motor_angles, joint_angles, qpos)  # Emit data

            print("State data requested!")

    @Slot()
    def request_keyframe_test(self, keyframe: Keyframe, dt: float):
        """Initiates a keyframe test by setting the simulation state and motor targets.

        This method sets the simulation's position and motor targets to those specified
        in the given keyframe and begins a test sequence if not already in progress.

        Args:
            keyframe (Keyframe): The keyframe containing the desired positions and motor targets.
            dt (float): The time duration for which the keyframe test should run.

        """
        if not self.is_testing:
            with QMutexLocker(self.mutex):
                self.sim.data.qpos = keyframe.qpos.copy()
                self.sim.forward()
                self.sim.set_motor_target(keyframe.motor_pos.copy())

            self.keyframe_test_dt = dt
            self.keyframe_test_counter = 0

            self.is_testing = True
            print("Keyframe test started!")

    @Slot()
    def request_trajectory_test(
        self,
        qpos_start: np.ndarray,
        action_traj: List[np.ndarray],
        dt: float,
        physics_enabled: bool,
    ):
        """Initiates a trajectory test by setting the initial conditions and parameters.

        This function sets up a trajectory test by initializing the simulation state with a given starting position, action trajectory, time step, and physics settings. It prepares the system for testing by resetting relevant counters and flags.

        Args:
            qpos_start (np.ndarray): The starting joint positions for the simulation.
            action_traj (List[np.ndarray]): A list of action vectors representing the trajectory to be tested.
            dt (float): The time step duration for each action in the trajectory.
            physics_enabled (bool): A flag indicating whether physics should be enabled during the trajectory test.

        """
        if not self.is_testing:
            with QMutexLocker(self.mutex):
                self.sim.data.qpos = qpos_start.copy()
                self.sim.forward()

            self.action_traj = action_traj
            self.traj_test_dt = dt
            self.traj_physics_enabled = physics_enabled
            self.traj_test_counter = 0

            print("Trajectory test started!")
            print(f"Trajectory length: {len(action_traj)}")

            self.ee_traj = []
            self.root_traj = []

            self.is_testing = True

    def run(self) -> None:
        """Executes the main loop for updating simulation states and handling various test scenarios.

        This method continuously runs while `self.running` is True, performing updates to the simulation
        based on requested state changes or test conditions. It handles updating the simulation's
        position (`qpos`), joint angles, keyframe tests, and trajectory tests. The method ensures
        thread safety using `QMutexLocker` and emits signals to notify when updates are complete.

        Attributes:
            self.running (bool): Flag to control the loop execution.
            self.update_qpos_requested (bool): Indicates if a position update is requested.
            self.update_joint_angles_requested (bool): Indicates if a joint angles update is requested.
            self.keyframe_test_counter (int): Counter for keyframe testing.
            self.traj_test_counter (int): Counter for trajectory testing.
            self.traj_physics_enabled (bool): Flag to enable physics during trajectory testing.
            self.action_traj (list): List of actions for trajectory testing.
            self.ee_traj (list): List to store end-effector trajectory data.
            self.root_traj (list): List to store root position trajectory data.
            self.keyframe_test_dt (float): Time delta for keyframe testing.
            self.traj_test_dt (float): Time delta for trajectory testing.
        """
        while self.running:
            if self.update_qpos_requested:
                with QMutexLocker(self.mutex):
                    self.sim.data.qpos = self.qpos_to_update.copy()
                    self.sim.forward()

                self.update_qpos_requested = False
                self.keyframe_test_counter = -1
                self.updated.emit()

            elif self.update_joint_angles_requested:
                joint_angles = self.sim.get_joint_angles()
                joint_angles.update(self.joint_angles_to_update)

                with QMutexLocker(self.mutex):
                    self.sim.set_joint_angles(joint_angles)
                    self.sim.forward()

                self.update_joint_angles_requested = False
                self.keyframe_test_counter = -1
                self.updated.emit()  # Notify UI that update is complete

            elif self.keyframe_test_counter >= 0 and self.keyframe_test_counter <= 100:
                if self.keyframe_test_counter == 100:
                    self.keyframe_test_counter += 1
                    self.is_testing = False
                    continue

                with QMutexLocker(self.mutex):
                    self.sim.step()

                self.keyframe_test_counter += 1
                self.msleep(int(self.keyframe_test_dt * 1000))

            elif self.traj_test_counter >= 0 and self.traj_test_counter <= len(
                self.action_traj
            ):
                if self.traj_test_counter == len(self.action_traj):
                    self.trajDataCurr.emit(self.ee_traj, self.root_traj)
                    self.traj_test_counter += 1
                    self.is_testing = False
                    continue

                t1 = time.time()

                with QMutexLocker(self.mutex):
                    if self.traj_physics_enabled:
                        self.sim.set_motor_target(
                            self.action_traj[self.traj_test_counter]
                        )
                        self.sim.step()
                    else:
                        self.sim.set_motor_angles(
                            self.action_traj[self.traj_test_counter]
                        )
                        self.sim.forward()

                ee_pose_combined = []
                for side in ["left", "right"]:
                    ee_pos = self.sim.data.site(f"{side}_ee_center").xpos.copy()
                    ee_quat = mat2quat(
                        self.sim.data.site(f"{side}_ee_center")
                        .xmat.reshape(3, 3)
                        .copy()
                    )
                    ee_pose_combined.extend(ee_pos)
                    ee_pose_combined.extend(ee_quat)

                self.ee_traj.append(np.array(ee_pose_combined, dtype=np.float32))
                self.root_traj.append(self.sim.data.qpos[:7])
                t2 = time.time()
                self.traj_test_counter += 1
                time_left = self.traj_test_dt - (t2 - t1)
                if time_left > 0:
                    self.msleep(int(time_left * 1000))

    def stop(self):
        """Stops the execution of the current process.

        Sets the running state to False and waits for the process to terminate.
        """
        self.running = False
        self.wait()


class MuJoCoApp(QMainWindow):
    """Main application window for interacting with MuJoCo simulations and creating keyframes."""

    def __init__(self, sim: MuJoCoSim, robot: Robot, task_name: str, run_name: str):
        """Initializes the class with simulation, robot, and task details, setting up directories and loading data.

        Args:
            sim (MuJoCoSim): The simulation environment instance.
            robot (Robot): The robot instance to be used in the simulation.
            task_name (str): The name of the task to be performed.
            run_name (str): The name of the run, used for directory and file management.

        Attributes:
            sim (MuJoCoSim): Stores the simulation environment instance.
            robot (Robot): Stores the robot instance.
            task_name (str): Stores the task name.
            result_dir (str): Directory path for storing results.
            data_path (str): Path to the data file for the task.
            mirror_joint_signs (dict): Dictionary mapping joint names to their mirror signs.
            paused (bool): Indicates if the simulation is paused.
            slider_columns (int): Number of columns for sliders in the UI.
            qpos_offset (int): Offset for the position of the robot.
            model: The model of the simulation.
            data: The data of the simulation.
            cam: The camera instance for the simulation.
            opt: Options for the MuJoCo visualization.
            scn: The scene for the MuJoCo visualization.
            mutex (QMutex): Mutex for thread safety.
            viewport (Viewport): The viewport for rendering the simulation.
            sim_thread (UpdateSimThread): Thread for updating the simulation.

        """
        super().__init__()

        self.sim = sim
        self.robot = robot
        self.task_name = task_name

        if run_name == task_name:
            time_str = time.strftime("%Y%m%d_%H%M%S")
            self.result_dir = os.path.join(
                "results", f"{robot.name}_keyframe_{sim.name}_{time_str}"
            )
            os.makedirs(self.result_dir, exist_ok=True)
            self.data_path = os.path.join(self.result_dir, f"{task_name}.pkl")
            shutil.copy2(os.path.join("motion", f"{task_name}.pkl"), self.data_path)

        elif len(run_name) > 0:
            self.data_path = find_latest_file_with_time_str(
                os.path.join("results", run_name), task_name
            )
            if self.data_path is None:
                self.data_path = os.path.join("results", run_name, f"{task_name}.pkl")

            self.result_dir = os.path.dirname(self.data_path)
        else:
            self.data_path = ""
            time_str = time.strftime("%Y%m%d_%H%M%S")
            self.result_dir = os.path.join(
                "results", f"{robot.name}_keyframe_{sim.name}_{time_str}"
            )
            os.makedirs(self.result_dir, exist_ok=True)

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

        self.model = sim.model
        self.data = sim.data
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)

        self.mutex = QMutex()

        self.viewport = Viewport(
            sim.model, sim.data, self.cam, self.opt, self.scn, self.mutex
        )
        self.viewport.updateRuntime.connect(self.show_runtime)

        self.sim_thread = UpdateSimThread(sim, robot, self.mutex, self)
        self.sim_thread.start()

        self.sim_thread.stateDataCurr.connect(self.update_keyframe_with_signal)
        self.sim_thread.trajDataCurr.connect(self.update_traj_with_signal)

        self.setup_ui()
        self.load_data()

    def create_free_camera(self):
        """Creates and configures a free camera for the MuJoCo simulation environment.

        This function initializes a free camera, sets its type, and configures its
        position to focus on the median position of all geometries in the simulation.
        The camera's distance and elevation are also set to provide a broad view of
        the environment.

        Returns:
            mujoco.MjvCamera: A configured free camera object.
        """
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        for i in range(3):
            cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
        cam.distance = self.model.stat.extent * 2
        cam.elevation = -45
        return cam

    @Slot(float)
    def show_runtime(self, fps: float):
        """Displays the average runtime and simulation time in the status bar.

        This method updates the status bar with the average runtime in milliseconds
        calculated from the frames per second (fps) and the simulation time in milliseconds
        from the data attribute.

        Args:
            fps (float): The frames per second value used to calculate the average runtime.
        """
        self.statusBar().showMessage(
            f"Average runtime: {fps * 1000:.0f}ms\t\
                                        Simulation time: {self.data.time * 1000:.0f}ms"
        )

    def setup_ui(self):
        """Sets up the user interface for the application, including buttons, checkboxes, entry fields, and sliders for managing keyframes, sequences, and joint configurations.

        The UI is organized into a vertical layout containing a grid of buttons for keyframe and sequence operations, checkboxes for toggling options, entry fields for inputting parameters, and a horizontal layout for displaying keyframe and sequence lists alongside joint sliders.

        The function connects various UI elements to their respective event handlers to facilitate user interaction with the application.
        """
        layout = QVBoxLayout()
        # Top button grid
        grid_layout = QGridLayout()
        # Buttons
        add_button = QPushButton("Add Keyframe")
        add_button.clicked.connect(self.add_keyframe)
        grid_layout.addWidget(add_button, 0, 0)

        remove_button = QPushButton("Remove Keyframe")
        remove_button.clicked.connect(self.remove_keyframe)
        grid_layout.addWidget(remove_button, 0, 1)

        # load_button = QPushButton("Load Keyframe")
        # load_button.clicked.connect(self.load_keyframe)
        # grid_layout.addWidget(load_button, 0, 2)

        update_button = QPushButton("Update Keyframe")
        update_button.clicked.connect(self.update_keyframe)
        grid_layout.addWidget(update_button, 0, 2)

        test_button = QPushButton("Test Keyframe")
        test_button.clicked.connect(self.test_keyframe)
        grid_layout.addWidget(test_button, 0, 3)

        ground_button = QPushButton("Put Feet on Ground")
        ground_button.clicked.connect(self.put_feet_on_ground)
        grid_layout.addWidget(ground_button, 0, 4)

        # Mirror & Reverse Mirror Checkboxes
        self.mirror_checked = QCheckBox("Mirror")
        self.mirror_checked.setChecked(True)
        self.mirror_checked.toggled.connect(self.on_mirror_checked)
        grid_layout.addWidget(self.mirror_checked, 0, 5)

        self.rev_mirror_checked = QCheckBox("Rev. Mirror")
        self.rev_mirror_checked.toggled.connect(self.on_rev_mirror_checked)
        grid_layout.addWidget(self.rev_mirror_checked, 0, 6)

        # Sequence Buttons
        add_to_sequence_button = QPushButton("Add to Sequence")
        add_to_sequence_button.clicked.connect(self.add_to_sequence)
        grid_layout.addWidget(add_to_sequence_button, 1, 0)

        remove_from_sequence_button = QPushButton("Remove from Sequence")
        remove_from_sequence_button.clicked.connect(self.remove_from_sequence)
        grid_layout.addWidget(remove_from_sequence_button, 1, 1)

        update_arrival_time_button = QPushButton("Update Arrival Time")
        update_arrival_time_button.clicked.connect(self.update_arrival_time)
        grid_layout.addWidget(update_arrival_time_button, 1, 2)

        move_up_button = QPushButton("Move Up")
        move_up_button.clicked.connect(self.move_up)
        grid_layout.addWidget(move_up_button, 1, 3)

        move_down_button = QPushButton("Move Down")
        move_down_button.clicked.connect(self.move_down)
        grid_layout.addWidget(move_down_button, 1, 4)

        test_trajectory_button = QPushButton("Display Trajectory")
        test_trajectory_button.clicked.connect(self.test_trajectory)
        grid_layout.addWidget(test_trajectory_button, 1, 5)

        # Physics Toggle
        self.physics_enabled = QCheckBox("Enable Physics")
        self.physics_enabled.setChecked(True)
        grid_layout.addWidget(self.physics_enabled, 1, 6)

        # Entry Fields (Arrival Time, Delta Time, Motion Name)
        arrival_time_label = QLabel("Arrival Time:")
        grid_layout.addWidget(arrival_time_label, 2, 0)
        self.arrival_time_entry = QLineEdit("0")
        grid_layout.addWidget(self.arrival_time_entry, 2, 1)

        dt_label = QLabel("Interpolation Delta Time:")
        grid_layout.addWidget(dt_label, 2, 2)
        self.dt_entry = QLineEdit("0.02")
        grid_layout.addWidget(self.dt_entry, 2, 3)

        motion_name_label = QLabel("Motion Name:")
        grid_layout.addWidget(motion_name_label, 2, 4)
        self.motion_name_entry = QLineEdit(self.task_name)
        grid_layout.addWidget(self.motion_name_entry, 2, 5)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_data)
        grid_layout.addWidget(save_button, 2, 6)

        button_frame = QWidget()
        button_frame.setLayout(grid_layout)
        layout.addWidget(button_frame)

        # Horizontal Layout for Keyframe & Sequence Lists
        hbox_layout = QHBoxLayout()

        viewport_container = QWidget.createWindowContainer(self.viewport)
        hbox_layout.addWidget(viewport_container, stretch=3)

        vbox_layout = QVBoxLayout()

        keyframe_label = QLabel("Keyframes:")
        vbox_layout.addWidget(keyframe_label)

        self.keyframe_listbox = QListWidget()
        self.keyframe_listbox.itemSelectionChanged.connect(self.on_keyframe_select)
        vbox_layout.addWidget(self.keyframe_listbox, stretch=1)

        sequence_label = QLabel("Sequence:")
        vbox_layout.addWidget(sequence_label)

        self.sequence_listbox = QListWidget()
        self.sequence_listbox.itemSelectionChanged.connect(self.on_sequence_select)
        vbox_layout.addWidget(self.sequence_listbox, stretch=1)

        list_frame = QWidget()
        list_frame.setLayout(vbox_layout)
        hbox_layout.addWidget(list_frame, stretch=1)

        # Joint Sliders
        joint_sliders_layout = QGridLayout()
        slider_columns = 2  # Columns for sliders
        for i in range(slider_columns):
            joint_sliders_layout.setColumnStretch(i * 3, 1)
            joint_sliders_layout.setColumnStretch(i * 3 + 1, 2)
            joint_sliders_layout.setColumnStretch(i * 3 + 2, 1)

        self.joint_sliders = {}
        self.joint_labels = {}
        self.joint_scale = 1000

        reordered_list = []
        # Separate left and right joints
        for joint in robot.joint_ordering:
            if "left" in joint:
                right_joint = joint.replace("left", "right")
                assert right_joint in robot.joint_ordering, f"{right_joint} not found!"
                reordered_list.append(joint)
                reordered_list.append(right_joint)
            elif "right" not in joint:
                reordered_list.append(joint)

        num_sliders = 0
        for joint_name in reordered_list:
            joint_range = robot.joint_limits[joint_name]

            row = num_sliders // slider_columns
            col = num_sliders % slider_columns

            label = QLabel(joint_name)
            joint_sliders_layout.addWidget(label, row, col * 3)

            # Scale value label (to display the current value)
            # value_label = QLabel(text="0.00")
            # QLineEdit is suprisingly faster than QLabel
            value_label = QLineEdit("0.00")

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(joint_range[0] * self.joint_scale))
            slider.setMaximum(int(joint_range[1] * self.joint_scale))
            slider.setValue(
                int(robot.default_joint_angles[joint_name] * self.joint_scale)
            )
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setSingleStep(1)

            value_label.returnPressed.connect(
                partial(self.on_joint_label_change, joint_name)
            )
            joint_sliders_layout.addWidget(value_label, row, col * 3 + 2)
            slider.sliderReleased.connect(
                partial(self.on_joint_slider_release, joint_name)
            )
            joint_sliders_layout.addWidget(slider, row, col * 3 + 1)

            self.joint_sliders[joint_name] = slider
            self.joint_labels[joint_name] = value_label
            num_sliders += 1

        joint_sliders_frame = QWidget()
        joint_sliders_frame.setLayout(joint_sliders_layout)
        hbox_layout.addWidget(joint_sliders_frame, stretch=4)

        horizontal_frame = QWidget()
        horizontal_frame.setLayout(hbox_layout)
        layout.addWidget(horizontal_frame)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(1280, 720)

        # self.update_sliders_periodically()

    def add_keyframe(self):
        """Adds a new keyframe to the keyframe list, ensuring a unique index and name.

        If a keyframe is selected in the listbox, a deep copy of the selected keyframe is created. The new keyframe's name is updated if it contains "default", and a unique index is assigned to it. The new keyframe is then appended to the keyframe list and displayed in the listbox.

        Attributes:
            keyframe_listbox (QListWidget): The listbox widget displaying keyframes.
            keyframes (list): A list of keyframe objects.
            motion_name_entry (QLineEdit): The input field for the motion name.

        """
        idx = -1
        if self.keyframe_listbox.selectedItems():
            idx = self.keyframe_listbox.currentRow()

        new_keyframe = copy.deepcopy(self.keyframes[idx])
        motion_name = self.motion_name_entry.text()
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
        self.keyframe_listbox.addItem(f"{new_keyframe.name}_{new_keyframe.index}")

    def remove_keyframe(self):
        """Removes the currently selected keyframe and its associated sequence from the lists.

        If the object has a `selected_keyframe` attribute, this method removes the keyframe at the `selected_keyframe` index from the `keyframes` list. It also removes the corresponding entry from the `sequence_list` where the name matches the pattern "{keyframe.name}_{keyframe.index}". After removal, it updates the sequence and keyframe listboxes to reflect the changes.
        """
        if hasattr(self, "selected_keyframe"):
            keyframe = self.keyframes[self.selected_keyframe]
            for name, arrival_time in self.sequence_list:
                if name == f"{keyframe.name}_{keyframe.index}":
                    self.sequence_list.remove((name, arrival_time))
                    self.update_sequence_listbox()

            self.keyframes.pop(self.selected_keyframe)
            self.update_keyframe_listbox()

    def load_keyframe(self):
        """Loads the currently selected keyframe and updates the simulation thread's position.

        This method checks if the object has a `selected_keyframe` attribute. If it does, it retrieves the keyframe from the `keyframes` list using the `selected_keyframe` index and updates the simulation thread's position (`qpos`) with the keyframe's position data.

        """
        if hasattr(self, "selected_keyframe"):
            keyframe = self.keyframes[self.selected_keyframe]
            self.sim_thread.update_qpos(keyframe.qpos)

    def update_keyframe(self):
        """Updates the keyframe by requesting state data if a keyframe is selected.

        This method checks if the object has an attribute `selected_keyframe`. If it does, it triggers a request for state data from the simulation thread, which is expected to update or refresh the keyframe data accordingly.
        """
        if hasattr(self, "selected_keyframe"):
            self.sim_thread.request_state_data()

    @Slot(np.ndarray, np.ndarray, np.ndarray)
    def update_keyframe_with_signal(self, motor_angles, joint_angles, qpos):
        """Updates the selected keyframe with new motor angles, joint angles, and position data.

        Args:
            motor_angles (list or array-like): The new motor angles to update the keyframe with.
            joint_angles (list or array-like): The new joint angles to update the keyframe with.
            qpos (list or array-like): The new position data to update the keyframe with.

        """
        if hasattr(self, "selected_keyframe"):
            idx = self.selected_keyframe
            self.keyframes[idx].motor_pos = motor_angles.copy()
            self.keyframes[idx].joint_pos = joint_angles.copy()
            self.keyframes[idx].qpos = qpos.copy()
            print(f"Keyframe {idx} updated!")

    def test_keyframe(self):
        """Tests a selected keyframe by sending a request to the simulation thread.

        This method checks if a keyframe is selected and, if so, retrieves it from the
        keyframes list. It then parses the time delta from the user interface and sends
        a request to the simulation thread to test the keyframe with the specified time
        delta.

        """
        if hasattr(self, "selected_keyframe"):
            keyframe = self.keyframes[self.selected_keyframe]
            dt = float(self.dt_entry.text())
            self.sim_thread.request_keyframe_test(keyframe, dt)

    def put_feet_on_ground(self):
        """Requests the simulation thread to place the feet on the ground.

        This method sends a request to the simulation thread to ensure that the feet are positioned on the ground. It acts as an interface to the simulation's functionality for managing the feet's contact with the ground.
        """
        self.sim_thread.request_feet_on_ground()

    def on_mirror_checked(self, checked):
        """Sets the reverse mirror checkbox to unchecked if the mirror checkbox is checked.

        Args:
            checked (bool): The current state of the mirror checkbox. If True, the reverse mirror checkbox will be unchecked.
        """
        if checked:
            self.rev_mirror_checked.setChecked(False)

    def on_rev_mirror_checked(self, checked):
        """Handles the event when the reverse mirror checkbox is checked.

        If the reverse mirror checkbox is checked, this function will uncheck the
        standard mirror checkbox to ensure only one of the mirror options is selected
        at a time.

        Args:
            checked (bool): The state of the reverse mirror checkbox. True if checked,
            False otherwise.
        """
        if checked:
            self.mirror_checked.setChecked(False)

    def add_to_sequence(self):
        """Adds the selected keyframe to the sequence list with its arrival time.

        If a keyframe is selected in the keyframe listbox, this method retrieves the keyframe's name and index, constructs a unique keyframe name, and appends it along with the specified arrival time to the sequence list. It also updates the sequence listbox to display the new entry.

        Args:
            None

        Returns:
            None
        """
        if self.keyframe_listbox.selectedItems():
            selected_index = self.keyframe_listbox.currentRow()
            keyframe = self.keyframes[selected_index]
            keyframe_name = f"{keyframe.name}_{keyframe.index}"
            arrival_time = float(self.arrival_time_entry.text())

            self.sequence_list.append((keyframe_name, arrival_time))
            self.sequence_listbox.addItem(f"{keyframe_name}    t={arrival_time}")

    def remove_from_sequence(self):
        """Removes the currently selected item from the sequence list and updates the listbox display.

        If an item is selected in the sequence listbox, this method removes the item from the internal sequence list at the current row index and refreshes the listbox to reflect the change. If no item is selected, the method does nothing.
        """
        if self.sequence_listbox.selectedItems():
            selected_index = self.sequence_listbox.currentRow()
            self.sequence_list.pop(selected_index)
            self.update_sequence_listbox()

    def update_arrival_time(self):
        """Updates the arrival time for the selected sequence and adjusts subsequent sequences accordingly.

        This method checks if a sequence is selected and retrieves its current arrival time. It then attempts to parse the new arrival time from the user input. If the input is invalid, a warning is displayed, and the input is reset. If valid, the method updates the arrival times for the selected sequence and all subsequent sequences in the list, and refreshes the sequence list display.

        """
        if hasattr(self, "selected_sequence"):
            arrival_time = self.sequence_list[self.selected_sequence][1]
            arrival_time_text = self.arrival_time_entry.text()
            try:
                arrival_time_value = float(arrival_time_text)
            except ValueError:
                self.show_warning("The input value is not a valid number!")
                self.arrival_time_entry.setText("0")
                return

            for i in range(self.selected_sequence, len(self.sequence_list)):
                self.sequence_list[i] = (
                    self.sequence_list[i][0],
                    self.sequence_list[i][1] + arrival_time_value - arrival_time,
                )

            self.update_sequence_listbox()

    def move_up(self):
        """Moves the selected item up in the listbox, either for keyframes or sequences, if applicable.

        This method checks if the keyframe or sequence listbox is focused and has a selected item. If so, it swaps the selected item with the one above it in the list, updates the listbox display, and adjusts the current selection to reflect the new position.

        """
        if (
            self.keyframe_listbox.hasFocus()
            and self.keyframe_listbox.selectedItems()
            and hasattr(self, "selected_keyframe")
        ):
            index = self.selected_keyframe
            self.keyframes[index - 1], self.keyframes[index] = (
                self.keyframes[index],
                self.keyframes[index - 1],
            )
            self.update_keyframe_listbox()
            self.keyframe_listbox.setCurrentRow(index - 1)
            self.selected_keyframe = index - 1
        elif (
            self.sequence_listbox.hasFocus()
            and self.sequence_listbox.selectedItems()
            and hasattr(self, "selected_sequence")
        ):
            index = self.selected_sequence
            self.sequence_list[index - 1], self.sequence_list[index] = (
                self.sequence_list[index],
                self.sequence_list[index - 1],
            )
            self.update_sequence_listbox()
            self.sequence_listbox.setCurrentRow(index - 1)
            self.selected_sequence = index - 1

    def move_down(self):
        """Moves the selected item down in the list.

        This method moves the currently selected keyframe or sequence down by one position in its respective list, if possible. It updates the list and the selection to reflect the change.

        """
        if (
            self.keyframe_listbox.hasFocus()
            and self.keyframe_listbox.selectedItems()
            and hasattr(self, "selected_keyframe")
        ):
            index = self.selected_keyframe
            self.keyframes[index + 1], self.keyframes[index] = (
                self.keyframes[index],
                self.keyframes[index + 1],
            )
            self.update_keyframe_listbox()
            self.keyframe_listbox.setCurrentRow(index + 1)
            self.selected_keyframe = index + 1
        elif (
            self.sequence_listbox.hasFocus()
            and self.sequence_listbox.selectedItems()
            and hasattr(self, "selected_sequence")
        ):
            index = self.selected_sequence
            self.sequence_list[index + 1], self.sequence_list[index] = (
                self.sequence_list[index],
                self.sequence_list[index + 1],
            )
            self.update_sequence_listbox()
            self.sequence_listbox.setCurrentRow(index + 1)
            self.selected_sequence = index + 1

    def test_trajectory(self):
        """Tests the trajectory of a sequence of keyframes by interpolating motor positions over time and initiating a simulation.

        This method extracts motor positions and arrival times from a sequence of keyframes, checks for correct sorting of arrival times, and interpolates motor positions to create a trajectory. It then requests a simulation test of the trajectory starting from a specified keyframe.

        Raises:
            Warning: If the arrival times are not sorted in ascending order.

        """
        # Extract positions and arrival times from the sequence
        start_idx = 0
        if self.sequence_listbox.selectedItems():
            start_idx = self.sequence_listbox.currentRow()

        action_list = []
        qpos_list = []
        times = []
        for keyframe_name, arrival_time in self.sequence_list:
            for keyframe in self.keyframes:
                if keyframe_name in f"{keyframe.name}_{keyframe.index}":
                    action_list.append(keyframe.motor_pos)
                    qpos_list.append(keyframe.qpos)
                    times.append(arrival_time)
                    break

        if np.any(np.diff(times) <= 0):
            self.show_warning("The arrival times are not sorted correctly!")
            return

        qpos_start = qpos_list[start_idx]
        dt = float(self.dt_entry.text())
        enabled = self.physics_enabled.isChecked()

        action_arr = np.array(action_list)
        times = np.array(times) - times[0]

        self.traj_times = np.array([t for t in np.arange(0, times[-1], dt)])
        self.action_traj = []
        for t in self.traj_times:
            if t < times[-1]:
                motor_pos = interpolate_action(t, times, action_arr)
            else:
                motor_pos = action_arr[-1]

            self.action_traj.append(motor_pos)

        traj_start = int(np.searchsorted(self.traj_times, times[start_idx]))

        self.sim_thread.request_trajectory_test(
            qpos_start, self.action_traj[traj_start:], dt, enabled
        )

    @Slot()
    def update_traj_with_signal(self, ee_traj, root_traj):
        """Updates the end-effector and root trajectories with the provided signals.

        Args:
            ee_traj: The new trajectory data for the end-effector.
            root_traj: The new trajectory data for the root.

        Sets the instance variables `ee_traj` and `root_traj` to the provided trajectory data.
        """
        self.ee_traj = ee_traj
        self.root_traj = root_traj

    def save_data(self):
        """Saves the current motion data and keyframes to specified directories in pickle format.

        This method serializes the current state of keyframes, sequences, and trajectories into a dictionary and saves it as a pickle file in the results directory. It also saves the motion data to a separate file in the 'motion' directory, with a prompt to confirm overwriting if the file already exists.

        """
        result_dict = {}
        saved_keyframes = []
        for keyframe in self.keyframes:
            saved_keyframes.append(asdict(keyframe))

        result_dict["keyframes"] = saved_keyframes
        result_dict["sequence"] = self.sequence_list
        result_dict["time"] = self.traj_times
        result_dict["action_traj"] = self.action_traj
        result_dict["ee_traj"] = self.ee_traj
        result_dict["root_traj"] = self.root_traj

        time_str = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(self.result_dir, f"{self.task_name}_{time_str}.pkl")
        with open(result_path, "wb") as f:
            print(f"Saving the results to {result_path}")
            pickle.dump(result_dict, f)

        motion_name = self.motion_name_entry.text()
        motion_file_path = os.path.join("motion", f"{motion_name}.pkl")
        # Check if file exists before saving
        if os.path.exists(motion_file_path):
            reply = QMessageBox.question(
                self,
                "Overwrite Confirmation",
                "The file is already saved in the results folder."
                + f" Do you want to update {motion_name}.pkl in the motion/ directory?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return  # User canceled, do not save

        # Proceed with saving
        with open(motion_file_path, "wb") as f:
            print(f"Saving the results to {motion_file_path}")
            pickle.dump(result_dict, f)

    def load_data(self):
        """Loads and processes keyframe data from a specified file path.

        This method initializes and clears existing keyframe and trajectory data, then attempts to load new data from a file specified by `self.data_path`. If the file contains valid data, it updates the keyframes, sequence list, and trajectory information. If the data is incompatible with the current robot configuration, it stops the simulation and raises an error. If no keyframes are found, it initializes a default keyframe.

        Raises:
            ValueError: If the loaded data is incompatible with the current robot configuration.

        """
        self.keyframes = []
        self.sequence_list = []
        self.ee_traj = []
        self.root_traj = []
        self.keyframe_listbox.clear()

        keyframes = []
        if len(self.data_path) > 0:
            with open(self.data_path, "rb") as f:
                print(f"Loading inputs from {self.data_path}")
                data = pickle.load(f)

            if isinstance(data, dict):
                keyframes = [Keyframe(**k) for k in data.get("keyframes", [])]
                self.sequence_list = data.get("sequence", [])
                for i, (name, arrival_time) in enumerate(self.sequence_list):
                    self.sequence_list[i] = (name.replace(" ", "_"), arrival_time)

                self.traj_times = data.get("time", [])
                self.action_traj = data.get("action_traj", [])
                self.update_sequence_listbox()
            else:
                keyframes = data

            if len(keyframes[0].motor_pos) != self.robot.nu:
                self.sim_thread.stop()
                raise ValueError(
                    "This data is saved for a different robot! Consider changing the robot name."
                )

        if len(keyframes) == 0:
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
            self.keyframe_listbox.addItem("default_0")
        else:
            for i, keyframe in enumerate(keyframes):
                self.keyframes.append(keyframe)
                self.keyframe_listbox.addItem(f"{keyframe.name}_{keyframe.index}")

    def on_keyframe_select(self):
        """Handles the event when a keyframe is selected from the listbox.

        This method checks if there are any selected items in the keyframe listbox.
        If a keyframe is selected, it updates the `selected_keyframe` attribute with
        the index of the currently selected keyframe and calls the `load_keyframe`
        method to load the selected keyframe's data.
        """
        if self.keyframe_listbox.selectedItems():
            self.selected_keyframe = self.keyframe_listbox.currentRow()
            self.load_keyframe()

    def on_sequence_select(self):
        """Selects the current sequence from the listbox if any item is selected.

        This method checks if there are any selected items in the sequence listbox. If there are, it updates the `selected_sequence` attribute with the index of the currently selected item.
        """
        if self.sequence_listbox.selectedItems():
            self.selected_sequence = self.sequence_listbox.currentRow()

    def update_keyframe_listbox(self):
        """Updates the keyframe listbox with the current keyframes.

        This method clears the existing items in the keyframe listbox and repopulates it with the current keyframes. Each keyframe is displayed in the format "{keyframe.name}_{keyframe.index}".
        """
        self.keyframe_listbox.clear()
        for keyframe in self.keyframes:
            self.keyframe_listbox.addItem(f"{keyframe.name}_{keyframe.index}")

    def update_sequence_listbox(self):
        """Updates the sequence listbox with formatted items from the sequence list.

        This method clears the current contents of the sequence listbox and repopulates it with items from the `sequence_list`. Each item in the listbox is formatted to replace spaces in the name with underscores and includes the arrival time.
        """
        self.sequence_listbox.clear()
        for name, arrival_time in self.sequence_list:
            self.sequence_listbox.addItem(
                f"{name.replace(' ', '_')}    t={arrival_time}"
            )

    def update_joint_pos(self, joint_name, value):
        """Updates the position of a specified joint and its mirrored counterpart if applicable.

        This method updates the angle of a given joint and, if mirror options are enabled,
        also updates the angle of its mirrored joint. The mirrored joint's angle is calculated
        based on the mirror and reverse mirror settings.

        Args:
            joint_name (str): The name of the joint to update.
            value (float): The new angle value for the specified joint.

        Returns:
            dict: A dictionary containing the updated joint angles, including any mirrored joints.
        """
        joint_angles_to_update = {joint_name: value}

        mirror_checked, rev_mirror_checked = (
            self.mirror_checked.isChecked(),
            self.rev_mirror_checked.isChecked(),
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
                joint_angles_to_update[mirrored_joint_name] = (
                    mirror_checked * value * mirror_sign
                    - rev_mirror_checked * value * mirror_sign
                )

        self.sim_thread.update_joint_angles(joint_angles_to_update)

        return joint_angles_to_update

    def on_joint_slider_release(self, joint_name):
        """Handles the event when a joint slider is released, updating the joint's position and reflecting the changes in the UI.

        Args:
            joint_name (str): The name of the joint associated with the slider that was released.

        Updates the joint's position based on the slider's value, adjusts the corresponding label to display the new value, and updates any other affected joint sliders and labels to reflect their new positions.
        """
        slider = self.joint_sliders[joint_name]
        value_label = self.joint_labels[joint_name]

        slider_value = slider.value() / self.joint_scale
        joint_angles_to_update = self.update_joint_pos(joint_name, slider_value)

        value_label.setText(f"{slider_value:.2f}")

        for name, value in joint_angles_to_update.items():
            if name != joint_name:
                self.joint_labels[name].setText(f"{value:.2f}")
                self.joint_sliders[name].setValue(value * self.joint_scale)

    def on_joint_label_change(self, joint_name):
        """Handles changes to the joint label by validating and updating the joint's position.

        This method is triggered when the text in a joint's label is changed. It attempts to convert the text to a float and checks if the value is within the joint's allowable range. If the value is valid, it updates the joint's position and adjusts the corresponding slider and labels. If the value is invalid or out of range, it displays a warning and resets the label to "0.00".

        Args:
            joint_name (str): The name of the joint whose label has changed.

        """
        slider = self.joint_sliders[joint_name]
        value_label = self.joint_labels[joint_name]

        text = value_label.text()
        try:
            text_value = float(text)  # Convert input to float
        except ValueError:
            self.show_warning("The input value is not a valid number!")
            value_label.setText("0.00")
            return

        joint_range = self.robot.joint_limits[joint_name]  # Get joint range
        if not (joint_range[0] <= text_value <= joint_range[1]):
            self.show_warning(
                f"The input value {text_value} is out of range [{joint_range[0]:.2f}, {joint_range[1]:.2f}]!"
            )
            value_label.setText("0.00")
            return

        joint_angles_to_update = self.update_joint_pos(joint_name, text_value)

        slider.setValue(text_value * self.joint_scale)

        for name, value in joint_angles_to_update.items():
            if name != joint_name:
                self.joint_labels[name].setText(f"{value:.2f}")
                self.joint_sliders[name].setValue(value * self.joint_scale)

    def show_warning(self, message, title="Warning"):
        """Displays a warning message box with a specified message and title.

        Args:
            message (str): The warning message to be displayed in the message box.
            title (str, optional): The title of the message box window. Defaults to "Warning".
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo Keyframe Editor.")
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
        help="The name of the task.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="The path of the keyframes. If not provided, a new folder will be created."
        + "If the same as the task name, a copy of the data in the motion folder will be created in the results folder."
        + "Othewerwise, please make sure this is a valid folder name inside the results folder.",
    )
    args = parser.parse_args()

    app = QApplication()

    robot = Robot(args.robot)
    sim = MuJoCoSim(robot, vis_type="none")

    window = MuJoCoApp(sim, robot, args.task, args.run_name)
    window.show()
    app.exec()
    window.sim_thread.stop()
