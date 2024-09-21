import signal
import sys
import time

import numpy as np
import serial
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import Robot


class ResponseTester:
    def __init__(self, sim_name="mujoco", robot_name="sysID_XC330"):
        # Initial values for the PID controller
        self.kP = 0.0
        self.kI = 0.0
        self.kD = 0.0
        self.kFF1 = 0.0
        self.kFF2 = 0.0
        self.kVP = 0.0
        self.kVI = 0.0

        # Dynamixel motor ID
        self.id = 0

        # create sim and robot object
        self.robot_name = robot_name
        self.sim_name = sim_name
        self.robot = Robot(robot_name)
        if self.sim_name == "mujoco":
            self.sim = MuJoCoSim(self.robot, vis_type="view", fixed_base=True)
        elif self.sim_name == "real":
            self.sim = RealWorld(self.robot)
        else:
            raise ValueError(f"Invalid sim_name: {self.sim}")

        # Connect to the Dynamixel controller
        self.controller = self.sim.dynamixel_controller
        # self.connect_dynamixel()

    # def connect_dynamixel(self):
    #     # Create a controller object
    #     config = DynamixelConfig(
    #         port="/dev/tty.usbserial-FT7WBA3B",  # FT8ISUJY
    #         baudrate=2e6,
    #         control_mode=["extended_position"],
    #         kP=[self.kP],
    #         kI=[self.kI],
    #         kD=[self.kD],
    #         kFF2=[self.kFF2],
    #         kFF1=[self.kFF1],
    #         init_pos=[0.0],
    #     )
    #     self.controller = DynamixelController(config, motor_ids=[self.id])

    def reset_motors(self, reset_time=1.0, target_pos=-np.pi / 2):
        print("Resetting motors...")
        self.controller.enable_motors()
        self.controller.set_kp_kd(kp=2400, kd=2400)
        state_dict = self.controller.get_motor_state()
        position = state_dict[self.id].pos

        for i in range(100):
            curr_pos = position * (1 - i / 100) + target_pos * (i / 100)
            print(f"Setting position to {curr_pos}", end="\r", flush=True)
            self.controller.set_pos(pos=[curr_pos])
            tslp = reset_time / 100
            time.sleep(tslp)
        self.controller.set_pos(pos=[target_pos])

    # offset is unit(rad) so that we are looking at the unit step response
    def step_response(self, duration=3.0, offset=1):
        # Reset the motor to the initial position
        rest_pos = -np.pi / 2
        self.reset_motors(target_pos=rest_pos)

        print("Running step response...")
        self.controller.set_parameters(
            kp=self.kP,
            ki=self.kI,
            kd=self.kD,
            kff1=self.kFF1,
            kff2=self.kFF2,
            ids=[self.id],
        )

        response = []
        start_time = time.time()

        # Set up tqdm progress bar with percentage
        with tqdm(total=100, desc="Step Response Progress", unit="%") as pbar:
            while time.time() - start_time < duration:
                curr_time = time.time() - start_time
                if curr_time < 1.0:
                    cmd_pos = rest_pos
                else:
                    cmd_pos = rest_pos + offset

                self.controller.set_pos(pos=[cmd_pos])
                state_dict = self.controller.get_motor_state()
                response.append(
                    (curr_time, state_dict[self.id].pos - rest_pos, cmd_pos - rest_pos)
                )

                # Calculate the percentage of time completed
                progress_percentage = round((curr_time / duration) * 100, 1)

                # Update the tqdm progress bar
                pbar.update(progress_percentage - pbar.n)  # Update only the increment

        self.controller.disable_motors()

        print("Step response complete")
        return np.array(response)


class PIDControllerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ================== Response Runner ==================
        self.tester = ResponseTester()

        # ================== GUI Setup ==================

        # Set up the main window
        self.setWindowTitle("Dynamixel Tuner")
        self.setGeometry(300, 300, 800, 500)  # Adjust width to accommodate plot

        # Create the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(
            self.central_widget
        )  # Horizontal layout for sliders and plot

        # Create the left layout for labels and sliders
        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout)

        # Create a grid layout for the labels (for double column display)
        self.label_layout = QGridLayout()

        # Create labels to display the current values of Kp, Ki, Kd, Kff1, Kff2, KVP, and KVI
        self.kp_label = QLabel(f"Kp: {self.tester.kP}", self)
        self.kp_label.setStyleSheet("font-size: 16px;")  # Set font size to 16px

        self.ki_label = QLabel(f"Ki: {self.tester.kI}", self)
        self.ki_label.setStyleSheet("font-size: 16px;")  # Set font size to 16px

        self.kd_label = QLabel(f"Kd: {self.tester.kD}", self)
        self.kd_label.setStyleSheet("font-size: 16px;")

        self.kff1_label = QLabel(f"Kff1: {self.tester.kFF1}", self)
        self.kff1_label.setStyleSheet("font-size: 16px;")

        self.kff2_label = QLabel(f"Kff2: {self.tester.kFF2}", self)
        self.kff2_label.setStyleSheet("font-size: 16px;")

        self.kvp_label = QLabel(f"KVP: {self.tester.kVP}", self)
        self.kvp_label.setStyleSheet("font-size: 16px;")

        self.kvi_label = QLabel(f"KVI: {self.tester.kVI}", self)
        self.kvi_label.setStyleSheet("font-size: 16px;")

        # Add the labels to the grid layout for two-column display
        self.label_layout.addWidget(self.kp_label, 0, 0)
        self.label_layout.addWidget(self.ki_label, 0, 1)
        self.label_layout.addWidget(self.kd_label, 1, 0)
        self.label_layout.addWidget(self.kff1_label, 1, 1)
        self.label_layout.addWidget(self.kff2_label, 2, 0)
        self.label_layout.addWidget(self.kvp_label, 2, 1)
        self.label_layout.addWidget(self.kvi_label, 3, 0)

        self.left_layout.addLayout(self.label_layout)

        # Create a "Plot Response" button and add it to the left layout
        self.plot_button = QPushButton("Plot Response", self)
        self.plot_button.setStyleSheet("font-size: 18px;")  # Set button font size
        self.left_layout.addWidget(self.plot_button)

        # Connect the button's click signal to the callback function
        self.plot_button.clicked.connect(self.on_plot_response)

        # Create a progress bar below the "Plot Response" button
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.left_layout.addWidget(self.progress_bar)

        # Create sliders for Kp, Ki, Kd, Kff1, Kff2, KVP, and KVI
        self.kp_slider = self.create_slider()
        self.ki_slider = self.create_slider()
        self.kd_slider = self.create_slider()
        self.kff1_slider = self.create_slider()
        self.kff2_slider = self.create_slider()
        self.kvp_slider = self.create_slider()
        self.kvi_slider = self.create_slider()

        # Add the sliders to the left layout
        self.left_layout.addWidget(self.kp_slider)
        self.left_layout.addWidget(self.ki_slider)
        self.left_layout.addWidget(self.kd_slider)
        self.left_layout.addWidget(self.kff1_slider)
        self.left_layout.addWidget(self.kff2_slider)
        self.left_layout.addWidget(self.kvp_slider)
        self.left_layout.addWidget(self.kvi_slider)

        # Connect the sliders to the respective update functions
        self.kp_slider.valueChanged.connect(self.update_kp)
        self.ki_slider.valueChanged.connect(self.update_ki)
        self.kd_slider.valueChanged.connect(self.update_kd)
        self.kff1_slider.valueChanged.connect(self.update_kff1)
        self.kff2_slider.valueChanged.connect(self.update_kff2)
        self.kvp_slider.valueChanged.connect(self.update_kvp)
        self.kvi_slider.valueChanged.connect(self.update_kvi)

        # Create the plot window (right side)
        self.plot_canvas = PlotCanvas(self, width=5, height=4)
        self.main_layout.addWidget(self.plot_canvas)

    def on_plot_response(self):
        """Callback function for Plot Response button."""
        print("Plot Response button clicked")
        raw_response = self.tester.step_response()
        self.plot_canvas.update_plot(raw_response)

    def create_slider(self):
        """Creates a QSlider configured for the Kp, Ki, Kd, Kff1, Kff2, KVP, and KVI values"""
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 16383)  # Set the slider's range from 0 to 16383
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(
            1024
        )  # Optional: Set tick intervals for easier adjustment
        return slider

    def update_kp(self, value):
        """Update the label to reflect the current value of Kp"""
        self.kp_label.setText(f"Kp: {value}")
        self.tester.kP = value

    def update_ki(self, value):
        """Update the label to reflect the current value of Ki"""
        self.ki_label.setText(f"Ki: {value}")
        self.tester.kI = value

    def update_kd(self, value):
        """Update the label to reflect the current value of Kd"""
        self.kd_label.setText(f"Kd: {value}")
        self.tester.kD = value

    def update_kff1(self, value):
        """Update the label to reflect the current value of Kff1"""
        self.kff1_label.setText(f"Kff1: {value}")
        self.tester.kFF1 = value

    def update_kff2(self, value):
        """Update the label to reflect the current value of Kff2"""
        self.kff2_label.setText(f"Kff2: {value}")
        self.tester.kFF2 = value

    def update_kvp(self, value):
        """Update the label to reflect the current value of KVP"""
        self.kvp_label.setText(f"KVP: {value}")
        self.tester.kVP = value

    def update_kvi(self, value):
        """Update the label to reflect the current value of KVI"""
        self.kvi_label.setText(f"KVI: {value}")
        self.tester.kVI = value

    def closeEvent(self, event):
        """Handle the window close event and disable motors."""
        print("Disabling motors...")
        self.tester.controller.disable_motors()
        event.accept()  # Close the window

    def disable_motors_on_signal(self):
        """Disable motors when Ctrl-C is pressed."""
        print("Caught Ctrl-C, disabling motors...")
        self.tester.controller.disable_motors()
        QApplication.quit()  # Close the application safely


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

        self.plot_initial()

    def plot_initial(self):
        """Initial empty plot"""
        self.axes.set_title("Time Domain Response")
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel("Position (rad)")
        self.axes.grid()
        self.draw()

    def update_plot(self, data):
        """Update the plot with new data"""
        self.axes.clear()
        time = data[:, 0]  # First column is time
        position = data[:, 1]  # Second column is position
        ref = data[:, 2]
        self.axes.plot(time, position, "b-", label="Position (rad)")
        self.axes.plot(time, ref, "r-", label="Reference (rad)")
        self.axes.set_title("Step Response")
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel("Position (rad)")

        # Calculate min and max of the position data
        pos_min = min(position)
        pos_max = max(position)

        # Ensure the y-axis limit is at least [-0.05, 1.05]
        y_min = min(-0.05, pos_min - 0.05)  # Allow extra margin below
        y_max = max(1.05, pos_max + 0.05)  # Allow extra margin above

        # Set y-axis limits dynamically
        self.axes.set_ylim([y_min, y_max])

        self.axes.grid()
        self.draw()


def handle_ctrl_c(window):
    """Ensure the application can be killed with Ctrl+C and disable motors"""
    signal.signal(signal.SIGINT, lambda sig, frame: window.disable_motors_on_signal())


# Standard boilerplate to run the application
if __name__ == "__main__":
    # Create an instance of QApplication
    app = QApplication(sys.argv)

    # Create an instance of the PIDControllerWindow class
    window = PIDControllerWindow()
    window.show()

    # Handle Ctrl-C gracefully and pass the window instance
    handle_ctrl_c(window)

    # Use a QTimer to prevent the application from blocking Ctrl-C
    timer = QTimer()
    timer.start(500)  # 500 ms interval
    timer.timeout.connect(lambda: None)  # Prevents the event loop from blocking

    # Start the event loop
    sys.exit(app.exec())
