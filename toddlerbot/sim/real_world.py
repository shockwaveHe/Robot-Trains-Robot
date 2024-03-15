from concurrent.futures import ThreadPoolExecutor

from transforms3d.axangles import axangle2mat

from toddlerbot.actuation.dynamixel.dynamixel_control import *
from toddlerbot.actuation.mighty_zap.mighty_zap_control import *
from toddlerbot.actuation.sunny_sky.sunny_sky_control import *
from toddlerbot.sim import *
from toddlerbot.sim.robot import HumanoidRobot


class RealWorld(BaseSim):
    def __init__(self, robot: Optional[HumanoidRobot] = None):
        super().__init__(robot)
        self.robot = robot

        self.dynamixel_joint2motor = {
            "left_hip_yaw": 7,
            "left_hip_roll": 8,
            "left_hip_pitch": 9,
            "right_hip_yaw": 10,
            "right_hip_roll": 11,
            "right_hip_pitch": 12,
        }
        self.dynamixel_init_pos = np.radians([135, 180, 180, 225, 180, 180])
        self.dynamixel_config = DynamixelConfig(
            port="/dev/tty.usbserial-FT8ISUJY",
            kP=[100, 200, 200, 100, 200, 200],
            kI=[0, 0, 0, 0, 0, 0],
            kD=[100, 100, 100, 100, 100, 100],
            current_limit=[350, 350, 350, 350, 350, 350],
            init_pos=self.dynamixel_init_pos,
        )
        self.sunny_sky_joint2motor = {"left_knee": 1}
        self.sunny_sky_config = SunnySkyConfig(port="/dev/tty.usbmodem1201")

        self.mighty_zap_init_pos = self.ankle_ik([0, 0])
        self.mighty_zap_config = MightyZapConfig(
            port="/dev/tty.usbserial-0001", init_pos=self.mighty_zap_init_pos
        )
        self.mighty_zap_joint2motor = {"left_ank_roll": 0, "left_ank_pitch": 1}

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(
                lambda: setattr(
                    self,
                    "dynamixel_controller",
                    DynamixelController(
                        self.dynamixel_config,
                        motor_ids=list(self.dynamixel_joint2motor.values()),
                    ),
                )
            )
            executor.submit(
                lambda: setattr(
                    self,
                    "sunny_sky_controller",
                    SunnySkyController(
                        self.sunny_sky_config,
                        motor_ids=list(self.sunny_sky_joint2motor.values()),
                    ),
                )
            )
            executor.submit(
                lambda: setattr(
                    self,
                    "mighty_zap_controller",
                    MightyZapController(
                        self.mighty_zap_config,
                        motor_ids=list(self.mighty_zap_joint2motor.values()),
                    ),
                )
            )

    def set_joint_angles(self, robot, joint_angles):
        # Directions are tuned to match the assembly of the robot.
        dynamixel_pos = self.dynamixel_init_pos - np.array(
            [joint_angles[k] for k in self.dynamixel_joint2motor.keys()]
        )
        sunny_sky_pos = [joint_angles[k] for k in self.sunny_sky_joint2motor.keys()]
        ankle_pos = [-joint_angles[k] for k in self.mighty_zap_joint2motor.keys()]
        mighty_zap_pos = self.ankle_ik(ankle_pos)

        log(f"{round_floats(dynamixel_pos, 4)}", header="Dynamixel")
        log(f"{round_floats(sunny_sky_pos, 4)}", header="SunnySky")
        log(f"{mighty_zap_pos}", header="MightyZap")

        # Execute set_pos calls in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.dynamixel_controller.set_pos, dynamixel_pos)
            executor.submit(self.sunny_sky_controller.set_pos, sunny_sky_pos)
            executor.submit(self.mighty_zap_controller.set_pos, mighty_zap_pos)

    def ankle_fk(self, d1, d2):
        # Implemented based on page 3 of the following paper:
        # http://link.springer.com/10.1007/978-3-319-93188-3_49
        # Notations are from the paper.

        # TODO: Double check the implementation
        offsets = self.robot.config.offsets

        s1 = np.array(offsets["s1"])
        s2 = np.array([s1[0], -s1[1], s1[2]])
        f1E = np.array(offsets["f1E"])
        f2E = np.array([f1E[0], -f1E[1], f1E[2]])
        nE = np.array(offsets["nE"])
        r = offsets["r"]
        mighty_zap_len = offsets["mighty_zap_len"]

        d1_raw = (d1 * 1e-5) + mighty_zap_len
        d2_raw = (d2 * 1e-5) + mighty_zap_len

        n_hat = nE
        f1 = f1E
        f2 = f2E

        # Calculate the ankle position
        n_hat = n_hat / np.linalg.norm(n_hat)
        f1 = f1 / np.linalg.norm(f1)
        f2 = f2 / np.linalg.norm(f2)

        # Calculate the ankle position
        delta1 = d1_raw * n_hat
        delta2 = d2_raw * n_hat

        ankle_pos = np.array(
            [
                np.arctan2(
                    np.linalg.norm(np.cross(n_hat, delta1)), np.dot(n_hat, delta1)
                ),
                np.arctan2(
                    np.linalg.norm(np.cross(n_hat, delta2)), np.dot(n_hat, delta2)
                ),
            ]
        )

        return ankle_pos

    def ankle_ik(self, ankle_pos):
        # Implemented based on page 3 of the following paper:
        # http://link.springer.com/10.1007/978-3-319-93188-3_49
        # Notations are from the paper.

        offsets = self.robot.config.offsets

        s1 = np.array(offsets["s1"])
        s2 = np.array([s1[0], -s1[1], s1[2]])
        f1E = np.array(offsets["f1E"])
        f2E = np.array([f1E[0], -f1E[1], f1E[2]])
        nE = np.array(offsets["nE"])
        r = offsets["r"]
        mighty_zap_len = offsets["mighty_zap_len"]

        R = axangle2mat([1, 0, 0], ankle_pos[0]) @ axangle2mat([0, 1, 0], ankle_pos[1])
        n_hat = R @ nE
        f1 = R @ f1E
        f2 = R @ f2E
        delta1 = s1 - f1
        delta2 = s2 - f2

        d1_raw = np.sqrt(
            np.dot(n_hat, delta1) ** 2
            + (np.linalg.norm(np.cross(n_hat, delta1)) - r) ** 2
        )
        d2_raw = np.sqrt(
            np.dot(n_hat, delta2) ** 2
            + (np.linalg.norm(np.cross(n_hat, delta2)) - r) ** 2
        )
        d1 = int((d1_raw - mighty_zap_len) * 1e5)
        d2 = int((d2_raw - mighty_zap_len) * 1e5)

        return [d1, d2]

    def read_state(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks to read state from each controller
            future_dynamixel = executor.submit(self.dynamixel_controller.read_state)
            future_sunny_sky = executor.submit(self.sunny_sky_controller.read_state)
            future_mighty_zap = executor.submit(self.mighty_zap_controller.read_state)

            # Retrieve results from futures once they are completed
            dynamixel_state = future_dynamixel.result()
            sunny_sky_state = future_sunny_sky.result()
            mighty_zap_state = future_mighty_zap.result()

            for i, id in enumerate(dynamixel_state.keys()):
                dynamixel_state[id].pos = (
                    self.dynamixel_init_pos[i] - dynamixel_state[id].pos
                )

            ankle_pos = self.ankle_fk(mighty_zap_state[0].pos, mighty_zap_state[1].pos)
            mighty_zap_state[0].pos = ankle_pos[0]
            mighty_zap_state[1].pos = ankle_pos[1]

            return {
                "dynamixel": dynamixel_state,
                "sunny_sky": sunny_sky_state,
                "mighty_zap": mighty_zap_state,
            }

    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        pass

    def get_torso_pose(self, robot: HumanoidRobot):
        # Placeholder
        return np.array(robot.config.com), np.eye(3)

    def simulate(
        self,
        step_func: Optional[Callable] = None,
        step_params: Optional[Tuple] = None,
        vis_flags: Optional[List] = [],
        sleep_time: float = 0.0,
    ):

        try:
            while True:
                if step_func is not None:
                    if step_params is None:
                        step_func()
                    else:
                        step_params = step_func(*step_params)

                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.close()

    def close(self):
        self.dynamixel_controller.close_motors()
        self.sunny_sky_controller.close_motors()
        self.mighty_zap_controller.close_motors()


if __name__ == "__main__":
    from toddlerbot.utils.vis_plot import *

    robot = HumanoidRobot("base")
    sim = RealWorld(robot)

    joint_angles = robot.initialize_joint_angles()

    joint2type = {}
    for name in joint_angles.keys():
        if name in sim.dynamixel_joint2motor:
            joint2type[name] = "dynamixel"
        elif name in sim.sunny_sky_joint2motor:
            joint2type[name] = "sunny_sky"
        elif name in sim.mighty_zap_joint2motor:
            joint2type[name] = "mighty_zap"

    time_start = time.time()
    time_seq_ref = []
    time_seq_dict = {}
    joint_angle_ref_dict = {}
    joint_angle_dict = {}
    i = 0
    while i < 10:
        time_ref = time.time() - time_start
        time_seq_ref.append(time_ref)
        for name, angle in joint_angles.items():
            if name not in joint_angle_ref_dict:
                joint_angle_ref_dict[name] = []
            joint_angle_ref_dict[name].append(angle)

        sim.set_joint_angles(robot, joint_angles)

        state_dict = sim.read_state()
        for name in joint_angles.keys():
            if name in sim.dynamixel_joint2motor:
                motor_state = state_dict["dynamixel"][sim.dynamixel_joint2motor[name]]
            elif name in sim.sunny_sky_joint2motor:
                motor_state = state_dict["sunny_sky"][sim.sunny_sky_joint2motor[name]]
            elif name in sim.mighty_zap_joint2motor:
                motor_state = state_dict["mighty_zap"][sim.mighty_zap_joint2motor[name]]
            else:
                motor_state = None

            if motor_state is not None:
                if name not in time_seq_dict:
                    time_seq_dict[name] = []
                    joint_angle_dict[name] = []

                time_seq_dict[name].append(motor_state.time - time_start)
                joint_angle_dict[name].append(motor_state.pos)

        i += 1

    sim.close()

    x_list = []
    y_list = []
    legend_labels = []
    for name in time_seq_dict.keys():
        x_list.append(time_seq_dict[name])
        x_list.append(time_seq_ref)
        y_list.append(joint_angle_dict[name])
        y_list.append(joint_angle_ref_dict[name])
        legend_labels.append(name)
        legend_labels.append(name + "_ref")

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    time_suffix = "tracking"
    colors_dict = {
        "dynamixel": "cyan",
        "sunny_sky": "oldlace",
        "mighty_zap": "whitesmoke",
    }
    for i, ax in enumerate(axs.flat):
        ax.set_ylim([-np.pi / 2, np.pi / 2])
        ax.set_facecolor(colors_dict[joint2type[legend_labels[2 * i]]])
        plot_line_graph(
            y_list[2 * i : 2 * i + 2],
            x_list[2 * i : 2 * i + 2],
            title=f"{legend_labels[2*i]}",
            x_label="time (s)",
            y_label="position (rad)",
            save_config=True if i == len(axs.flat) - 1 else False,
            save_path="results/plots" if i == len(axs.flat) - 1 else None,
            time_suffix=time_suffix,
            ax=ax,
            legend_labels=legend_labels[2 * i : 2 * i + 2],
        )()

    time_str = time.strftime("%Y%m%d_%H%M%S")
    file_name_before = f"{legend_labels[-2]}_{time_suffix}"
    file_name_after = f"joint_angles_{time_suffix}_{time_str}"
    os.rename(
        os.path.join("results/plots", f"{file_name_before}.png"),
        os.path.join("results/plots", f"{file_name_after}.png"),
    )
    os.rename(
        os.path.join("results/plots", f"{file_name_before}_config.pkl"),
        os.path.join("results/plots", f"{file_name_after}_config.pkl"),
    )

    log(
        f"Renamed the files from {file_name_before} to {file_name_after}",
        header="RealWorld",
    )
