from concurrent.futures import ThreadPoolExecutor

from scipy.optimize import root
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
        self.ankle_pos_last = [0, 0]

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
            kP=[400, 1200, 1200, 400, 1200, 1200],
            kI=[100, 100, 100, 100, 100, 100],
            kD=[200, 400, 400, 200, 400, 400],
            current_limit=[350, 350, 350, 350, 350, 350],
            init_pos=self.dynamixel_init_pos,
        )
        self.sunny_sky_joint2motor = {"left_knee": 1}
        self.sunny_sky_config = SunnySkyConfig(port="/dev/tty.usbmodem21201")

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
        dynamixel_pos = [joint_angles[k] for k in self.dynamixel_joint2motor.keys()]
        sunny_sky_pos = [joint_angles[k] for k in self.sunny_sky_joint2motor.keys()]
        ankle_pos = [joint_angles[k] for k in self.mighty_zap_joint2motor.keys()]
        mighty_zap_pos = self.ankle_ik(ankle_pos)

        log(f"{round_floats(dynamixel_pos, 4)}", header="Dynamixel", level="debug")
        log(f"{round_floats(sunny_sky_pos, 4)}", header="SunnySky", level="debug")
        log(f"{mighty_zap_pos}", header="MightyZap", level="debug")

        # Execute set_pos calls in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.dynamixel_controller.set_pos, dynamixel_pos)
            executor.submit(self.sunny_sky_controller.set_pos, sunny_sky_pos)
            executor.submit(self.mighty_zap_controller.set_pos, mighty_zap_pos)

    def ankle_fk(self, mighty_zap_pos):
        def objective_function(ankle_pos, target_pos):
            pos = self.ankle_ik(ankle_pos)
            error = np.array(pos) - np.array(target_pos)
            return error

        result = root(
            lambda x: objective_function(x, mighty_zap_pos),
            self.ankle_pos_last,
            method="hybr",
            options={"xtol": 1e-6},
        )

        if result.success:
            optimized_ankle_pos = result.x
            # log(
            #     f"Solved ankle position: {optimized_ankle_pos}",
            #     header="MightyZap",
            #     level="debug",
            # )
            return optimized_ankle_pos
        else:
            log(f"Solving ankle position failed", header="MightyZap", level="debug")
            return self.ankle_pos_last

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

        ankle_roll = -ankle_pos[0]
        ankle_pitch = -ankle_pos[1]
        R_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(ankle_roll), -np.sin(ankle_roll)],
                [0, np.sin(ankle_roll), np.cos(ankle_roll)],
            ]
        )
        R_pitch = np.array(
            [
                [np.cos(ankle_pitch), 0, np.sin(ankle_pitch)],
                [0, 1, 0],
                [-np.sin(ankle_pitch), 0, np.cos(ankle_pitch)],
            ]
        )
        R = np.dot(R_roll, R_pitch)
        n_hat = np.dot(R, nE)
        f1 = np.dot(R, f1E)
        f2 = np.dot(R, f2E)
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
        d1 = (d1_raw - mighty_zap_len) * 1e5
        d2 = (d2_raw - mighty_zap_len) * 1e5

        return np.array([d1, d2])

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

            ankle_pos = self.ankle_fk(
                [mighty_zap_state[0].pos, mighty_zap_state[1].pos]
            )
            for i in range(len(ankle_pos)):
                mighty_zap_state[i].pos = ankle_pos[i]
                self.ankle_pos_last[i] = ankle_pos[i]

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

    plot_joint_tracking(
        time_seq_dict,
        time_seq_ref,
        joint_angle_dict,
        joint_angle_ref_dict,
        robot.joint2type,
    )
