from transforms3d.axangles import axangle2mat

from toddlerbot.actuation.dynamixel.dynamixel_control import *
from toddlerbot.actuation.mighty_zap.mighty_zap_control import *
from toddlerbot.actuation.sunny_sky.sunny_sky_control import *
from toddlerbot.sim import *
from toddlerbot.sim.robot import HumanoidRobot


class MotorController(BaseSim):
    def __init__(
        self,
        robot: Optional[HumanoidRobot] = None,
    ):
        super().__init__(robot)

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
        self.dynamixel_controller = DynamixelController(
            self.dynamixel_config, motor_ids=list(self.dynamixel_joint2motor.values())
        )

        self.sunny_sky_joint2motor = {"left_knee": 0}
        self.sunny_sky_init_pos = [0]
        self.sunny_sky_config = SunnySkyConfig(
            port="/dev/tty.usbmodem11101", init_pos=self.sunny_sky_init_pos
        )
        self.sunny_sky_controller = SunnySkyController(
            self.sunny_sky_config, motor_ids=list(self.sunny_sky_joint2motor.values())
        )

        self.mighty_zap_joint2motor = {"left_ank_roll": 0, "left_ank_pitch": 1}
        self.mighty_zap_init_pos = self.ankle_ik([0, 0])
        self.mighty_zap_config = MightyZapConfig(
            port="/dev/tty.usbserial-0001", init_pos=self.mighty_zap_init_pos
        )
        self.mighty_zap_controller = MightyZapController(
            self.mighty_zap_config, motor_ids=list(self.mighty_zap_joint2motor.values())
        )

    def set_joint_angles(self, joint_angles):
        dynamixel_pos = [joint_angles[k] for k in self.dynamixel_joint2motor.keys()]
        sunny_sky_pos = [joint_angles[k] for k in self.sunny_sky_joint2motor.keys()]
        ankle_pos = [joint_angles[k] for k in self.mighty_zap_joint2motor.keys()]
        mighty_zap_pos = self.ankle_ik(ankle_pos)

        self.dynamixel_controller.set_pos(dynamixel_pos)
        self.sunny_sky_controller.set_pos(sunny_sky_pos)
        self.mighty_zap_controller.set_pos(mighty_zap_pos)

    def ankle_ik(self, ankle_pos):
        # Implemented based on page 3 of the following paper:
        # http://link.springer.com/10.1007/978-3-319-93188-3_49
        # Notations are from the paper.

        s1 = np.array(self.robot.config["offsets"]["s1"])
        s2 = np.array([s1[0], -s1[1], s1[2]])
        f1E = np.array(self.robot.config["offsets"]["f1E"])
        f2E = np.array([f1E[0], -f1E[1], f1E[2]])
        nE = np.array([self.robot.config["offsets"]["nE"]])
        r = self.robot.config["offsets"]["r"]
        mighty_zap_len = self.robot.config["offsets"]["mighty_zap_len"]

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
        d1 = (d1_raw - mighty_zap_len) * 1e5
        d2 = (d2_raw - mighty_zap_len) * 1e5

        return [d1, d2]

    def read_state(self):
        dynamixel_state = self.dynamixel_controller.read_state()
        sunny_sky_state = self.sunny_sky_controller.read_state()
        mighty_zap_state = self.mighty_zap_controller.read_state()

        return {
            "dynamixel": dynamixel_state,
            "sunny_sky": sunny_sky_state,
            "mighty_zap": mighty_zap_state,
        }

    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        pass

    def get_com(self, robot: HumanoidRobot):
        pass

    def get_zmp(self, robot: HumanoidRobot):
        pass

    def simulate(
        self,
        step_func: Optional[Callable] = None,
        step_params: Optional[Tuple] = None,
        vis_flags: Optional[List] = [],
        sleep_time: float = 0.0,
    ):
        pass


if __name__ == "__main__":
    motor_controller = MotorController()
    i = 0
    while i < 30:
        motor_controller.set_joint_angles(
            [*np.radians([150, 165, 165, 210, 165, 195]), np.pi / 4, 3000, 3000]
        )
        state_dict = motor_controller.read_state()
        # print(state_dict)
        # p_error = np.abs(state[0] - pos)
        # print(p_error)

        # if p_error.mean() < 0.01:
        #     break

        i += 1

    i = 0
    while i < 30:
        motor_controller.set_joint_angles(
            [
                *motor_controller.dynamixel_config.init_pos,
                0,
                *motor_controller.mighty_zap_config.init_pos,
            ]
        )
        state_dict = motor_controller.read_state()
        # print(state_dict)
        # p_error = np.abs(state[0] - pos)
        # print(p_error)

        # if p_error.mean() < 0.01:
        #     break

        i += 1
