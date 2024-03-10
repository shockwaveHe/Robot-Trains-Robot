from toddlerbot.actuation.dynamixel.dynamixel_control import *
from toddlerbot.actuation.mighty_zap.mighty_zap_control import *
from toddlerbot.actuation.sunny_sky.sunny_sky_control import *
from toddlerbot.sim import *


class MotorController(BaseSim):
    def __init__(
        self,
        robot: Optional[HumanoidRobot] = None,
    ):
        super().__init__(robot)

        self.dynamixel_config = DynamixelConfig(
            port="/dev/tty.usbserial-FT8ISUJY",
            kP=[100, 200, 200, 100, 200, 200],
            kI=[0, 0, 0, 0, 0, 0],
            kD=[100, 100, 100, 100, 100, 100],
            current_limit=[350, 350, 350, 350, 350, 350],
            init_pos=np.radians([135, 180, 180, 225, 180, 180]),
        )
        self.dynamixel_controller = DynamixelController(
            self.dynamixel_config, motor_ids=[7, 8, 9, 10, 11, 12]
        )

        self.sunny_sky_config = SunnySkyConfig(port="/dev/tty.usbmodem11101")
        self.sunny_sky_controller = SunnySkyController(
            self.sunny_sky_config, motor_ids=[1]
        )

        self.mighty_zap_config = MightyZapConfig(
            port="/dev/tty.usbserial-0001",
            init_pos=[1800, 1800],
        )
        self.mighty_zap_controller = MightyZapController(
            self.mighty_zap_config, motor_ids=[0, 1]
        )

    def get_joints_info(self, robot: HumanoidRobot):
        joints_info = {}
        for i in range(1, self.model.njnt):
            name = self.model.joint(i).name
            joints_info[name] = {
                "idx": self.model.joint(i).id,
                "type": self.model.joint(i).type,
                "lowerLimit": self.model.joint(i).range[0],
                "upperLimit": self.model.joint(i).range[1],
                "active": name in robot.config.act_params.keys(),
            }

        return joints_info

    def initialize_joint_angles(self, robot: HumanoidRobot):
        joint_angles = {}
        for name, info in robot.joints_info.items():
            if info["active"]:
                joint_angles[name] = self.data.joint(name).qpos.item()
        return joint_angles

    def set_joint_angles(self, joint_positions):
        self.dynamixel_controller.set_pos(joint_positions[:6])
        self.sunny_sky_controller.set_pos(joint_positions[6:7])
        self.mighty_zap_controller.set_pos(joint_positions[7:9])

    def read_state(self):
        dynamixel_state = self.dynamixel_controller.read_state()
        sunny_sky_state = self.sunny_sky_controller.read_state()
        mighty_zap_state = self.mighty_zap_controller.read_state()

        return {
            "dynamixel": dynamixel_state,
            "sunny_sky": sunny_sky_state,
            "mighty_zap": mighty_zap_state,
        }


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
