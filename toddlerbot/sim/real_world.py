import queue
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from threading import Thread

from toddlerbot.actuation.dynamixel.dynamixel_control import *
from toddlerbot.actuation.mighty_zap.mighty_zap_control import *
from toddlerbot.actuation.sunny_sky.sunny_sky_control import *
from toddlerbot.sim import *


class RealWorld(BaseSim):
    def __init__(self, robot: Optional[HumanoidRobot] = None):
        super().__init__()
        self.name = "real_world"
        self.interp_method = "cubic"
        self.interp_freq = 1000
        self.negated_joint_names = [
            "left_hip_yaw",
            "right_hip_yaw",
            "right_hip_pitch",
            "left_ank_pitch",
        ]

        self.dynamixel_init_pos = np.radians([245, 180, 180, 287, 180, 180])
        self.dynamixel_config = DynamixelConfig(
            port="/dev/tty.usbserial-FT8ISUJY",
            kP=[800, 800, 800, 800, 800, 800],
            kI=[100, 100, 100, 100, 100, 100],
            kD=[400, 400, 400, 400, 400, 400],
            current_limit=[350, 350, 350, 350, 350, 350],
            init_pos=self.dynamixel_init_pos,
            gear_ratio=np.array([19 / 21, 1, 1, 19 / 21, 1, 1]),
        )

        self.sunny_sky_config = SunnySkyConfig(port="/dev/tty.usbmodem101")
        # Temorarily hard-coded joint range for SunnySky
        joint_range_dict = {1: (0, np.pi / 2), 2: (0, -np.pi / 2)}

        self.ankle2mighty_zap = {"left": [0, 1], "right": [2, 3]}
        self.last_mighty_zap_pos = {
            id: 0 for id in np.concatenate(list(self.ankle2mighty_zap.values()))
        }
        mighty_zap_init_pos = []
        for ids in self.ankle2mighty_zap.values():
            mighty_zap_init_pos += robot.ankle_ik([0] * len(ids))

        self.mighty_zap_config = MightyZapConfig(
            port="/dev/tty.usbserial-0001", init_pos=mighty_zap_init_pos
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_dynamixel = executor.submit(
                DynamixelController,
                self.dynamixel_config,
                motor_ids=sorted(list(robot.dynamixel_joint2id.values())),
            )
            future_sunny_sky = executor.submit(
                SunnySkyController,
                self.sunny_sky_config,
                joint_range_dict=joint_range_dict,
            )
            future_mighty_zap = executor.submit(
                MightyZapController,
                self.mighty_zap_config,
                motor_ids=sorted(list(robot.mighty_zap_joint2id.values())),
            )

            # Assign the results of futures to the attributes
            self.dynamixel_controller = future_dynamixel.result()
            self.sunny_sky_controller = future_sunny_sky.result()
            self.mighty_zap_controller = future_mighty_zap.result()

        self.queues = {
            "dynamixel": queue.Queue(),
            "sunny_sky": queue.Queue(),
            "mighty_zap": queue.Queue(),
        }

        self.threads = {
            "dynamixel": Thread(target=self.command_worker, args=("dynamixel",)),
            "sunny_sky": Thread(target=self.command_worker, args=("sunny_sky",)),
            "mighty_zap": Thread(target=self.command_worker, args=("mighty_zap",)),
        }

        self.time_start = time.time()

        for name, thread in self.threads.items():
            log("The command thread is initializing...", header=name.capitalize())
            thread.start()

        self.counter = 0

    def command_worker(self, actuator_type):
        while True:
            command = self.queues[actuator_type].get()
            if command is None:  # Use None as a signal to stop the thread
                break

            # Unpack the command
            command_type, args, future = command
            # Here, based on command_type, call the appropriate method
            # For simplicity, assuming a 'set_pos' command type
            if command_type == "set_pos":
                pos, delta_t = args
                if actuator_type == "dynamixel":
                    self.dynamixel_controller.set_pos(pos, delta_t=delta_t)
                elif actuator_type == "sunny_sky":
                    self.sunny_sky_controller.set_pos(pos, delta_t=delta_t)
                elif actuator_type == "mighty_zap":
                    self.mighty_zap_controller.set_pos(pos, delta_t=delta_t)

            elif command_type == "get_state":
                if actuator_type == "dynamixel":
                    self._get_dynamixel_state(future)
                elif actuator_type == "sunny_sky":
                    self._get_sunny_sky_state(future)
                elif actuator_type == "mighty_zap":
                    self._get_mighty_zap_state(future)

            self.queues[actuator_type].task_done()

    def _negate_joint_angles(self, joint_angles):
        negated_joint_angles = {}
        for name, angle in joint_angles.items():
            if name in self.negated_joint_names:
                negated_joint_angles[name] = -angle
            else:
                negated_joint_angles[name] = angle

        return negated_joint_angles

    def set_joint_angles(self, robot, joint_angles, interp=True, delta_t=0.01):
        # Directions are tuned to match the assembly of the robot.
        joint_angles = self._negate_joint_angles(joint_angles)
        joint_order = list(joint_angles.keys())
        pos = np.array(list(joint_angles.values()))

        def set_pos_helper(pos):
            dynamixel_pos = [
                pos[joint_order.index(k)] for k in robot.dynamixel_joint2id.keys()
            ]
            sunny_sky_pos = [
                pos[joint_order.index(k)] for k in robot.sunny_sky_joint2id.keys()
            ]
            ankle_pos = [
                pos[joint_order.index(k)] for k in robot.mighty_zap_joint2id.keys()
            ]

            mighty_zap_pos = []
            for ids in self.ankle2mighty_zap.values():
                mighty_zap_pos += robot.ankle_ik(np.array(ankle_pos)[ids])

            log(f"{round_floats(dynamixel_pos, 4)}", header="Dynamixel", level="debug")
            log(f"{round_floats(sunny_sky_pos, 4)}", header="SunnySky", level="debug")
            log(f"{mighty_zap_pos}", header="MightyZap", level="debug")

            # Execute set_pos calls in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(
                    self.dynamixel_controller.set_pos, dynamixel_pos, interp=False
                )
                executor.submit(
                    self.sunny_sky_controller.set_pos, sunny_sky_pos, interp=False
                )
                executor.submit(
                    self.mighty_zap_controller.set_pos, mighty_zap_pos, interp=False
                )

        if interp:
            pos_start = np.array(
                [state.pos for state in self.get_joint_state(robot).values()]
            )
            interpolate_pos(
                set_pos_helper,
                pos_start,
                pos,
                delta_t,
                self.interp_method,
                self.name,
                sleep_time=1 / self.interp_freq,
            )
        else:
            set_pos_helper(pos)

    def _get_dynamixel_state(self, future):
        dynamixel_state = self.dynamixel_controller.get_motor_state()
        future.set_result(dynamixel_state)

    def _get_sunny_sky_state(self, future):
        sunny_sky_state = self.sunny_sky_controller.get_motor_state()
        future.set_result(sunny_sky_state)

    def _get_mighty_zap_state(self, future):
        mighty_zap_state = self.mighty_zap_controller.get_motor_state()
        future.set_result(mighty_zap_state)

    # @profile
    def get_joint_state(self, robot: HumanoidRobot):
        future_dynamixel = Future()
        self.queues["dynamixel"].put(("get_state", None, future_dynamixel))
        future_sunny_sky = Future()
        self.queues["sunny_sky"].put(("get_state", None, future_sunny_sky))
        future_mighty_zap = Future()
        self.queues["mighty_zap"].put(("get_state", None, future_mighty_zap))

        # Retrieve results from futures once they are completed
        dynamixel_state = future_dynamixel.result()
        sunny_sky_state = future_sunny_sky.result()
        mighty_zap_state = future_mighty_zap.result()

        for ids in self.ankle2mighty_zap.values():
            ankle_pos = robot.ankle_fk(
                [mighty_zap_state[id].pos for id in ids],
                [self.last_mighty_zap_pos[id] for id in ids],
            )

            for i in range(len(ids)):
                mighty_zap_state[ids[i]].pos = ankle_pos[i]
                self.last_mighty_zap_pos[ids[i]] = ankle_pos[i]

        joint_state_dict = {}
        for name in robot.joints_info.keys():
            id = None
            if name in robot.dynamixel_joint2id:
                id = robot.dynamixel_joint2id[name]
                time = dynamixel_state[id].time
                pos = dynamixel_state[id].pos
            elif name in robot.sunny_sky_joint2id:
                id = robot.sunny_sky_joint2id[name]
                time = sunny_sky_state[id].time
                pos = sunny_sky_state[id].pos
            elif name in robot.mighty_zap_joint2id:
                id = robot.mighty_zap_joint2id[name]
                time = mighty_zap_state[id].time
                pos = mighty_zap_state[id].pos

            if id is not None:
                joint_state_dict[name] = JointState(
                    time=time, pos=-pos if name in self.negated_joint_names else pos
                )

        return joint_state_dict

    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        pass

    def get_torso_pose(self, robot: HumanoidRobot):
        # Placeholder
        return np.array(robot.com), np.eye(3)

    def get_zmp(self, robot: HumanoidRobot):
        pass

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

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        # Signal threads to stop
        for name in self.queues.keys():
            self.queues[name].put(None)
            self.threads[name].join()

        self.dynamixel_controller.close_motors()
        self.sunny_sky_controller.close_motors()
        self.mighty_zap_controller.close_motors()


if __name__ == "__main__":
    from toddlerbot.utils.vis_plot import *

    robot = HumanoidRobot("toddlerbot_legs")
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

        joint_state_dict = sim.get_joint_state(robot)
        for name, joint_state in joint_state_dict.items():
            if name not in time_seq_dict:
                time_seq_dict[name] = []
                joint_angle_dict[name] = []

            time_seq_dict[name].append(joint_state.time - time_start)
            joint_angle_dict[name].append(joint_state.pos)

        i += 1

    sim.close()
