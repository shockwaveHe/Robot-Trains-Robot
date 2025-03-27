import numpy as np
from toddlerbot.sim import BaseSim
from toddlerbot.sim.robot import Robot
from multiprocessing import shared_memory
from toddlerbot.sim import Obs


class RealWorldFinetuning(BaseSim):
    def __init__(self, robot: Robot):
        super().__init__("real_world_finetuning")
        self.robot = robot

        shm_name = "force_shm"

        self.stopped = False

        try:
            print("Creating shared memory")
            self.arm_shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=112
            )
        except FileExistsError:
            print("Using existing shared memory")
            self.arm_shm = shared_memory.SharedMemory(
                name=shm_name, create=False, size=112
            )

        self.arm_force = np.ndarray(
            shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[16:40]
        )
        self.arm_torque = np.ndarray(
            shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[40:64]
        )
        self.arm_ee_pos = np.ndarray(
            shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[64:88]
        )
        self.arm_ee_vel = np.ndarray(
            shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[88:112]
        )
        self.Tr_base = np.array(
            [
                [0.8191521, 0.5735765, 0.0000000],
                [-0.5735765, 0.8191521, 0.0000000],
                [0.0000000, 0.0000000, 1.0000000],
            ]
        )

    def force_schedule(self):
        raise NotImplementedError

    def get_observation(self):
        time_curr = 0.0
        motor_pos = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        motor_vel = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        motor_tor = np.zeros(len(self.robot.motor_ordering), dtype=np.float32)
        obs = Obs(
            time=time_curr,
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_tor=motor_tor,
            ee_force=self.Tr_base @ self.arm_force,
            ee_torque=self.Tr_base @ self.arm_torque,
            arm_ee_pos=self.Tr_base @ self.arm_ee_pos,
            arm_ee_vel=self.Tr_base @ self.arm_ee_vel,
        )
        # import ipdb; ipdb.set_trace()
        return obs

    def step(self):
        pass

    def reset(self):
        return self.get_observation()

    def close(self):
        self.arm_shm.close()
        try:
            self.arm_shm.unlink()
        except FileNotFoundError:
            pass
        print("Shared memory unlinked")

    def set_motor_kps(self, motor_kps):
        pass

    def set_motor_target(self, motor_angles):
        pass
