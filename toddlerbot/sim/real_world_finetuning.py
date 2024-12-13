import numpy as np
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.mujoco_sim import MujocoSim
from toddlerbot.sim.robot import Robot
from multiprocessing import shared_memory
import struct
from toddlerbot.sim import Obs

class RealWorldFinetuning(RealWorld):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.robot = robot

        shm_name = 'force_shm'

        self.stopped = False

        try:
            self.arm_shm = shared_memory.SharedMemory(name=shm_name, create=True, size=88)
        except FileExistsError:
            self.arm_shm = shared_memory.SharedMemory(name=shm_name, create=False, size=88)
        
        self.arm_force = np.ndarray(shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[16:40])
        self.arm_torque = np.ndarray(shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[40:64])
        self.arm_ee_pos = np.ndarray(shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[64:88])

    
    def force_schedule(self):
        raise NotImplementedError
    
    def get_observation(self, retries = 0):
        obs = super().get_observation(retries)
        obs.ee_force = self.arm_force
        obs.ee_torque = self.arm_torque
        obs.arm_ee_pos = self.arm_ee_pos
    
    def step(self):        
        pass

    def is_done(self, obs: Obs):
        if np.abs(obs.ee_force[2]) < self.arm_healty_ee_force[0] or np.abs(obs.ee_force[2]) > self.arm_healty_ee_force[1]:
            print(f"Force Z of {obs.ee_force[2]} is out of range")
            return True
        if np.abs(obs.arm_ee_pos[2]) < self.arm_healty_ee_pos[0] or np.abs(obs.arm_ee_pos[2]) > self.arm_healty_ee_pos[1]:
            print(f"Position Z of {obs.arm_ee_pos[2]} is out of range")
            return True
        if np.abs(obs.ee_force[0]) > self.arm_healty_ee_force_xy[1] or np.abs(obs.ee_force[1]) > self.arm_healty_ee_force_xy[1]:
            print(f"Force XY of {obs.ee_force[:2]} is out of range")
            return True
        if np.abs(obs.ee_force[0]) < self.arm_healty_ee_force_xy[0] or np.abs(obs.ee_force[1]) < self.arm_healty_ee_force_xy[0]:
            print(f"Force XY of {obs.ee_force[:2]} is out of range")
            return True
        return False

    def reset(self):
        input("Press Enter to reset...")
        self.z_pos_delta = 0.2
        self.arm_shm.buf[8:16] = struct.pack('d', self.z_pos_delta)
        input("Press Enter to finish...")
        self.z_pos_delta = -0.2
        self.arm_shm.buf[8:16] = struct.pack('d', self.z_pos_delta)

    def close(self):
        self.stopped = True
        super().close()