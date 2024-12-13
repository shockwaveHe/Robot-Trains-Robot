import time
import numpy as np
import serial
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import Robot
from multiprocessing import shared_memory
import threading
import struct
from toddlerbot.sim import Obs

class RealWorldFinetuning(RealWorld):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.robot = robot

        shm_name = 'force_shm'

        self.speed = 0.0
        self.force = 1.0
        self.z_pos_delta = 0.0
        self.stopped = False

        self.serial_thread = threading.Thread(target=self.serial_thread_func)
        self.serial_thread.start()
        try:
            self.arm_shm = shared_memory.SharedMemory(name=shm_name, create=True, size=88)
        except FileExistsError:
            self.arm_shm = shared_memory.SharedMemory(name=shm_name, create=False, size=88)
        self.arm_shm.buf[:8] = struct.pack('d', self.force)
        self.arm_shm.buf[8:16] = struct.pack('d', self.z_pos_delta)
        # TODO: verify the buffer size
        self.arm_force = np.ndarray(shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[16:40])
        self.arm_torque = np.ndarray(shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[40:64])
        self.arm_ee_pos = np.ndarray(shape=(3,), dtype=np.float64, buffer=self.arm_shm.buf[64:88])
        self.x_force_threshold = 0.5
        self.treadmill_speed_kp = 1
        self.arm_healty_ee_pos = np.array([0.0, 0.0])
        self.arm_healty_ee_force_z = np.array([1.0, 20.0])
        self.arm_healty_ee_force_xy = np.array([0.0, 5.0])

    def update_speed(self):
        if self.force > self.x_force_threshold:
            self.speed += self.treadmill_speed_kp * (self.force - self.x_force_threshold)
        elif self.force < -self.x_force_threshold:
            self.speed += self.treadmill_speed_kp * (self.force + self.x_force_threshold)
    
    def force_schedule(self):
        raise NotImplementedError
    
    def get_observation(self, retries = 0):
        obs = super().get_observation(retries)
        obs.ee_force = self.arm_force
        obs.ee_torque = self.arm_torque
        obs.arm_ee_pos = self.arm_ee_pos
    
    def step(self):
        super().step()
        
        self.arm_shm.buf[:8] = struct.pack('d', self.force)
        self.arm_shm.buf[8:16] = struct.pack('d', self.z_pos_delta)
        self.update_speed()

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

    def serial_thread_func(self):
        # Configure the serial connection
        ser = serial.Serial(
            port='/dev/ttyUSB0',     # Replace with your port name on Linux
            baudrate=9600,           # Baud rate
            parity=serial.PARITY_NONE,    # Parity (None)
            stopbits=serial.STOPBITS_ONE,  # Stop bits (1)
            bytesize=serial.EIGHTBITS,     # Data bits (8)
            timeout=0.1              # Set a timeout to prevent blocking
        )

        # Check if the port is open
        if ser.is_open:
            print(f"Connected to {ser.name}")

        try:
            while not self.stopped:
                # Get the current speed
                current_speed = self.speed

                # Prepare the data to send
                data_to_send = f"{int(current_speed)}\n".encode('utf-8')
                ser.write(data_to_send)  # Send the data

                # Optionally read a response (if the device sends back data)
                response = ser.readline()  # Read a line from the device
                # if response:
                #     print(f"Send: {data_to_send}, Received: {response}")

                # Sleep to prevent excessive CPU usage
                time.sleep(0.01)
        except Exception as e:
            print(f"Error in serial thread: {e}")
        finally:
            # Send zero speed before closing
            ser.write(b"0\n")
            ser.close()
            print("Serial connection closed.")

    def close(self):
        self.stopped = True
        self.serial_thread.join()
        self.arm_shm.close()
        self.arm_shm.unlink()
        super().close()