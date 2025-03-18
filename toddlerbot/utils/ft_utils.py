import socket
import struct
import threading
import time
from typing import Tuple
import queue
import numpy as np

class NetFTException(Exception):
    pass

class NetFTSensor:
    RDT_PORT = 49152
    RDT_RECORD_SIZE = 36
    COMMAND_HEADER = 0x1234
    CMD_START_HIGH_SPEED_STREAMING = 0x0002
    CMD_STOP_STREAMING = 0x0000

    def __init__(self, ip_address: str = "192.168.2.1", 
                 counts_per_force: float = 1000000.0,
                 counts_per_torque: float = 1000000.0,
                 timeout: float = 1.0,
                 max_queue_size: int = 300):
        self.ip_address = ip_address
        self.counts_per_force = counts_per_force
        self.counts_per_torque = counts_per_torque
        self.timeout = timeout
        
        # Thread-safe data storage with timestamps
        self._data_queue = queue.Queue(maxsize=max_queue_size)
        self._last_smoothed_time = None  # For tracking smoothing window
        
        # Setup UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip_address, self.RDT_PORT))
        
        # Start streaming
        self._send_command(self.CMD_START_HIGH_SPEED_STREAMING)
        self._running = True
        self._receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receiver_thread.start()

        # Bias force and torque
        self.bias_period = 3
        self.bias_force = None
        self.bias_torque = None

        self.Tr_base = np.array([
            [0.8191521,  0.5735765,  0.0000000],
			[-0.5735765,  0.8191521,  0.0000000],
			[0.0000000,  0.0000000,  1.0000000]
        ])

        # self.Tr_base = np.eye(3)
        self.get_bias()

    def get_bias(self):
        input("Press Enter to start biasing...")
        start = time.time()
        bias_force, bias_torque = [], []
        while time.time() - start < self.bias_period:
            force, torque = self.get_smoothed_data()
            bias_force.append(force)
            bias_torque.append(torque)
            time.sleep(0.001)

        self.bias_force = np.array(bias_force).sum(axis=0) / len(bias_force)
        self.bias_torque = np.array(bias_torque).sum(axis=0) / len(bias_torque)
        print(f"Bias force: {self.bias_force} N, Bias torque: {self.bias_torque} Nm")

    def _send_command(self, command: int):
        """Send RDT command to sensor"""
        cmd_struct = struct.pack('!HHI', 
            self.COMMAND_HEADER,
            command,
            0  # Infinite samples
        )
        self.sock.send(cmd_struct)

    def _parse_rdt_packet(self, data: bytes) -> Tuple[float, float, float, float, float, float]:
        """Parse RDT data packet into force/torque values"""
        if len(data) != self.RDT_RECORD_SIZE:
            raise NetFTException(f"Invalid packet size: {len(data)} bytes")

        unpacked = struct.unpack('!IIIiiiiii', data)
        (_, _, status, fx, fy, fz, tx, ty, tz) = unpacked

        if status != 0:
            raise NetFTException(f"Sensor error: status code {status}")

        return (
            fx / self.counts_per_force,
            fy / self.counts_per_force,
            fz / self.counts_per_force,
            tx / self.counts_per_torque,
            ty / self.counts_per_torque,
            tz / self.counts_per_torque
        )

    def _receive_loop(self):
        """Main receive loop running in background thread"""
        while self._running:
            try:
                data, _ = self.sock.recvfrom(1024)
                ft_data = self._parse_rdt_packet(data)
                timestamp = time.monotonic()
                
                try:
                    # Store data with timestamp
                    self._data_queue.put((timestamp, *ft_data), block=False)
                except queue.Full:
                    # Maintain queue size by removing oldest entry
                    self._data_queue.get()
                    self._data_queue.put((timestamp, *ft_data), block=False)

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error receiving data: {str(e)}")
                self.stop()

    def get_latest_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get latest force and torque measurements"""
        _, fx, fy, fz, tx, ty, tz = self._data_queue.get_nowait()
        force, torque = np.array([fx, fy, fz]), np.array([tx, ty, tz])
        force = self.Tr_base @ force
        torque = self.Tr_base @ torque
        if self.bias_force is not None and self.bias_torque is not None:
            force -= self.bias_force
            torque -= self.bias_torque

        return force, torque
    
    def get_smoothed_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns averaged force and torque over all readings since last call
        Returns ((0,0,0), (0,0,0)) if no new data available
        """
        current_time = time.monotonic()
        sum_fx = sum_fy = sum_fz = 0.0
        sum_tx = sum_ty = sum_tz = 0.0
        count = 0

        # Process all available data in the queue
        while not self._data_queue.empty():
            try:
                item = self._data_queue.get_nowait()
                timestamp, fx, fy, fz, tx, ty, tz = item
                
                # First call: process all available data
                if self._last_smoothed_time is None:
                    valid = True
                else:
                    valid = timestamp > self._last_smoothed_time

                if valid:
                    sum_fx += fx
                    sum_fy += fy
                    sum_fz += fz
                    sum_tx += tx
                    sum_ty += ty
                    sum_tz += tz
                    count += 1

            except queue.Empty:
                break

        # Calculate averages
        if count > 0:
            avg_force = np.array([
                sum_fx / count,
                sum_fy / count,
                sum_fz / count
            ])
            avg_torque = np.array([
                sum_tx / count,
                sum_ty / count,
                sum_tz / count
            ])
        else:
            avg_force = np.zeros(3)
            avg_torque = np.zeros(3)

        # Update smoothing window timestamp
        self._last_smoothed_time = current_time

        # Transform force and torque to base frame
        avg_force = self.Tr_base @ avg_force
        avg_torque = self.Tr_base @ avg_torque

        # Bias force and torque
        if self.bias_force is not None and self.bias_torque is not None:
            avg_force -= self.bias_force
            avg_torque -= self.bias_torque

        return avg_force, avg_torque

    def stop(self):
        """Stop streaming and clean up resources"""
        self._running = False
        try:
            self._send_command(self.CMD_STOP_STREAMING)
        except:
            pass
        self.sock.close()

if __name__ == "__main__":
    sensor = NetFTSensor()
    try:
        while True:
            force, torque = sensor.get_smoothed_data()
            print(f"Force: {force}, Torque: {torque}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        sensor.stop()
        print("Sensor stopped")