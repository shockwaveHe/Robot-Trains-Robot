"""Communication using the DynamixelSDK."""

##This is based off of the dynamixel SDK
import atexit
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import dynamixel_sdk
import numpy as np
import numpy.typing as npt

from toddlerbot.utils.misc_utils import log

PROTOCOL_VERSION = 2.0

# The following addresses assume XH motors.
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POS_VEL = 128
ADDR_PRESENT_POS_VEL_CUR = 126

# Data Byte Length
LEN_PRESENT_POSITION = 4
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_CURRENT = 2
LEN_PRESENT_POS_VEL = 8
LEN_PRESENT_POS_VEL_CUR = 10
LEN_GOAL_POSITION = 4

DEFAULT_POS_SCALE = 2.0 * np.pi / 4096  # 0.088 degrees
# See http://emanual.robotis.com/docs/en/dxl/x/xh430-v210/#goal-velocity
DEFAULT_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0  # 0.229 rpm
DEFAULT_CUR_SCALE = 1.34


def dynamixel_cleanup_handler():
    """Cleanup function to ensure Dynamixels are disconnected properly."""
    open_clients: List[DynamixelClient] = list(DynamixelClient.OPEN_CLIENTS)  # type: ignore
    for open_client in open_clients:
        if open_client.port_handler.is_using:
            log("Forcing client to close.", header="Dynamixel", level="warning")
        open_client.port_handler.is_using = False
        open_client.disconnect()


def signed_to_unsigned(value: int, size: int) -> int:
    """Converts the given value to its unsigned representation."""
    if value < 0:
        bit_size = 8 * size
        max_value = (1 << bit_size) - 1
        value = max_value + value
    return value


def unsigned_to_signed(value: int, size: int) -> int:
    """Converts the given value from its unsigned representation."""
    bit_size = 8 * size
    if (value & (1 << (bit_size - 1))) != 0:
        value = -((1 << bit_size) - value)
    return value


class DynamixelClient:
    """Client for communicating with Dynamixel motors.

    NOTE: This only supports Protocol 2.
    """

    # The currently open clients.
    OPEN_CLIENTS: Set[Any] = set()

    def __init__(
        self,
        motor_ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 1000000,
        lazy_connect: bool = False,
    ):
        """Initializes a new client.

        Args:
            motor_ids: All motor IDs being used by the client.
            port: The Dynamixel device to talk to. e.g.
                - Linux: /dev/ttyUSB0
                - Mac: /dev/tty.usbserial-*
                - Windows: COM1
            baudrate: The Dynamixel baudrate to communicate with.
            lazy_connect: If True, automatically connects when calling a method
                that requires a connection, if not already connected.
            pos_scale: The scaling factor for the positions. This is
                motor-dependent. If not provided, uses the default scale.
            vel_scale: The scaling factor for the velocities. This is
                motor-dependent. If not provided uses the default scale.
            cur_scale: The scaling factor for the currents. This is
                motor-dependent. If not provided uses the default scale.
        """
        self.dxl = dynamixel_sdk

        self.motor_ids = list(motor_ids)
        self.port_name = port
        self.baudrate = baudrate
        self.lazy_connect = lazy_connect

        self.port_handler = self.dxl.PortHandler(port)
        self.packet_handler = self.dxl.PacketHandler(PROTOCOL_VERSION)

        self._bulk_reader = self.dxl.GroupBulkRead(
            self.port_handler, self.packet_handler
        )
        for motor_id in self.motor_ids:
            success = self._bulk_reader.addParam(
                motor_id, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR
            )
            if not success:
                raise OSError(
                    "[Motor ID: {}] Could not add parameter to bulk read.".format(
                        motor_id
                    )
                )

        self._sync_readers: Dict[Tuple[int, int], dynamixel_sdk.GroupSyncRead] = {}
        self._sync_writers: Dict[Tuple[int, int], dynamixel_sdk.GroupSyncWrite] = {}

        self._data_dict: Dict[str, npt.NDArray[np.float32]] = {}

        self.OPEN_CLIENTS.add(self)

    @property
    def is_connected(self) -> bool:
        return self.port_handler.is_open

    def connect(self):
        """Connects to the Dynamixel motors.

        NOTE: This should be called after all DynamixelClients on the same
            process are created.
        """
        assert not self.is_connected, "Client is already connected."

        if self.port_handler.openPort():
            log(f"Succeeded to open port: {self.port_name}", header="Dynamixel")
        else:
            raise OSError(
                (
                    "Failed to open port at {} (Check that the device is powered "
                    "on and connected to your computer)."
                ).format(self.port_name)
            )

        if self.port_handler.setBaudRate(self.baudrate):
            log(f"Succeeded to set baudrate to {self.baudrate}", header="Dynamixel")
        else:
            raise OSError(
                (
                    "Failed to set the baudrate to {} (Ensure that the device was "
                    "configured for this baudrate)."
                ).format(self.baudrate)
            )

        # Start with all motors enabled.  NO, I want to set settings before enabled
        # self.set_torque_enabled(self.motor_ids, True)

    def disconnect(self):
        """Disconnects from the Dynamixel device."""
        if not self.is_connected:
            return
        if self.port_handler.is_using:
            log(
                "Port handler in use; cannot disconnect.",
                header="Dynamixel",
                level="error",
            )
            return
        # Ensure motors are disabled at the end.
        self.set_torque_enabled(self.motor_ids, False)
        self.port_handler.closePort()
        if self in self.OPEN_CLIENTS:
            self.OPEN_CLIENTS.remove(self)

    def set_torque_enabled(
        self,
        motor_ids: Sequence[int],
        enabled: bool,
        retries: int = -1,
        retry_interval: float = 0.25,
    ):
        """Sets whether torque is enabled for the motors.

        Args:
            motor_ids: The motor IDs to configure.
            enabled: Whether to engage or disengage the motors.
            retries: The number of times to retry. If this is <0, will retry
                forever.
            retry_interval: The number of seconds to wait between retries.
        """
        remaining_ids = list(motor_ids)
        while remaining_ids:
            remaining_ids = list(
                self.write_byte(
                    remaining_ids,
                    int(enabled),
                    ADDR_TORQUE_ENABLE,
                )
            )
            if remaining_ids:
                log(
                    f"Could not set torque {'enabled' if enabled else 'disabled'} for IDs: {str(remaining_ids)}",
                    header="Dynamixel",
                    level="error",
                )
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1

    def read_pos(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float32]]:
        """Returns the current positions and velocities."""
        return self.sync_read(
            ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION, DEFAULT_POS_SCALE
        )

    def read_vel(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float32]]:
        """Returns the current positions and velocities."""
        return self.sync_read(
            ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY, DEFAULT_VEL_SCALE
        )

    def read_cur(self, retries: int = 0) -> Tuple[float, npt.NDArray[np.float32]]:
        """Returns the current positions and velocities."""
        return self.sync_read(
            ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT, DEFAULT_CUR_SCALE
        )

    # @profile()
    def read_pos_vel(
        self, retries: int = 0
    ) -> Tuple[float, npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Returns the current positions and velocities."""
        comm_time, data_dict = self.bulk_read(["pos", "vel"], retries=retries)
        return comm_time, data_dict["pos"].copy(), data_dict["vel"].copy()

    def read_pos_vel_cur(
        self, retries: int = 0
    ) -> Tuple[
        float, npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        # NEED to update line 115 and 349 if calling this function
        """Returns the current positions and velocities."""
        comm_time, data_dict = self.bulk_read(["pos", "vel", "cur"], retries=retries)
        return (
            comm_time,
            data_dict["pos"].copy(),
            data_dict["vel"].copy(),
            data_dict["cur"].copy(),
        )

    def write_desired_pos(
        self, motor_ids: Sequence[int], positions: npt.NDArray[np.float32]
    ):
        """Writes the given desired positions.

        Args:
            motor_ids: The motor IDs to write to.
            positions: The joint angles in radians to write.
        """
        assert len(motor_ids) == len(positions)

        # Convert to Dynamixel position space.
        positions /= DEFAULT_POS_SCALE
        self.sync_write(
            motor_ids, list(positions), ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )

    def write_byte(
        self,
        motor_ids: Sequence[int],
        value: int,
        address: int,
    ) -> Sequence[int]:
        """Writes a value to the motors.

        Args:
            motor_ids: The motor IDs to write to.
            value: The value to write to the control table.
            address: The control table address to write to.

        Returns:
            A list of IDs that were unsuccessful.
        """
        self.check_connected()
        errored_ids: List[int] = []
        for motor_id in motor_ids:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, address, value
            )
            success = self.handle_packet_result(
                comm_result,
                dxl_error,
                motor_id,
                context="write_byte",
            )
            if not success:
                errored_ids.append(motor_id)
        return errored_ids

    # @profile()
    def bulk_read(
        self, attr_list: List[str], retries: int
    ) -> Tuple[float, Dict[str, npt.NDArray[np.float32]]]:
        """Reads values from a group of motors.

        Args:
            motor_ids: The motor IDs to read from.
            address: The control table address to read from.
            size: The size of the control table value being read.

        Returns:
            The values read from the motors.
        """
        self.check_connected()

        if len(self._data_dict) == 0:
            self._data_dict = {
                attr: np.zeros(len(self.motor_ids), dtype=np.float32)
                for attr in attr_list
            }

        comm_time = 0.0
        success = False
        while not success:
            # fastSyncRead does not work for 2XL and 2XC
            comm_result = self._bulk_reader.txPacket()
            comm_time = time.time()
            if comm_result == self.dxl.COMM_SUCCESS:
                comm_result = self._bulk_reader.rxPacket()

            success = self.handle_packet_result(comm_result, context="bulk_read")

            if retries == 0:
                break

            retries -= 1

        if not success:
            return comm_time, self._data_dict

        errored_ids: List[int] = []
        for i, motor_id in enumerate(self.motor_ids):
            # Check if the data is available.
            available = self._bulk_reader.isAvailable(
                motor_id, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR
            )
            if not available:
                errored_ids.append(motor_id)
                continue

            if "pos" in attr_list:
                data_unsigned = self._bulk_reader.getData(
                    motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                )
                data_signed = unsigned_to_signed(
                    data_unsigned,
                    size=LEN_PRESENT_POSITION,
                )
                self._data_dict["pos"][i] = float(data_signed) * DEFAULT_POS_SCALE

            if "vel" in attr_list:
                data_unsigned = self._bulk_reader.getData(
                    motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                )
                data_signed = unsigned_to_signed(
                    data_unsigned,
                    size=LEN_PRESENT_VELOCITY,
                )
                self._data_dict["vel"][i] = float(data_signed) * DEFAULT_VEL_SCALE

            if "cur" in attr_list:
                data_unsigned = self._bulk_reader.getData(
                    motor_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
                )
                data_signed = unsigned_to_signed(
                    data_unsigned,
                    size=LEN_PRESENT_CURRENT,
                )
                self._data_dict["cur"][i] = float(data_signed) * DEFAULT_CUR_SCALE

        if errored_ids:
            log(
                f"Bulk read failed for: {str(errored_ids)}",
                header="Dynamixel",
                level="error",
            )

        return comm_time, self._data_dict

    def sync_read(
        self, address: int, size: int, scale: float
    ) -> Tuple[float, npt.NDArray[np.float32]]:
        """Reads values from a group of motors.

        Args:
            motor_ids: The motor IDs to read from.
            address: The control table address to read from.
            size: The size of the control table value being read.

        Returns:
            The values read from the motors.
        """
        self.check_connected()
        key = (address, size)
        if key not in self._sync_readers:
            self._sync_readers[key] = self.dxl.GroupSyncRead(
                self.port_handler, self.packet_handler, address, size
            )
            for motor_id in self.motor_ids:
                success = self._sync_readers[key].addParam(motor_id)
                if not success:
                    raise OSError(
                        "[Motor ID: {}] Could not add parameter to sync read.".format(
                            motor_id
                        )
                    )

        sync_reader: dynamixel_sdk.GroupSyncRead = self._sync_readers[key]

        comm_time = 0.0
        success = False
        while not success:
            # fastSyncRead does not work for 2XL and 2XC
            # time_1 = time.time()
            comm_result = sync_reader.txPacket()
            comm_time = time.time()
            if comm_result == self.dxl.COMM_SUCCESS:
                comm_result = sync_reader.rxPacket()
            # time_2 = time.time()
            # print(f"RTT: {time_2 - time_1}")

            success = self.handle_packet_result(comm_result, context="sync_read")

        errored_ids: List[int] = []
        data_arr = np.zeros(len(self.motor_ids), dtype=np.float32)
        for i, motor_id in enumerate(self.motor_ids):
            # Check if the data is available.
            available = sync_reader.isAvailable(motor_id, address, size)
            if not available:
                errored_ids.append(motor_id)
                continue

            data_unsigned = sync_reader.getData(motor_id, address, size)
            data_signed = unsigned_to_signed(data_unsigned, size=size)
            data_arr[i] = float(data_signed) * scale

        if errored_ids:
            log(
                f"Sync read failed for: {str(errored_ids)}",
                header="Dynamixel",
                level="error",
            )

        return comm_time, data_arr

    def sync_write(
        self,
        motor_ids: Sequence[int],
        values: Sequence[Any],
        address: int,
        size: int,
    ):
        """Writes values to a group of motors.

        Args:
            motor_ids: The motor IDs to write to.
            values: The values to write.
            address: The control table address to write to.
            size: The size of the control table value being written to.
        """
        self.check_connected()
        key = (address, size)
        if key not in self._sync_writers:
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, address, size
            )
        sync_writer = self._sync_writers[key]

        errored_ids: List[int] = []
        for motor_id, desired_pos in zip(motor_ids, values):
            value = signed_to_unsigned(int(desired_pos), size=size)
            value_bytes = value.to_bytes(size, byteorder="little")
            success = sync_writer.addParam(motor_id, value_bytes)
            if not success:
                errored_ids.append(motor_id)

        if errored_ids:
            log(
                f"Sync write failed for: {str(errored_ids)}",
                header="Dynamixel",
                level="error",
            )

        comm_result = sync_writer.txPacket()
        self.handle_packet_result(comm_result, context="sync_write")

        sync_writer.clearParam()

    def check_connected(self):
        """Ensures the robot is connected."""
        if self.lazy_connect and not self.is_connected:
            self.connect()
        if not self.is_connected:
            raise OSError("Must call connect() first.")

    def handle_packet_result(
        self,
        comm_result: int,
        dxl_error: Optional[int] = None,
        dxl_id: Optional[int] = None,
        context: Optional[str] = None,
    ):
        """Handles the result from a communication request."""
        error_message = None
        if comm_result != self.dxl.COMM_SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        elif dxl_error is not None:
            error_message = self.packet_handler.getRxPacketError(dxl_error)
        if error_message:
            if dxl_id is not None:
                error_message = "[Motor ID: {}] {}".format(dxl_id, error_message)
            if context is not None:
                error_message = "> {}: {}".format(context, error_message)

            log(error_message, header="Dynamixel", level="warning")

            return False

        return True

    def clear_multi_turn(self, motor_ids: Sequence[int]):
        for motor_id in motor_ids:
            self.packet_handler.clearMultiTurn(self.port_handler, motor_id)

    def reboot(self, motor_ids: Sequence[int]):
        for motor_id in motor_ids:
            self.packet_handler.reboot(self.port_handler, motor_id)

    def convert_to_unsigned(self, value: int, size: int) -> int:
        """Converts the given value to its unsigned representation."""
        if value < 0:
            max_value = (1 << (8 * size)) - 1
            value = max_value + value
        return value

    def __enter__(self):
        """Enables use as a context manager."""
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.disconnect()

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.disconnect()


# Register global cleanup function.
atexit.register(dynamixel_cleanup_handler)
