import asyncio

import serial_asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class MightyZapClient:
    PROTOCOL_TX_BUF_SIZE = 50
    PROTOCOL_RX_BUF_SIZE = 50
    MIGHTYZAP_PING = 0xF1
    MIGHTYZAP_READ_DATA = 0xF2
    MIGHTYZAP_WRITE_DATA = 0xF3
    MIGHTYZAP_REG_WRITE = 0xF4
    MIGHTYZAP_ACTION = 0xF5
    MIGHTYZAP_RESET = 0xF6
    MIGHTYZAP_RESTART = 0xF8
    MIGHTYZAP_FACTORY_RESET = 0xF9
    MIGHTYZAP_SYNC_WRITE = 0x73
    BROADCAST_ID = 0xFE

    def __init__(self, portname, baudrate=57600, timeout=0.1):
        self.tx_buffer = [0] * self.PROTOCOL_TX_BUF_SIZE
        self.tx_buffer_index = 0
        self.rx_buffer = [0] * self.PROTOCOL_RX_BUF_SIZE
        self.rx_buffer_size = 0
        self.actuator_ID = 0
        self.checksum = 0
        self.portname = portname
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None

        asyncio.run(self.async_init())

    async def async_init(self):
        self.serial = await serial_asyncio.open_serial_connection(
            url=self.portname, baudrate=self.baudrate
        )

    def set_protocol_header(self):
        self.tx_buffer_index = 0
        for _ in range(3):
            self.tx_buffer[self.tx_buffer_index] = 0xFF
            self.tx_buffer_index += 1
        self.tx_buffer[self.tx_buffer_index] = self.actuator_ID
        self.tx_buffer_index += 1

    def set_protocol_instruction(self, ins):
        self.tx_buffer_index = 5
        self.tx_buffer[self.tx_buffer_index] = ins
        self.tx_buffer_index += 1

    def add_protocol_parameter(self, para):
        self.tx_buffer[self.tx_buffer_index] = para
        self.tx_buffer_index += 1

    def set_protocol_length_checksum(self):
        self.tx_buffer[4] = self.tx_buffer_index - 4
        self.checksum = (sum(self.tx_buffer[3 : self.tx_buffer_index]) & 0xFF) ^ 0xFF
        self.tx_buffer[self.tx_buffer_index] = self.checksum
        self.tx_buffer_index += 1

    def get_id(self):
        return self.actuator_ID

    def set_id(self, id):
        self.actuator_ID = id

    async def close(self):
        if self.serial:
            await self.serial.close()

    async def send_packet(self):
        await self.serial[1].write(bytes(self.tx_buffer[: self.tx_buffer_index]))

    async def receive_packet(self, size):
        reader = self.serial[0]
        read_buffer = await reader.readexactly(size)
        expected_header = b"\xff\xff\xff"
        if len(read_buffer) >= size and expected_header in read_buffer[:3]:
            self.rx_buffer[:size] = read_buffer[:size]
            return 1
        else:
            return -1

    async def send_command(self, id, instruction, parameters):
        self.set_id(id)
        self.set_protocol_header()
        self.set_protocol_instruction(instruction)
        for param in parameters:
            if isinstance(param, int):
                self.add_protocol_parameter(param)
            elif isinstance(param, list):
                for byte in param:
                    self.add_protocol_parameter(byte)
        self.set_protocol_length_checksum()
        await self.send_packet()

    async def action(self, id):
        await self.send_command(id, self.MIGHTYZAP_ACTION, [])

    async def reset_write(self, id, option):
        await self.send_command(id, self.MIGHTYZAP_RESET, [option])

    async def restart(self, id):
        await self.send_command(id, self.MIGHTYZAP_RESTART, [])

    async def factory_reset_write(self, id, option):
        await self.send_command(id, self.MIGHTYZAP_FACTORY_RESET, [option])

    async def ping(self, id):
        await self.send_command(id, self.MIGHTYZAP_PING, [])

    async def read_data(self, id, addr, size):
        await self.send_command(id, self.MIGHTYZAP_READ_DATA, [addr, size])

    def parse_value(self, value):
        if 0 <= value <= 0xFF:  # Value fits in 1 byte
            single_data = [value]
            size = 1
        elif 0x100 <= value <= 0xFFFF:  # Value fits in 2 bytes
            single_data = [value & 0xFF, value >> 8]
            size = 2
        else:
            raise ValueError("Value out of range for a 2-byte representation.")

        return single_data, size

    async def write_data(self, id, addr, value):
        if isinstance(id, list) and isinstance(value, list):
            # Ensure id and value lists are of the same length
            assert len(id) == len(value)

            # Prepare data for sync write, interleaving ID and value pairs
            data = []
            for single_id, single_value in zip(id, value):
                single_data, size = self.parse_value(single_value)
                data.extend([single_id] + single_data)

            parameters = [addr, size] + data
            await self.send_command(
                self.BROADCAST_ID, self.MIGHTYZAP_SYNC_WRITE, parameters
            )
        else:
            single_data, _ = self.parse_value(value)
            parameters = [addr] + single_data
            # Send command for a single motor
            await self.send_command(id, self.MIGHTYZAP_WRITE_DATA, parameters)

    async def goal_position(self, id, position):
        await self.write_data(id, 0x86, position)

    async def present_position(self, id):
        await self.read_data(id, 0x8C, 2)
        if await self.receive_packet(9) == 1:
            return (self.rx_buffer[7] << 8) | self.rx_buffer[6]
        return -1

    async def set_return_delay_time(self, id, time):
        await self.write_data(id, 0x5, time)
