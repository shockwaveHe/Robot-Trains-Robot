import serial


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
        self.serial = serial.Serial(port=portname, baudrate=baudrate, timeout=timeout)

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

    def close(self):
        self.serial.close()

    def send_packet(self):
        if self.serial.is_open:
            self.serial.write(self.tx_buffer[: self.tx_buffer_index])

    def receive_packet(self, size):
        if not self.serial.is_open:
            return -1

        expected_header = b"\xff\xff\xff"
        # Attempt to read the remaining bytes needed to reach the desired size
        read_buffer = self.serial.read(size)
        # Check if the buffer contains the expected header and has reached the necessary size
        if len(read_buffer) >= size and expected_header in read_buffer[:3]:
            # Assuming the header is at the start for now, further logic could adjust for different positions
            self.rx_buffer[:size] = read_buffer[:size]
            return 1
        else:
            # If the function exits the loop without returning success, it's due to a timeout or other failure
            return -1

    def send_command(self, id, instruction, parameters):
        """
        General method to prepare and send a command to the actuator.
        """
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
        self.send_packet()

    def action(self, id):
        self.send_command(id, self.MIGHTYZAP_ACTION, [])

    def reset_write(self, id, option):
        self.send_command(id, self.MIGHTYZAP_RESET, [option])

    def restart(self, id):
        self.send_command(id, self.MIGHTYZAP_RESTART, [])

    def factory_reset_write(self, id, option):
        self.send_command(id, self.MIGHTYZAP_FACTORY_RESET, [option])

    def ping(self, id):
        self.send_command(id, self.MIGHTYZAP_PING, [])

    def read_data(self, id, addr, size):
        self.send_command(id, self.MIGHTYZAP_READ_DATA, [addr, size])

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

    def write_data(self, id, addr, value):
        if isinstance(id, list) and isinstance(value, list):
            # Ensure id and value lists are of the same length
            assert len(id) == len(value)

            # Prepare data for sync write, interleaving ID and value pairs
            data = []
            for single_id, single_value in zip(id, value):
                single_data, size = self.parse_value(single_value)
                data.extend([single_id] + single_data)

            parameters = [addr, size] + data
            self.send_command(self.BROADCAST_ID, self.MIGHTYZAP_SYNC_WRITE, parameters)
        else:
            single_data, _ = self.parse_value(value)
            parameters = [addr] + single_data
            # Send command for a single motor
            self.send_command(id, self.MIGHTYZAP_WRITE_DATA, parameters)

    def goal_position(self, id, position):
        self.write_data(id, 0x86, position)

    def present_position(self, id):
        self.read_data(id, 0x8C, 2)
        if self.receive_packet(9) == 1:
            return (self.rx_buffer[7] << 8) | self.rx_buffer[6]
        return -1

    def goal_speed(self, id, speed):
        self.write_data(id, 0x15, speed)

    def goal_current(self, id, curr):
        self.write_data(id, 0x34, curr)

    def acceleration(self, id, acc):
        self.write_data(id, 0x21, acc)

    def deceleration(self, id, dec):
        self.write_data(id, 0x22, dec)

    def short_stroke_limit(self, id, limit):
        self.write_data(id, 0x06, limit)

    def long_stroke_limit(self, id, limit):
        self.write_data(id, 0x08, limit)

    def force_enable(self, id, enable):
        self.write_data(id, 0x80, enable)

    def set_shutdown_enable(self, id, flag):
        self.write_data(id, 0x12, flag)

    def get_shutdown_enable(self, id):
        self.read_data(id, 0x12, 1)
        if self.receive_packet(8) == 1:
            return self.rx_buffer[6]
        return -1

    def set_error_indicator_enable(self, id, flag):
        self.write_data(id, 0x11, flag)

    def get_error_indicator_enable(self, id):
        self.read_data(id, 0x11, 1)
        if self.receive_packet(8) == 1:
            return self.rx_buffer[6]
        return -1

    def read_error(self, id):
        self.ping(id)
        if self.receive_packet(7) == 1:
            return self.rx_buffer[5]
        return -1

    def set_return_delay_time(self, id, time):
        self.write_data(id, 0x5, time)
