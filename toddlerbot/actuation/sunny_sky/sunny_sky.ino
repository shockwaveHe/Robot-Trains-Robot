/*
 * MIT Mini Cheetah ESC CAN motor controller
 */

#include <CANSAME5x.h>

// Global variabls
CANSAME5x CAN;                     // CAN object
unsigned long previous_micros = 0; // Stores the last time the loop was executed
unsigned long current_micros;      // Stores the current time in microseconds
int16_t last_t_raw = 2048;
unsigned long BAUD_RATE = 115200; // Baud rate for serial communication

// Range Constants
const float P_MIN = -12.5; // Min position [rad]
const float P_MAX = 12.5;  // Max position [rad]
const float V_MIN = -65;   // Min velocity [rad/s]
const float V_MAX = 65;    // Max velocity [rad/s]
const float I_MIN = -8;    // Min current [A]
const float I_MAX = 8;     // Max current [A]
const float I_TOL = 0.1;   // Current tolerance [A]
const float VB_MIN = 0;    // Min voltage [V]
const float VB_MAX = 80;   // Max voltage [V]

// Data received from the serial port
static bool recv_in_progress = false;
static bool length_received = false;
static uint16_t ndx = 0;
static uint16_t payload_length = 0;
byte received_bytes[1024]; // an array to store the received data
bool new_data = false;
char start_marker = '<';
char end_marker = '>';

// Define the CAN ids for the motors
const int num_can_ids = 2;
uint8_t can_ids[num_can_ids]; // Array to hold CAN IDs

struct MotorCommand
{
    float p_des, v_des, i_ff;
    uint16_t kp, kd;
    byte packet_buffer[8];
};

struct MotorState
{
    float p, v, t, vb;
};

MotorCommand motor_commands[num_can_ids];
MotorState motor_states[num_can_ids];

void resetCAN()
{
    Serial.println("Attempting to reset CAN controller...");
    CAN.end();
    delay(100);
    CAN.begin(1000000);
    if (!CAN.begin(1000000))
    {
        Serial.println("Error: CAN reset failed!");
        while (1)
            delay(10); // Halt the system on persistent failure
    }
    Serial.println("CAN reset successful.");
}

void setupCAN()
{
    pinMode(PIN_CAN_STANDBY, OUTPUT);
    digitalWrite(PIN_CAN_STANDBY, LOW); // Disable STANDBY mode
    pinMode(PIN_CAN_BOOSTEN, OUTPUT);
    digitalWrite(PIN_CAN_BOOSTEN, HIGH); // Enable BOOST mode

    Serial.println("Initializing CAN interface...");
    if (!CAN.begin(1000000))
    {
        Serial.println("Error: Starting CAN failed!");
        resetCAN(); // Attempt to reset the CAN bus
    }
    else
    {
        Serial.println("CAN interface started successfully.");
    }
}

void testCAN()
{
    // Send a test packet
    CAN.beginPacket(0x001);
    CAN.write('H'); // Simple test message
    CAN.write('i');
    if (CAN.endPacket())
    {
        Serial.println("Test packet sent successfully.");
    }
    else
    {
        Serial.println("Failed to send test packet.");
    }
}

void setup()
{
    // Start Serial
    Serial.begin(BAUD_RATE);
    while (!Serial)
        delay(10);

    setupCAN();

    testCAN();

    current_micros = micros(); // Start timer in microseconds

    for (int i = 0; i < num_can_ids; i++)
    {
        can_ids[i] = i + 1; // Assuming CAN IDs start from 1
    }
}

void loop()
{
    recvWithStartEndMarkers();

    if (new_data == true)
    {
        // Assuming the first byte is the number of commands
        uint8_t num_motors = received_bytes[0];
        int offset = 1; // Start reading commands after the num_motors byte

        for (int i = 0; i < num_motors; i++)
        {
            // Extract the motor ID
            uint8_t id = received_bytes[offset];
            int index = findMotorIndex(id);

            if (index != -1)
            {
                // Advance the offset to the start of the command data
                offset += 1;

                // Process the special command pattern or normal command data
                if (received_bytes[offset] == 0xFF)
                {
                    // Special command pattern
                    memcpy(&motor_commands[index].packet_buffer, &received_bytes[offset], 8);
                    offset += 8; // Move past this command to the next
                }
                else if (strncmp((const char *)&received_bytes[offset], "get_state", 9) == 0)
                {
                    if (i == 0)
                    {
                        Serial.print(">tx_data:");
                    }
                    Serial.print(String(id) + "," + String(motor_states[index].p) + "," + String(motor_states[index].v) + "," + String(motor_states[index].t) + "," + String(motor_states[index].vb) + ";");
                    if (i == num_motors - 1)
                    {
                        Serial.println();
                    }
                    offset += 9; // Move past this command to the next
                }
                else
                {
                    // Assuming normal command structure: <B2f2Hf>
                    memcpy(&motor_commands[index].p_des, &received_bytes[offset], sizeof(float));
                    offset += sizeof(float); // Advance past p_des
                    memcpy(&motor_commands[index].v_des, &received_bytes[offset], sizeof(float));
                    offset += sizeof(float); // Advance past v_des
                    memcpy(&motor_commands[index].kp, &received_bytes[offset], sizeof(uint16_t));
                    offset += sizeof(uint16_t); // Advance past kp
                    memcpy(&motor_commands[index].kd, &received_bytes[offset], sizeof(uint16_t));
                    offset += sizeof(uint16_t); // Advance past kd
                    memcpy(&motor_commands[index].i_ff, &received_bytes[offset], sizeof(float));
                    offset += sizeof(float); // Advance past i_ff

                    // Serial.println(">rx_data:" + String(id) + "," + String(motor_commands[index].p_des) + "," + String(motor_commands[index].v_des) + "," + String(motor_commands[index].kp) + "," + String(motor_commands[index].kd) + "," + String(motor_commands[index].i_ff) + ";");
                }
            }
            else
            {
                Serial.println("Error: Received CAN ID " + String(id) + " not recognized.");
                // Skip this command entirely (assuming fixed size for now)
                offset += 8; // Adjust based on your command size
            }
        }
    }

    new_data = false;

    sendPacket();

    readPacket();

    // Loop statistics
    // float loopTime = (micros() - currentMicros) / 1000000.0;
    // currentMicros = micros(); // Get the current time in microseconds
    // Serial.print(">LoopFrequency[kHz]:");
    // Serial.println(1.0 / loopTime / 1000.0);
    // delay(100);
}

// ==================
//  Helper functions
// ==================

void printPacket(byte packet[8])
{
    String output = "Packet:";
    for (int i = 0; i < 8; i++)
    {
        char hexStr[3];                         // Temporary string buffer for the hexadecimal representation of the byte
        sprintf(hexStr, "%02X", packet[i]);     // Converts byte to a two-digit hexadecimal number
        output += String(" ") + String(hexStr); // Append the hexadecimal string to the output String
    }

    Serial.println(output);
}

int findMotorIndex(uint8_t id)
{
    for (int i = 0; i < num_can_ids; ++i)
    {
        if (can_ids[i] == id)
        {
            return i;
        }
    }
    return -1; // Return -1 if not found
}

void recvWithStartEndMarkers()
{
    byte rb;

    while (Serial.available() > 0 && !new_data)
    {
        rb = Serial.read();

        if (!recv_in_progress)
        {
            if (rb == start_marker)
            {
                recv_in_progress = true;
                ndx = 0;
            }
        }
        else if (!length_received)
        {
            // Ensure 2 bytes for the length are available
            if (Serial.available() > 0)
            {
                // Read the payload length as little-endian
                payload_length = rb | (Serial.read() << 8);
                if (payload_length > sizeof(received_bytes))
                {
                    Serial.println("Error: Payload length exceeds buffer size.");
                    recv_in_progress = false;
                    continue;
                }
                length_received = true;
            }
        }
        else if (ndx < payload_length)
        {
            // Protect against buffer overflow
            if (ndx < sizeof(received_bytes))
            {
                received_bytes[ndx++] = rb;
            }
        }
        else
        {
            if (rb == end_marker)
            {
                received_bytes[ndx] = '\0'; // Terminate the string
                new_data = true;
            }
            // Complete message received
            recv_in_progress = false;
            length_received = false;
            ndx = 0;
        }
    }
}

void sendPacket()
{
    for (int i = 0; i < num_can_ids; i++)
    {
        int index = findMotorIndex(can_ids[i]);
        if (index == -1)
            continue; // Skip if motor index is not found

        CAN.beginPacket(can_ids[i]);

        bool is_empty = true;
        for (int j = 0; j < 8; j++)
        {
            if (motor_commands[index].packet_buffer[j] != 0)
            {
                is_empty = false;
                break;
            }
        }

        byte cmd_packet[8];
        if (is_empty)
        {
            encodePacket(cmd_packet, motor_commands[index]);
        }
        else
        {
            memcpy(&cmd_packet, &motor_commands[index].packet_buffer, sizeof(cmd_packet));
            memset(&motor_commands[index].packet_buffer, 0, sizeof(cmd_packet));
        }

        bool success = true;
        for (int j = 0; j < sizeof(cmd_packet); j++)
        {
            if (!CAN.write(cmd_packet[j]))
            {
                success = false;
                break;
            }
        }

        CAN.endPacket();

        if (!success)
        {
            Serial.println("Error: Failed to send packet for CAN ID " + String(can_ids[i]));
        }
    }
}

void readPacket()
{
    while (CAN.parsePacket())
    {
        if (CAN.packetExtended() || CAN.packetRtr())
        {
            // Handle extended and remote transmission request packets if needed
            Serial.println("Non-standard packet received. Ignored.");
            continue;
        }

        int packet_size = CAN.available();
        if (packet_size != 7)
        {
            Serial.println("Error: Incorrect packet size: " + String(packet_size));
            continue;
        }

        // Serial.println("Packet size: " + String(packet_size));
        byte packet[8];
        for (int i = 0; i < packet_size; i++)
        {
            packet[i] = CAN.read(); // Read the packet data
        }

        // Obtain CAN ID of the received packet
        uint8_t id = packet[0];
        int index = findMotorIndex(id);

        if (index == -1)
        {
            Serial.println("Error: Received packet from unrecognized CAN ID: " + String(id));
        }
        else
        {
            decodePacket(packet, motor_states[index]);
        }
    }
}

// Purpose: Given a command, modify the packet array
// Note: id is 11 bits, packet can contain up to 8 bytes of data
void encodePacket(byte packet[8], MotorCommand &command)
{
    uint16_t pos_cmd = float_to_uint(command.p_des, P_MIN, P_MAX, 16); // Convert position command
    uint16_t vel_cmd = float_to_uint(command.v_des, V_MIN, V_MAX, 12); // Convert velocity command
    uint16_t ffi_cmd = float_to_uint(command.i_ff, I_MIN, I_MAX, 12);  // Convert feed-forward current command

    // Packing the commands into the packet array
    packet[0] = pos_cmd >> 8;                                       // Position command high byte
    packet[1] = pos_cmd & 0xFF;                                     // Position command low byte
    packet[2] = (vel_cmd >> 4) & 0xFF;                              // Velocity command high part
    packet[3] = ((vel_cmd & 0xF) << 4) | ((command.kp >> 8) & 0xF); // Velocity low part + Kp high part
    packet[4] = command.kp & 0xFF;                                  // Kp low byte
    packet[5] = command.kd >> 4;                                    // Kd high part
    packet[6] = ((command.kd & 0xF) << 4) | ((ffi_cmd >> 8) & 0xF); // Kd low part + Feed-forward current high part
    packet[7] = ffi_cmd & 0xFF;                                     // Feed-forward current low byte
}

/*
 * Input: Standard 8 byte CAN packet encoded by ESC
 * Purpose: 1. Decode the packet to human readable values
 *          2. Print the values
 *          3. Set the zero_cmd flag if the position command is within tolerance of zero
 * Note: Position [16 bits], Velocity [12 bits], Torque/Current [12 bits], In Voltage [8 bits]
 */
void decodePacket(byte packet[8], MotorState &state)
{
    // Decoding the packet from bin
    uint8_t id = packet[0];
    uint16_t p_raw = (packet[1] << 8) | packet[2];
    uint16_t v_raw = (packet[3] << 4) | (packet[4] >> 4);
    uint16_t t_raw = ((packet[4] & 0x0F) << 8) | packet[5];
    uint8_t vb_raw = packet[6];

    int16_t signed_t_raw = (int16_t)t_raw; // Cast to signed 16-bit integer needed for arithmetic
    signed_t_raw += (abs(signed_t_raw - last_t_raw) > 4000) ? ((signed_t_raw < last_t_raw) ? 4096 : -4096) : 0;
    last_t_raw = signed_t_raw;

    state.p = uint_to_float(p_raw, P_MIN, P_MAX, 16);
    state.v = uint_to_float(v_raw, V_MIN, V_MAX, 12);
    state.t = int_to_float_overflow(signed_t_raw, I_MIN, I_MAX, 12); // Special treatment for current, it might go out of range and wrap around.
    state.vb = uint_to_float(vb_raw, VB_MIN, VB_MAX, 8);
}

// Convert uint [0,2^bits) to float [minv,maxv)
float uint_to_float(uint16_t value, float minv, float maxv, uint8_t bits)
{
    float scale = (maxv - minv) / (pow(2, bits) - 1);
    return value * scale + minv;
}

// Convert signed int [0,2^bits) to float [minv,maxv), may exceed the range if necessary
float int_to_float_overflow(int16_t value, float minv, float maxv, uint8_t bits)
{
    float scale = (maxv - minv) / (pow(2, bits) - 1);
    return (float)value * scale + minv;
}

// Convert float [minv, maxv) to uint [0,2^bits)
uint16_t float_to_uint(float value, float minv, float maxv, uint8_t bits)
{
    value = constrain(value, minv, maxv);
    float scale = (maxv - minv) / (pow(2, bits) - 1);
    return (value - minv) / scale;
}
