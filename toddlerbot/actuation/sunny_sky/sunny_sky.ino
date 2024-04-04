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
const float I_MIN = -4;    // Min current [A]
const float I_MAX = 4;     // Max current [A]
const float VB_MIN = 0;    // Min voltage [V]
const float VB_MAX = 80;   // Max voltage [V]

// Data received from the serial port
const byte num_bytes = 32;
byte received_bytes[num_bytes]; // an array to store the received data
boolean new_data = false;
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

void setup()
{
    // Start Serial
    Serial.begin(BAUD_RATE);
    while (!Serial)
        delay(10);

    // Prepare CAN pins
    pinMode(PIN_CAN_STANDBY, OUTPUT);
    digitalWrite(PIN_CAN_STANDBY, false); // turn off STANDBY
    pinMode(PIN_CAN_BOOSTEN, OUTPUT);
    digitalWrite(PIN_CAN_BOOSTEN, true); // turn on booster

    // Start the CAN bus at 1Mbps
    if (!CAN.begin(1000000))
    {
        Serial.println("Starting CAN failed!");
        while (1)
            delay(10);
    }

    current_micros = micros(); // Start timer in microseconds

    for (int i = 0; i < num_can_ids; i++)
    {
        can_ids[i] = i + 1; // Assuming CAN IDs start from 1
    }
}

void loop()
{
    recvWithStartEndMarkers();

    byte cmd_packet[8];
    if (new_data == true)
    {
        uint8_t id = received_bytes[0];
        int index = findMotorIndex(id);

        if (index != -1)
        { // Ensure the ID was found
            if (received_bytes[1] == 0xFF)
            { // Check for special command pattern
                memcpy(&motor_commands[index].packet_buffer, &received_bytes[1], 8);
            }
            else
            {
                // Extract the command data directly into the motor_commands structure for the found index
                // Check if the position command hits the hardware limit
                if (motor_states[index].t < I_MIN || motor_states[index].t > I_MAX)
                {
                    motor_commands[index].p_des = motor_states[index].p;
                }
                else
                {
                    memcpy(&motor_commands[index].p_des, &received_bytes[1], sizeof(float));
                }

                memcpy(&motor_commands[index].v_des, &received_bytes[1 + sizeof(float)], sizeof(float));
                memcpy(&motor_commands[index].kp, &received_bytes[1 + 2 * sizeof(float)], sizeof(uint16_t));
                memcpy(&motor_commands[index].kd, &received_bytes[1 + 2 * sizeof(float) + sizeof(uint16_t)], sizeof(uint16_t));
                memcpy(&motor_commands[index].i_ff, &received_bytes[1 + 2 * sizeof(float) + 2 * sizeof(uint16_t)], sizeof(float));

                Serial.println(">rx_data:" + String(id) + "," + String(motor_commands[index].p_des) + "," + String(motor_commands[index].v_des) + "," + String(motor_commands[index].kp) + "," + String(motor_commands[index].kd) + "," + String(motor_commands[index].i_ff));
            }
        }
        else
        {
            Serial.println("Error: Received CAN ID not recognized.");
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
    static boolean recv_in_progress = false;
    static byte ndx = 0;
    byte rb;

    while (Serial.available() > 0 && new_data == false)
    {
        rb = Serial.read();

        if (recv_in_progress == true)
        {
            if (rb != end_marker)
            {
                received_bytes[ndx] = rb;
                ndx++;
                if (ndx >= num_bytes)
                {
                    ndx = num_bytes - 1;
                }
            }
            else
            {
                received_bytes[ndx] = '\0'; // terminate the string
                recv_in_progress = false;
                ndx = 0;
                new_data = true;
            }
        }

        else if (rb == start_marker)
        {
            recv_in_progress = true;
        }
    }
}

void sendPacket()
{
    for (int i = 0; i < num_can_ids; i++)
    {
        CAN.beginPacket(can_ids[i]); // Use the canId parameter to specify the target CAN ID
        int index = findMotorIndex(can_ids[i]);

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
            // printPacket(cmd_packet);
        }
        else
        {
            memcpy(&cmd_packet, &motor_commands[index].packet_buffer, 8);
            memset(&motor_commands[index].packet_buffer, 0, 8);
        }

        for (int j = 0; j < 8; j++)
        {
            CAN.write(cmd_packet[j]);
        }

        CAN.endPacket();
    }
}

void readPacket()
{
    while (CAN.parsePacket())
    {
        int packet_size = CAN.available();
        if (packet_size && packet_size == 7)
        {
            // Serial.println("Packet size: " + String(packet_size));
            byte packet[8];
            for (int i = 0; i < packet_size; i++)
            {
                packet[i] = CAN.read(); // Read the packet data
            }

            // Obtain CAN ID of the received packet
            uint8_t id = packet[0];
            int index = findMotorIndex(id);

            if (index != -1)
            { // Ensure the CAN ID is recognized
                // Assuming a decodePacket function that updates the motor state
                decodePacket(packet, motor_states[index]);
            }
            else
            {
                // Handle unrecognized CAN ID, if necessary
                Serial.println("Received packet from unrecognized CAN ID.");
            }
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

    Serial.println(">tx_data:" + String(id) + "," + String(state.p) + "," + String(state.v) + "," + String(state.t) + "," + String(state.vb));
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
