/*
 * MIT Mini Cheetah ESC CAN motor controller
 */

#include <CANSAME5x.h>

// Global variabls
CANSAME5x CAN;                    // CAN object
unsigned long previousMicros = 0; // Stores the last time the loop was executed
unsigned long currentMicros;      // Stores the current time in microseconds
bool zero_cmd = false;            // Flag to zero out commands, =true if within tolerance of target
int16_t last_t_raw = 2048;

// Range Constants
const float P_MIN = -12.5; // Min position [rad]
const float P_MAX = 12.5;  // Max position [rad]
const float V_MIN = -65;   // Min velocity [rad/s]
const float V_MAX = 65;    // Max velocity [rad/s]
const float I_MIN = -4;    // Min current [A]
const float I_MAX = 4;     // Max current [A]
const float VB_MIN = 0;    // Min voltage [V]
const float VB_MAX = 80;   // Max voltage [V]

// Target commands
float pos_cmd = 1.57; // Target position command [rad]
// const float vel_cmd = 0.0;          // Target velocity command [rad/s]
// const uint16_t kp   = 512;            // Proportional gain
// const uint16_t kd   = 256;            // Derivative gain
// const float ff_current = 0.0;       // Feed-forward current [A]
const float p_tol = 0.02; // Tolerance for position command [rad]
float pos_offset = 0.0;
int dir = 1;

void setup()
{

    // Start Serial
    Serial.begin(115200);
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

    currentMicros = micros(); // Start timer in microseconds
}

void loop()
{

    // prepare packet
    byte cmd_packet[8];
    if (zero_cmd)
    {
        //    encodePacket(cmd_packet, 0.0, 0.0, 0, 10, 0.0);
        encodePacket(cmd_packet, pos_cmd, 0.0, 40, 8, 0.0);
    }
    else
    {
        //                       pos, vel, kp , kd , I_FF
        encodePacket(cmd_packet, pos_cmd, 0.0, 40, 50, 0.0); // hold position
        // encodePacket(cmd_packet, 0.0, 0.0, 0, 5, 0.0); // compliance
        // encodePacket(cmd_packet, 0.0, 0.0, 0, 0, 1.0);      // torque mode
        // encodePacket(cmd_packet, 0.0, 0.0, 0, 1, 0.0);    // Constant speed ??
    }

    pos_cmd = pos_cmd + dir * 0.000;
    if (pos_cmd > 12.0)
    {
        dir = -1;
    }
    if (pos_cmd < -12.0)
    {
        dir = 1;
    }

    // Send packet
    CAN.beginPacket(1);
    for (int i = 0; i < 8; i++)
    {
        CAN.write(cmd_packet[i]);
    }
    CAN.endPacket();

    // Read returned status packet
    readAndDecodePacket();

    // Loop statistics
    float loopTime = (micros() - currentMicros) / 1000000.0;
    currentMicros = micros(); // Get the current time in microseconds
    Serial.print(">LoopFrequency[kHz]:");
    Serial.println(1.0 / loopTime / 1000.0);
    // delay(100);
}

// ==================
//  Helper functions
// ==================

// Purpose: Given a command, modify the packet array
// Note: id is 11 bits, packet can contain up to 8 bytes of data
void encodePacket(byte packet[8], float pos_cmd, float vel_cmd, uint16_t kp, uint16_t kd, float ff_current)
{

    uint16_t positionCommand = float_to_uint(pos_cmd, P_MIN, P_MAX, 16);       // 16 bits 65535
    uint16_t velocityCommand = float_to_uint(vel_cmd, V_MIN, V_MAX, 12);       // 12 bits 4095
    uint16_t feedForwardCurrent = float_to_uint(ff_current, I_MIN, I_MAX, 12); // 12 bits

    // Packing the commands into the packet array
    packet[0] = positionCommand >> 8;                                  // Position command high byte
    packet[1] = positionCommand & 0xFF;                                // Position command low byte
    packet[2] = (velocityCommand >> 4) & 0xFF;                         // Velocity command high part
    packet[3] = ((velocityCommand & 0xF) << 4) | ((kp >> 8) & 0xF);    // Velocity low part + Kp high part
    packet[4] = kp & 0xFF;                                             // Kp low byte
    packet[5] = kd >> 4;                                               // Kd high part
    packet[6] = ((kd & 0xF) << 4) | ((feedForwardCurrent >> 8) & 0xF); // Kd low part + Feed-forward current high part
    packet[7] = feedForwardCurrent & 0xFF;                             // Feed-forward current low byte
}

void readAndDecodePacket()
{

    int packetSize = CAN.parsePacket();

    // non zero packet size means things waiting in buffer
    if (packetSize)
    {
        // Only read rx data from controller (which is 7 bytes long)
        if (packetSize == 7)
        {
            byte packet[8];
            for (int i = 0; i < 7; i++)
            {
                packet[i] = CAN.read();
            }
            decodePacket(packet);
        }
    }
}

/*
 * Input: Standard 8 byte CAN packet encoded by ESC
 * Purpose: 1. Decode the packet to human readable values
 *          2. Print the values
 *          3. Set the zero_cmd flag if the position command is within tolerance of zero
 * Note: Position [16 bits], Velocity [12 bits], Torque/Current [12 bits], In Voltage [8 bits]
 */
void decodePacket(byte packet[8])
{

    int offset = 1; // Starting offset of data in packet

    // Decoding the packet from bin
    uint16_t p_raw = (packet[0 + offset] << 8) | packet[1 + offset];
    uint16_t v_raw = (packet[2 + offset] << 4) | (packet[3 + offset] >> 4);
    uint16_t t_raw = ((packet[3 + offset] & 0x0F) << 8) | packet[4 + offset];
    uint8_t vb_raw = packet[5 + offset];

    int16_t signed_t_raw = (int16_t)t_raw; // Cast to signed 16-bit integer needed for arithmetic
    if (abs(signed_t_raw - last_t_raw) > 2000)
    {
        if (signed_t_raw < last_t_raw)
        {
            signed_t_raw = signed_t_raw + 4096;
        }
        else
        {
            signed_t_raw = signed_t_raw - 4096;
        }
    }
    last_t_raw = signed_t_raw;

    float p = uint_to_float(p_raw, P_MIN, P_MAX, 16);
    float v = uint_to_float(v_raw, V_MIN, V_MAX, 12);
    float t = int_to_float_overflow(signed_t_raw, I_MIN, I_MAX, 12); // Special treatment for current, it might go out of range and wrap around.
    float vb = uint_to_float(vb_raw, VB_MIN, VB_MAX, 8);

    // Printing decoded values
    Serial.print(">Position[rad]:");
    Serial.println(p); // Serial.print(",");
    Serial.print(">Velocity[rad/s]:");
    Serial.println(v); // Serial.print(",");
    Serial.print(">Current[A]:");
    Serial.println(t); // Serial.print(",");
    Serial.print(">Voltage[V]:");
    Serial.println(vb); // Serial.print(",");

    // Set zero_cmd flag if position command is within tolerance of zero
    if (abs(p - pos_cmd) < p_tol)
    {
        zero_cmd = true;
    }
    else
    {
        zero_cmd = false;
    }
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
