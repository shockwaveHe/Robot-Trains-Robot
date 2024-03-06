/*
 * MIT Mini Cheetah ESC CAN motor controller
 */

#include <CANSAME5x.h>

// Global variabls
CANSAME5x CAN;                    // CAN object
unsigned long previousMicros = 0; // Stores the last time the loop was executed
unsigned long currentMicros;      // Stores the current time in microseconds
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

// Target commands
byte cmd_packet[8];

const byte numBytes = 32;
byte receivedBytes[numBytes]; // an array to store the received data
char startMarker = '<';
char endMarker = '>';

boolean newData = false;

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

    currentMicros = micros(); // Start timer in microseconds
}

void loop()
{
    recvWithStartEndMarkers();

    if (newData == true)
    {
        // Serial.print(">Received data: ");
        // for (int i = 0; i < numBytes; i++)
        // {
        //     Serial.print(receivedBytes[i]);
        //     if (receivedBytes[i + 1] == '\0')
        //     {
        //         Serial.println();
        //         break;
        //     }
        //     else
        //     {
        //         Serial.print(", ");
        //     }
        // }

        encodePacket();

        newData = false;
    }

    // Serial.print(">Packet data: ");
    // for (int i = 0; i < 8; i++)
    // {
    //     Serial.print(cmd_packet[i]);
    //     if (i == 7)
    //     {
    //         Serial.println();
    //     }
    //     else
    //     {
    //         Serial.print(", ");
    //     }
    // }

    // Send packet
    CAN.beginPacket(1);
    for (int i = 0; i < 8; i++)
    {
        CAN.write(cmd_packet[i]);
    }
    CAN.endPacket();

    int packetSize = CAN.parsePacket();
    // non zero packet size means things waiting in buffer
    if (packetSize && packetSize == 7)
    {
        byte packet[8];
        for (int i = 0; i < 7; i++)
        {
            packet[i] = CAN.read();
        }
        decodePacket(packet);
    }

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

void recvWithStartEndMarkers()
{
    static boolean recvInProgress = false;
    static byte ndx = 0;
    byte rb;

    while (Serial.available() > 0 && newData == false)
    {
        rb = Serial.read();

        if (recvInProgress == true)
        {
            if (rb != endMarker)
            {
                receivedBytes[ndx] = rb;
                ndx++;
                if (ndx >= numBytes)
                {
                    ndx = numBytes - 1;
                }
            }
            else
            {
                receivedBytes[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        else if (rb == startMarker)
        {
            recvInProgress = true;
        }
    }
}

// void sendWithStartEndMarkers(uint8_t id, float p, float v, float t, float vb)
// {
//     char buffer[1 + 1 + 4 * sizeof(float) + 1]; // Buffer size includes start/end markers and data

//     buffer[0] = startMarker;
//     buffer[1] = id; // Assuming the ID is directly after the start marker
//     // Copy the floats into the buffer, starting after the ID
//     memcpy(buffer + 2, &p, sizeof(p));
//     memcpy(buffer + 2 + sizeof(p), &v, sizeof(v));
//     memcpy(buffer + 2 + sizeof(p) + sizeof(v), &t, sizeof(t));
//     memcpy(buffer + 2 + sizeof(p) + sizeof(v) + sizeof(t), &vb, sizeof(vb));
//     buffer[sizeof(buffer) - 1] = endMarker; // Set the last byte as the end marker

//     Serial.write(buffer, sizeof(buffer));
// }

// Purpose: Given a command, modify the packet array
// Note: id is 11 bits, packet can contain up to 8 bytes of data
void encodePacket()
{
    if (receivedBytes[0] == 0xFF)
    { // Check for special command pattern
        memcpy(cmd_packet, receivedBytes, 8);
    }
    else
    { // Handle regular command packets
        float p_des, v_des, i_ff;
        uint16_t kp, kd;

        memcpy(&p_des, &receivedBytes[1], sizeof(p_des));
        memcpy(&v_des, &receivedBytes[1 + sizeof(p_des)], sizeof(v_des));
        memcpy(&kp, &receivedBytes[1 + sizeof(p_des) + sizeof(v_des)], sizeof(kp));
        memcpy(&kd, &receivedBytes[1 + sizeof(p_des) + sizeof(v_des) + sizeof(kp)], sizeof(kd));
        memcpy(&i_ff, &receivedBytes[1 + sizeof(p_des) + sizeof(v_des) + sizeof(kp) + sizeof(kd)], sizeof(i_ff));

        Serial.println(">rx_data:" + String(p_des) + "," + String(v_des) + "," + String(kp) + "," + String(kd) + "," + String(i_ff));

        uint16_t pos_cmd = float_to_uint(p_des, P_MIN, P_MAX, 16); // 16 bits 65535
        uint16_t vel_cmd = float_to_uint(v_des, V_MIN, V_MAX, 12); // 12 bits 4095
        uint16_t ffi_cmd = float_to_uint(i_ff, I_MIN, I_MAX, 12);  // 12 bits

        // Packing the commands into the packet array
        cmd_packet[0] = pos_cmd >> 8;                               // Position command high byte
        cmd_packet[1] = pos_cmd & 0xFF;                             // Position command low byte
        cmd_packet[2] = (vel_cmd >> 4) & 0xFF;                      // Velocity command high part
        cmd_packet[3] = ((vel_cmd & 0xF) << 4) | ((kp >> 8) & 0xF); // Velocity low part + Kp high part
        cmd_packet[4] = kp & 0xFF;                                  // Kp low byte
        cmd_packet[5] = kd >> 4;                                    // Kd high part
        cmd_packet[6] = ((kd & 0xF) << 4) | ((ffi_cmd >> 8) & 0xF); // Kd low part + Feed-forward current high part
        cmd_packet[7] = ffi_cmd & 0xFF;                             // Feed-forward current low byte
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
    // for (size_t i = 0; i < 8; i++)
    // {
    //     Serial.print(packet[i], HEX);
    //     Serial.print(" "); // Add space between bytes for readability
    // }
    // Serial.println(); // End of line after printing the array

    // Decoding the packet from bin
    uint8_t id = packet[0];
    uint16_t p_raw = (packet[1] << 8) | packet[2];
    uint16_t v_raw = (packet[3] << 4) | (packet[4] >> 4);
    uint16_t t_raw = ((packet[4] & 0x0F) << 8) | packet[5];
    uint8_t vb_raw = packet[6];

    int16_t signed_t_raw = (int16_t)t_raw; // Cast to signed 16-bit integer needed for arithmetic
    signed_t_raw += (abs(signed_t_raw - last_t_raw) > 4000) ? ((signed_t_raw < last_t_raw) ? 4096 : -4096) : 0;
    last_t_raw = signed_t_raw;

    float p = uint_to_float(p_raw, P_MIN, P_MAX, 16);
    float v = uint_to_float(v_raw, V_MIN, V_MAX, 12);
    float t = int_to_float_overflow(signed_t_raw, I_MIN, I_MAX, 12); // Special treatment for current, it might go out of range and wrap around.
    float vb = uint_to_float(vb_raw, VB_MIN, VB_MAX, 8);

    // Printing decoded values
    // Serial.println();
    // Serial.print(">Position[rad]:");
    // Serial.println(p); // Serial.print(",");
    // Serial.print(">Velocity[rad/s]:");
    // Serial.println(v); // Serial.print(",");
    // Serial.print(">Current[A]:");
    // Serial.println(t); // Serial.print(",");
    // Serial.print(">Voltage[V]:");
    // Serial.println(vb); // Serial.print(",");

    Serial.println(">tx_data:" + String(id) + "," + String(p) + "," + String(v) + "," + String(t) + "," + String(vb));

    // sendWithStartEndMarkers(id, p, v, t, vb);
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
