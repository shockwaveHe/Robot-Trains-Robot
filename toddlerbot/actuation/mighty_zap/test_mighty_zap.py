import sys
import time

import serial

from toddlerbot.actuation.mighty_zap import mighty_zap

MightyZap = mighty_zap

Actuator_ID = 0

MightyZap.OpenMightyZap("/dev/ttyUSB0", 57600)
time.sleep(0.1)
pos = 0

MightyZap.GoalPosition(Actuator_ID, 3000)
while pos < 2990:
    pos = MightyZap.PresentPosition(Actuator_ID)
    print(pos)

MightyZap.GoalPosition(Actuator_ID, 0)
while pos > 10:
    pos = MightyZap.PresentPosition(Actuator_ID)
    print(pos)
