import time

from toddlerbot.actuation.mighty_zap import mighty_zap

Actuator_ID_1 = 0
Actuator_ID_2 = 1

# Open connection to MightyZap
mighty_zap.OpenMightyZap("/dev/tty.usbserial-0001", 57600)
time.sleep(0.1)

# Set both actuators to their goal positions simultaneously
mighty_zap.GoalPosition(Actuator_ID_1, 3000)
mighty_zap.GoalPosition(Actuator_ID_2, 3000)

# Wait until both actuators reach their target position
while True:
    pos1 = mighty_zap.PresentPosition(Actuator_ID_1)
    pos2 = mighty_zap.PresentPosition(Actuator_ID_2)
    print(f"Actuator 1 Position: {pos1}, Actuator 2 Position: {pos2}")

    # Check if both actuators have reached their target positions
    if pos1 >= 2990 and pos2 >= 2990:
        break  # Exit the loop if both have reached or surpassed the target position


# Set both actuators back to position 0 simultaneously
mighty_zap.GoalPosition(Actuator_ID_1, 3000)
mighty_zap.GoalPosition(Actuator_ID_2, 0)

# Wait until both actuators return to their initial position
while True:
    pos1 = mighty_zap.PresentPosition(Actuator_ID_1)
    pos2 = mighty_zap.PresentPosition(Actuator_ID_2)
    print(f"Actuator 1 Position: {pos1}, Actuator 2 Position: {pos2}")

    # Check if both actuators have returned to position 0
    if pos1 >= 2990 and pos2 <= 10:
        break  # Exit the loop if both are close enough to position 0


# Set both actuators back to position 0 simultaneously
mighty_zap.GoalPosition(Actuator_ID_1, 0)
mighty_zap.GoalPosition(Actuator_ID_2, 3000)

# Wait until both actuators return to their initial position
while True:
    pos1 = mighty_zap.PresentPosition(Actuator_ID_1)
    pos2 = mighty_zap.PresentPosition(Actuator_ID_2)
    print(f"Actuator 1 Position: {pos1}, Actuator 2 Position: {pos2}")

    # Check if both actuators have returned to position 0
    if pos1 <= 10 and pos2 >= 2990:
        break  # Exit the loop if both are close enough to position 0

# Set both actuators back to position 0 simultaneously
mighty_zap.GoalPosition(Actuator_ID_1, 0)
mighty_zap.GoalPosition(Actuator_ID_2, 0)

# Wait until both actuators return to their initial position
while True:
    pos1 = mighty_zap.PresentPosition(Actuator_ID_1)
    pos2 = mighty_zap.PresentPosition(Actuator_ID_2)
    print(f"Actuator 1 Position: {pos1}, Actuator 2 Position: {pos2}")

    # Check if both actuators have returned to position 0
    if pos1 <= 10 and pos2 <= 10:
        break  # Exit the loop if both are close enough to position 0
