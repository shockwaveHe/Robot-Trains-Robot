import argparse
import time

from tqdm import tqdm

from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.misc_utils import log

# This script is used to teleoperate the robot using a joystick. Note that this
# script is not for teleoperated data collection, instead it is for
# pupperteering the robot in real-time.


def main(ip: str):
    """Initializes and runs a teleoperation control loop, sending joystick inputs over a ZMQ connection.

    Args:
        ip (str): The IP address for the ZMQ connection.

    The function sets up a ZMQ sender node and a joystick interface, then enters a loop to continuously read joystick inputs. It toggles logging based on a specific button press and sends control inputs as messages over the ZMQ connection. The loop runs at a fixed control rate, updating a progress bar to indicate activity. The loop can be interrupted with a keyboard interrupt, which gracefully closes the progress bar.
    """

    zmq = ZMQNode(type="sender", ip=ip)
    joystick = Joystick()

    header_name = "Teleoperation"
    p_bar = tqdm(desc="Running the teleoperation")
    start_time = time.time()
    step_idx = 0
    time_until_next_step = 0.0
    control_dt = 0.02
    is_running = False
    is_button_pressed = False

    try:
        while True:
            control_inputs = joystick.get_controller_input()
            for task, input in control_inputs.items():
                if task == "teleop":
                    if abs(input) > 0.5:
                        # Button is pressed
                        if not is_button_pressed:
                            is_button_pressed = True  # Mark the button as pressed
                            is_running = not is_running  # Toggle logging

                            print(
                                f"\nLogging is now {'enabled' if is_running else 'disabled'}.\n"
                            )
                    else:
                        # Button is released
                        is_button_pressed = False  # Reset button pressed state

            # compile data to send to follower
            msg = ZMQMessage(time=time.time(), control_inputs=control_inputs)
            # print(f"Sending: {msg}")
            zmq.send_msg(msg)

            step_idx += 1

            p_bar_steps = int(1 / control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            step_end = time.time()

            time_until_next_step = start_time + control_dt * step_idx - step_end
            # print(f"time_until_next_step: {time_until_next_step * 1000:.2f} ms")
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        log("KeyboardInterrupt recieved. Closing...", header=header_name)

    finally:
        p_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the teleoperation.")
    parser.add_argument(
        "--ip",
        type=str,
        default="",
        help="The ip address of toddy.",
    )
    args = parser.parse_args()

    main(args.ip)
