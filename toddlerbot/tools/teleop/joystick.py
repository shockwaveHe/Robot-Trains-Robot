import pygame
from pygame.joystick import JoystickType


def initialize_joystick():
    # Initialize Pygame
    try:
        pygame.init()
        # Initialize the joystick
        pygame.joystick.init()
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    except Exception:
        joystick = None

    return joystick


def get_controller_input(joystick: JoystickType, dead_zone: float = 0.05):
    # Process pygame events
    pygame.event.pump()

    # Get joystick axes (assuming standard Xbox controller)
    axis_0 = joystick.get_axis(0)  # Left stick horizontal (for linear velocity y)
    axis_1 = joystick.get_axis(1)  # Left stick vertical (for linear velocity x)
    axis_3 = joystick.get_axis(
        3
    )  # Right stick horizontal (for angular velocity z and heading direction)

    # Adjust axis values (e.g., invert axis if needed, apply scaling, etc.)
    linear_vel_x = -axis_1  # Inverting because pushing stick up gives negative values
    linear_vel_y = -axis_0  # Inverting because pushing stick left gives negative values
    angular_vel_z = axis_3

    # Apply dead zones or thresholds for more precise control
    linear_vel_x = 0 if abs(linear_vel_x) < dead_zone else linear_vel_x
    linear_vel_y = 0 if abs(linear_vel_y) < dead_zone else linear_vel_y
    angular_vel_z = 0 if abs(angular_vel_z) < dead_zone else angular_vel_z

    return [linear_vel_x, linear_vel_y, angular_vel_z]


if __name__ == "__main__":
    try:
        joystick = initialize_joystick()
        assert joystick is not None, "No joystick found."

        while True:
            # Get the mapped controller input
            command = get_controller_input(joystick)

            # Print the command
            print(
                f"Command: [Linear X: {command[0]:.2f}, Linear Y: {command[1]:.2f}, Angular Z: {command[2]:.2f}]"
            )

            # Add a small delay to avoid spamming
            pygame.time.wait(100)

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        pygame.quit()
