import pygame

# Test script to read joystick inputs using Pygame

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    # Check if joystick is available
    if pygame.joystick.get_count() == 0:
        print("No joysticks found.")
        pygame.quit()
        exit()

    # Initialize the first joystick (js0)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Initialized joystick: {joystick.get_name()}")

    # Loop to capture joystick inputs
    try:
        while True:
            # Handle Pygame events
            pygame.event.pump()

            # Read axis values
            axes = joystick.get_numaxes()
            for i in range(axes):
                axis = joystick.get_axis(i)
                print(f"Axis {i} value: {axis}")

            # Read button values
            buttons = joystick.get_numbuttons()
            for i in range(buttons):
                button = joystick.get_button(i)
                print(f"Button {i} value: {button}")

            # Read hat (D-pad) values
            hats = joystick.get_numhats()
            for i in range(hats):
                hat = joystick.get_hat(i)
                print(f"Hat {i} value: {hat}")

            # Add a small delay to avoid spamming the console
            pygame.time.wait(100)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Quit Pygame and clean up
        joystick.quit()
        pygame.quit()
