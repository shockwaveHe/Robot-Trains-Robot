import pygame


def listen_to_xbox_controller_pygame():
    """Listens to Xbox Wireless Controller inputs using pygame and prints them."""
    pygame.init()
    pygame.joystick.init()  # Initialize the joystick module

    # Check for joystick count
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No gamepad connected.")
        return

    # Initialize the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    try:
        # Loop to keep checking for events
        running = True
        while running:
            for event in pygame.event.get():  # User did something
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button {event.button} pressed.")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Button {event.button} released.")
                elif event.type == pygame.JOYAXISMOTION:
                    print(f"Joystick {event.axis} moved to {event.value}.")
    except KeyboardInterrupt:
        print("Listening stopped by user.")
    finally:
        # Properly shutdown Pygame
        pygame.quit()


listen_to_xbox_controller_pygame()
