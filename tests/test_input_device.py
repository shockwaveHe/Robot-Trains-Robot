import evdev

# List all input devices
devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

for device in devices:
    print(f"Device: {device.path}")
    print(f"  Name: {device.name}")

    # Get device information
    info = device.info
    print(f"  Vendor: {info.vendor}")
    print(f"  Product: {info.product}")

    # Check if it's a joystick
    if "joystick" in device.name.lower():
        print("  This is a joystick!")

        # Check for common vendor/product IDs for Steam Deck or Xbox controller
        if info.vendor == 0x28DE and info.product == 0x1205:
            print("  Detected: Steam Deck Controller")
        elif info.vendor == 0x045E and info.product in [0x028E, 0x02EA]:
            print("  Detected: Xbox Controller")
        else:
            print("  Unknown joystick device.")
    print()
