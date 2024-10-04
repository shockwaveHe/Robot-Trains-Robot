### Debug tips
1. Magnet could be rubbing with the encoder, causing large current and jittering motion
1. If you find it hard plugging in the male connector to the female connector, check the pins in the female connector. Pins could be bent (especially SMALL ones), causing no connection
1. Cables could break (especially CAN bus), check with multimeter if connection is unstable
1. Build before flash the ESC control code
1. Don't drag the cables when removing the connector. Use the plastic tips on both sides
1. Tape the DC power supply voltage and current buttons to avoid accidentally changing the settings

Dynamixel XC330 can not sustain 4Mbps baudrate, it will raise a bunch of warnings (no status packet). Please down grade all xc330 to 2Mbps.

### Jetson Tips
- We recommand using jtop to monitor the performance of the system


### Good coding practices
1. Use dataclass and argparse if possible
1. Write Google style docstring
1. Write type hint
1. Assert, and raise errors if possible
1. Use pure functions if possible
1. Put the magic numbers together in one place
1. Write inline document if possible
1. Use shell scripts
1. Consider writing unit tests