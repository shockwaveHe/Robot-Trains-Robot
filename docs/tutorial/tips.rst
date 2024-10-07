Debug tips
==========

1. Magnet could be rubbing with the encoder, causing large current and
   jittering motion
2. If you find it hard plugging in the male connector to the female
   connector, check the pins in the female connector. Pins could be bent
   (especially SMALL ones), causing no connection
3. Cables could break (especially CAN bus), check with multimeter if
   connection is unstable
4. Build before flash the ESC control code
5. Don’t drag the cables when removing the connector. Use the plastic
   tips on both sides
6. Tape the DC power supply voltage and current buttons to avoid
   accidentally changing the settings

Dynamixel XC330 can not sustain 4Mbps baudrate, it will raise a bunch of
warnings (no status packet). Please down grade all xc330 to 2Mbps.

Jetson Tips
===========

-  We recommand using jtop to monitor the performance of the system

Good coding practices
=====================

1. Use dataclass and argparse if possible
2. Write Google style docstring
3. Write type hint
4. Assert, and raise errors if possible
5. Use pure functions if possible
6. Put the magic numbers together in one place
7. Write inline document if possible
8. Use shell scripts
9. Consider writing unit tests
