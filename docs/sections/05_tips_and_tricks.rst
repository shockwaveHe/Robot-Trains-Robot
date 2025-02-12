.. _tips:

Tips and Tricks
===============

Here are some tips and tricks to help you get started with ToddlerBot.
Please read them carefully to avoid common pitfalls. We will continue updating this section
as more pop up in our mind.


Hardware
------------
- When using the DC power supply, tape the voltage and current buttons to avoid
  accidentally changing the settings.

- Make sure to use the voltage tester when running the robot on battery power to
  avoid over-discharging the battery. This is IMPORTANT to prevent damaging the battery.

- With the current comminucation setup, Dynamixel XC330 can not sustain 4Mbps baudrate, it will raise a bunch of
  warnings (no status packet). Please make sure to use 2Mbps baudrate for all the motors.

- Make sure the fan of Jetson Orin is not covered by clothes or linens to avoid overheating.

- Make sure to secure the robot before you kill a script than controls the motors or turn off the power. 
  The robot may fall and get damaged. But no need to be too cautious, the robot is robust and easy to repair.

- Sometimes when the battery is low, the robot may not be able to do tasks that require high torque. 
  You can check the voltage tester to see if the battery is low. If so, please charge the battery.

- If you insert the battery box into the torso in a correct pose, the cover knobs should close easily. 
  If you find it hard to close the front cover, please check the pose of the battery box and organize the wires.

- Due to limited space budget, we do not have enough space for a HDMI cable to connect to the Jetson from the bottom. 
  You can unscrew the Jetson from the torso and then connect the HDMI cable to the Jetson. 
  We find that it's easier to use SSH to connect to the Jetson in most cases.

- Repeated screwing and unscrewing may strip some holes on the 3D-printed parts. If this happens, you can simply reprint the part.

- If you ever find yourself facing a bug that occurs sporadically and goes away by itself. Double check all the wirings. It could be a loose cable somewhere.
  For example, we once found dynamixel wizard cannot identify any motor, but if we only plug in an individual motor chain, it works no problem. 
  All the motor and u2d2 are function. The problem occasionally disappears and all motors comes back. Eventually, we found a loose cable near the ankle. 
  It's just loose enough to disconnect on motion and come back to contact after. It is suspected to dump stuff into the data line when it reboots and mess with the communication of the rest of the motors. 

- Doing a standing test from time to time is very helpful to identify loose screws. You can do this by holding the robot in the air
  with the standing policy running, and use your hand to sense the amount of backlash in the joints. If you feel the backlash is too much,
  it may be caused by loose screws. Tighten the screws to fix the issue. Some joints with XC330s tend to have larger backlash than others, which is normal.

- The waist joint can suffer from wear and tear over time. If you find the waist joint is loose, you can reprint the gears and replace them.
  We find that PLA-CF with more wall loops for all the gears involved in the waist joint can increase the lifespan of the gears.


Software
------------
- We recommand using jtop to monitor the performance of Jetson Orin. You can install it by running:
  ::
  
    pip install jetson-stats

- If you encounter the following error:
  ::
  
    ('Was not able to enable feature', 4)

  Make sure the IMU's pins are connected correctly. Replug the IMU and try again.

- If you see the following message:
  "Voltage too low. Please check the power supply or charge the batteries."
  Stop the robot immediately and check the power supply.


- If you encounter the following error:
  ::
  
    Cannot set torque disabled / enabled...

  Use the EStop button to reboot the motors.

- If you encouter lagging issues with the Dynamixel motors, you can try to set the latency timer to 1.
  If you're on macOS, according to the discussion `here <https://openbci.com/forum/index.php?p=/discussion/3108/driver-latency-timer-fix-for-macos-11-m1-m2>`__ and
  `in this blog post <https://www.mattkeeter.com/blog/2022-05-31-xmodem/#ftdi>`__,
  we need to run a small C program on Mac to set the latency timer to 1.
  Run the following commands to set it up:
  ::

     brew install libftdi
     cd toddlerbot/actuation/latency_timer_setter_macOS
     cc -arch arm64 -I/opt/homebrew/include/libftdi1 -L/opt/homebrew/lib -lftdi1 main.c -o set_latency_timer
     ./set_latency_timer

- If you encounter the following warning:
  ::

   [Warning] [Dynamixel] > bulk_read: [TxRxResult] Incorrect (No) status packet!

  If only a few show up in the terminal, there's no need to worry. However, if many are, it may indicate a poor connection
  or wrong configuration to initialize the motors, e.g., setting the ``--robot toddlerbot_gripper`` argument when running the 
  script on ToddlerBot with a compliant palm gripper.

- To set up headless rending with MuJoCo, add ``export MUJOCO_GL="egl"`` to your ``.bashrc``. Then ``source ~/.bashrc``.

- If you want to view the camera feed or interactive visualization on the Jetson Orin through an ssh session, you can follow 
  `this guide <https://www.businessnewsdaily.com/11035-how-to-use-x11-forwarding.html>` to set up X11 forwarding.

