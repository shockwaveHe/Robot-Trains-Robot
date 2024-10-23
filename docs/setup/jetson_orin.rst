
.. _jetson_orin:

Jetson Orin
===========

Jetson Orin is the on-board compute for the toddlerbot. We recommend
Jetson Orin NX 16GB.

Flash the System
----------------

#. Follow the instructions on `this page <https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/#flash-jetpack>`__ to flash the system.
   We've also provided some tips below to help you through the process.

   - For the Enter Force Recover Mode, you can refer to this photo if the GIF on the page is not clear.

      .. image:: ../_static/setup/jetson_flash_pins.png

   - Click on JP6.0 tab and download `the image corresponding to Jetson Orin NX 16GB <https://szseeedstudio-my.sharepoint.cn/:u:/g/personal/youjiang_yu_szseeedstudio_partner_onmschina_cn/EbEZRxHDtgBDjBrHK_7ltfEB6JBa3VGXLx3meNc0OJUL_g?e=8MNsTg>`__.
      It took us 1 hour to download the image.

#. After flashing, unplug the powercable, the USB-C cable and the jumper wire. Replug the power cable, the HDMI cable, the keyboard and the mouse.
   The system should boot up now. The start screen should look like this:

   .. image:: ../_static/setup/jetson_start_screen.jpg

#. Enter your username and password.

#. Set the APP Partition to the max size.

#. No need to install the Chromium browser.

#. Power off the Jetson Orin and install the WiFi card like the photo below.

   .. image:: ../_static/setup/jetson_wifi.jpg

#. Power on the Jetson Orin and connect to the WiFi.

#. Select the power mode on the top right corner to **0:MAXN**.

.. note::
   The USB-C port on Jetson is only for flashing, which means transfering data
   through this port won't work.


-  Add yourself to the correct user group
   `doc <https://github.com/NVIDIA/jetson-gpio>`__
-  Be sure to modify the address of rules: e.g

.. code:: bash

   sudo groupadd -f -r gpio
   sudo usermod -a -G gpio $USER
   sudo chown root.gpio /dev/gpiochip0
   sudo chmod 660 /dev/gpiochip0

   python -m pip install Jetson.GPIO
   sudo cp ~/miniforge3/envs/toddlerbot/lib/python3.10/site-packages/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger

-  also add yourself to i2c and dialout group.

.. code:: bash

   sudo usermod -aG i2c $USER
   sudo usermod -aG dialout $USER


Set up the Real-Time Kernel
---------------------------




Set up Conda
------------
See :ref:`general_setup`.


Install PyTorch
----------------

Follow the instructions in setup/general.rst to install PyTorch.
Follow `these releases <https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048>`__
to install PyTorch. For reference, we downloaded `the wheel file for PyTorch v2.3.0 with JetPack 6.0 (L4T R36.2 / R36.3) + CUDA 12.2 <https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl>`__.

Assuming the toddlerbot conda environment is activated, install the wheel with:
::

   pip install <path/to/the/wheel>


Edit sudoers safely:

::

   sudo visudo

Add a line for specific commands:

::

   youruser ALL=(ALL) NOPASSWD: /bin/echo, /usr/bin/tee

This allows the user ``youruser`` to run echo and tee without a
password. Ensure you replace youruser with the actual user that the
script runs under.

Install miniforge: Download ``Linux aarch64 (arm64)`` from `their
website <https://github.com/conda-forge/miniforge>`__. Do NOT run the
install script with sudo. Answer ``yes`` to all the options.

For the accuracy of teleoperation and logging over network, we need to
install ntp package to sync time of the jetson to server.

::

   sudo apt install ntp
   sudo systemctl enable ntp


nano /etc/ntp.conf

comment out the following lines:

# pool 0.ubuntu.pool.ntp.org iburst
# pool 1.ubuntu.pool.ntp.org iburst
# pool 2.ubuntu.pool.ntp.org iburst
# pool 3.ubuntu.pool.ntp.org iburst
# pool ntp.ubuntu.com

add:

server <server_ip_address> iburst

::

   sudo systemctl start ntp
