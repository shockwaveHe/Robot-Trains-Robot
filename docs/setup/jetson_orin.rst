
.. _jetson_orin:

Jetson Orin
===========

Jetson Orin is the on-board compute for the toddlerbot. We recommend
Jetson Orin NX 16GB due to its cost effectiveness.

Flash the System
----------------

#. Follow the instructions on `this page <https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/#flash-jetpack>`__ to flash the system.
   We've also provided some tips below to help you through the process.

   - For the Enter Force Recover Mode, you can refer to this photo if the GIF on the page is not clear.

      .. image:: ../_static/setup/jetson_flash_pins.png

   - Click on JP6.0 tab and download `the image corresponding to Jetson Orin NX 16GB <https://szseeedstudio-my.sharepoint.cn/:u:/g/personal/youjiang_yu_szseeedstudio_partner_onmschina_cn/EbEZRxHDtgBDjBrHK_7ltfEB6JBa3VGXLx3meNc0OJUL_g?e=8MNsTg>`__. It took us 1 hour to download the image.

#. After flashing, unplug the powercable, the USB-C cable and the jumper wire. Replug the power cable, the HDMI cable, the keyboard and the mouse.
   The system should boot up now. The start screen should look like this:

   .. image:: ../_static/setup/jetson_start_screen.jpg

#. Enter your username and password.

#. Set the APP Partition to the max size.

#. No need to install the Chromium browser.

#. Power off the Jetson Orin and install the WiFi card like the photo below.

   .. image:: ../_static/setup/jetson_wifi.jpg

#. Power on the Jetson Orin and connect to the WiFi.

#. Press :kbd:`Ctrl` + :kbd:`Alt` + :kbd:`T` to open the terminal. Run ``ifconfig`` to check the IP address of the Jetson Orin.
   Write down the IP address and the hostname of the Jetson Orin, e.g., ``192.168.0.237`` and ``toddy@toddy-desktop``.

#. Select the power mode on the top right corner to **0:MAXN**.

#. Now feel free to unplug the HDMI cable, the keyboard and the mouse. We will access the Jetson Orin through SSH for the following steps.

.. note::
   The USB-C port on Jetson is only for flashing, which means transfering data
   through this port won't work.


Set up the Real-Time Kernel
---------------------------
#. SSH into the Jetson Orin.

   .. code:: bash

      ssh toddy@toddy-desktop.local

#. Follow the instructions on `this page <https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html#real-time-kernel-using-ota-update>`__ to install the real-time kernel.
   We found that it's OK to ignore the warning message below, but please let us know if you encounter any issues:

   .. code:: bash

      Errors were encountered while processing:
      nvidia-l4t-rt-kernel
      nvidia-l4t-rt-kernel-oot-modules
      nvidia-l4t-rt-kernel-headers
      nvidia-l4t-display-rt-kernel
      E: Sub-process /usr/bin/dpkg returned an error code (1)

#. After rebooting, check the kernel version with ``uname -r``. It should be ``5.15.136-rt-tegra``. The numbers may vary, but ``-rt-tegra`` should be there.

Set up Conda and the repo
-------------------------
#. See :ref:`general_setup`. ``wget`` is handy for downloading the installer from the terminal.

#. Besides the general setup, we need to install a few packages specifically for Jetson Orin. Let's start with the GPIO.

   Revised from instructions on `this page <https://github.com/NVIDIA/jetson-gpio>`__, we need to run the following commands to set it up:

.. code:: bash

   sudo groupadd -f -r gpio
   sudo usermod -aG gpio $USER
   sudo chown root.gpio /dev/gpiochip0
   sudo chmod 660 /dev/gpiochip0

   conda activate toddlerbot
   pip install Jetson.GPIO
   sudo cp ~/miniforge3/envs/toddlerbot/lib/python3.10/site-packages/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger

#.  To access the dynamixel motors and the imu, we need to add the user to i2c and dialout group.

.. code:: bash

   sudo usermod -aG i2c $USER
   sudo usermod -aG dialout $USER


#. Now reboot the Jetson Orin. Run ``groups`` to check if the user is in the i2c, dialout, and gpio group. An example output looks like this:

.. code:: bash

   toddy adm dialout cdrom sudo audio dip video plugdev render i2c lpadmin sambashare gdm weston-launch gpio


Initial test
--------------

Based on the instructions in :ref:`electronics`, Connect the dynamixel motors and the imu to the Jetson Orin and mount it on Toddy. 
Run the following commands to test if everything works properly.

.. code:: bash

   python -m toddlerbot.policies.run_policy --policy stand --sim real --robot toddlerbot


Teleoperation
--------------
For the accuracy of teleoperation and logging over network, we need to
install ntp package to sync time of the jetson to server.

.. code:: bash

   sudo apt install ntp ntpdate
   sudo systemctl enable ntp
   
   sudo nano /etc/ntp.conf

   comment out the following lines:

   # pool 0.ubuntu.pool.ntp.org iburst
   # pool 1.ubuntu.pool.ntp.org iburst
   # pool 2.ubuntu.pool.ntp.org iburst
   # pool 3.ubuntu.pool.ntp.org iburst
   # pool ntp.ubuntu.com

   add:

   server <ip address of the steam deck> iburst

   sudo systemctl start ntp

For the fisheye cameras to work, we need to install the following packages:

.. code:: bash

   sudo apt install v4l-utils ffmpeg

For the speaker and microphone to work, we need to install the following packages:

.. code:: bash

   sudo apt install portaudio19-dev flac


(Optional) Set up sshfs
-----------------------
TODO