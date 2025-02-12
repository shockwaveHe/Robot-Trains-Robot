.. _steam_deck:

Steam Deck
==========

Steam Deck is another option for remote control. It is a handheld-sized desktop computer that runs Arch Linux.

Install VSCode
--------------

Install VSCode from the built-in app store. However, VSCode shells don't work for us. We recommend directly using the terminals
to run scripts.

Unlock the Filesystem
---------------------

By default, Steam Deck's filesystem is read-only.

Follow the instructions `here <https://christitus.com/unlock-steam-deck/>`__ to unlock the filesystem.


Access to USB Devices
---------------------

Add the user to the ``uucp`` group by running the following command:

::

   sudo usermod -aG uucp $USER

Access to the Joystick
----------------------

We find that Steam overrides the Joystick access. Therefore, to access
the joystick device from Python, you need to make sure to **shut down
Steam** before running the scripts.

Test the Joystick by running this script:

::

   python tests/test_joystick.py


NTP Server
-----------------------------
For the accuracy of teleoperation and logging over network, we need to
install NTP package to sync time of Jetson to Steam Deck.

Run the following commands to set it up:

::

   sudo pacman -S ntp
   sudo systemctl enable ntp

   nano /etc/ntp.conf

   add:

   restrict <client_ip_address> mask 255.255.255.255 nomodify notrap

   sudo systemctl start ntp
