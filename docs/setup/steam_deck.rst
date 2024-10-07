Steam Deck
==========

Steam Deck is an important part of our teleopration device, which
controls the head movement and lower body skills.

Install VSCode
--------------

1. Install VSCode from the built-in app store. However, VSCode shells
   somehow don’t work for us. We recommend directly using the terminals.

Unlock the Filesystem
---------------------

By default, Steam Deck’s filesystem is read-only, which disallows
``sudo`` access.

Follow the instructions `here <TODO>`__ to unlock the filesystem.

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
