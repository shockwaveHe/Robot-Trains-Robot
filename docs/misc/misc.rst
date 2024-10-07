Install legged_gym
==================

Follow `these
instructions <https://github.com/leggedrobotics/legged_gym>`__.

Install IssacGym Preview Release 4 from `this
website <https://developer.nvidia.com/isaac-gym>`__.

Change line 135 in ``isaacgym/python/isaacgym/torch_utils.py`` from

::

   def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):

to

::

   def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):

Run

::

   pip install tensorboard

Change
``~/anaconda3/envs/toddlerbot/lib/python3.10/site-packages/torch/utils/tensorboard/__init__.py``
from

::

   import tensorboard
   from setuptools import distutils
   LooseVersion = distutils.version.LooseVersion

   # ...

   del distutils

   # ...

to

::

   import tensorboard
   # from setuptools import distutils
   # LooseVersion = distutils.version.LooseVersion
   from distutils.version import LooseVersion

   # ...

   # del distutils

   # ... 

According to `this
issue <https://github.com/tensorflow/tensorboard/issues/6808>`__, run

::

   pip install protobuf==4.25

Install Isaac Sim
=================

Isaac Sim 4.1

::

   conda activate isaaclab
   ./isaaclab.sh -i

   # ModuleNotFoundError: No module named 'omni.isaac.kit':
   source _isaac_sim/setup_conda_env.sh

Working with Arduino and ESC boards
===================================

Install Arduino in VSCode
-------------------------

MacOS

::

   brew install arduino-cli

Linux Download the binary `from this
website <https://arduino.github.io/arduino-cli/0.35/installation/#latest-release>`__.
Put ``arduino-cli`` in ``/usr/bin``.

Install the Arduino VSCode extension.

Open VSCode settings: 1. Set Arduino Command Path: arduino-cli 1. Set
Arduino Path: /opt/homebrew/bin (MacOS) or /usr/bin (Linux) 1. Set Use
Arduino Cli: yes

Use the VSCode Adafruit Board Manager to install
``Adafruit SAMD Boards``.

Use the VSCode Adafruit Library Manager to install ``Adafruit CAN``.

Follow this
`tutorial <https://learn.adafruit.com/adafruit-feather-m4-can-express/arduino-ide-setup>`__
if you have other issues.

::

   screen /dev/tty.usbxxxxx 115200

Flush code to the ESC board
---------------------------

::

   brew install open-ocd
   brew install gcc-arm-embedded

Install these VSCode extensions: 1. C/C++ 1. C/C++ Extension Pack 1.
Cortex-Debug 1. Makefile Tools

::

   cd toddlerbot/actuation/sunny_sky/knee_esc
   cmake .
   make

Add the following configuration to launch.json:

::

   {
       "cwd": "${workspaceRoot}",
       "executable": "toddlerbot/actuation/sunny_sky/knee_esc/Release/motorcontrol.elf",
       "name": "Flush Knee ESC",
       "request": "launch",
       "type": "cortex-debug",
       "servertype": "openocd",
       "configFiles": [
           "toddlerbot/actuation/sunny_sky/knee_esc/daplink.cfg"
       ],
       "searchDir": [],
       "runToEntryPoint": "main",
       "showDevDebugOutput": "none"
   }

Connect the ESC board to the computer. Run Flush Knee ESC from the “Run
and Debug” panel.

Restart the power and reconnect the ESC board. Press ``E`` and then
``Esc`` to check if the menu shows up.

Sunny Sky Brushless motors
--------------------------

1. Check the cables first
2. Unscrew the panel support and calibrate the motor. Make sure the pole
   pairs are 12.
3. Run ``toddlerbot/actuation/sunny_sky/sunny_sky_control.py`` as a unit
   test

Realsense setup
---------------

1. We will refer to `this
   guide <https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md>`__
   to install the librealsense on jetson.
