.. _api:

API Reference
=================

The codebase is organized as follows:

.. code:: bash

   .
   ├── datasets  # store the manipulate datasets
   ├── docs  # store the source files of the documentation
   ├── examples  # some example scripts to get started
   ├── motion  # store the keyframe motion data
   ├── results  # store the results of the experiments
   ├── scripts  # store the utility scripts
   ├── tests   # store the test scripts
   ├── toddlerbot  # the main codebase
   │   ├── actuation  # code for interacting with Dynamixel Motors
   │   ├── algorithms  # zero moment point planner
   │   ├── brax  # our fork of Google's Brax
   │   ├── descriptions  # URDF, MJCF, and config files for all the variations of Toddy
   │   ├── locomotion  # RL training code in MJX
   │   ├── manipulation  # diffusion policy
   │   ├── policies  # deployment code with run_policy.py as the runner and others as policy classes.
   │   ├── reference  # code to generate reference motion
   │   ├── sensing  # code to interact with the sensors
   │   ├── sim   # code for information exchange with MuJoCo and the real world sensors
   │   ├── tools  # zero_point calibration, keyframe editing, joystick teleoperation, system identification, sim2real evaluation, etc.
   │   ├── utils  # utility functions
   │   └── visualization  # visualization functions

The Docstrings are generated using ChatGPT and Sphinx. 
Please open an GitHub issue if you find any errors or have any suggestions.

.. toctree::
   :maxdepth: 4

   ../api/modules