.. _software:

Getting Started (Software)
=================================

.. toctree::
   :maxdepth: 1
   :hidden:

   ../software/01_setup
   ../software/02_jetson_orin
   ../software/03_rog_ally_x
   ../software/04_steam_deck

Welcome! This page summarizes the process you should follow to set up the software stack of ToddlerBot. 
In each step, there are pointers to the detailed instructions.

Your Laptop or Workstation
-----------------------------

The typical workflow involves developing code on a laptop or workstation while working in simulation, 
then deploying it to ToddlerBot by running it on the Jetson Orin. For example, I develop the code 
on my MacBook Pro and use SSH to access the Jetson Orin to execute code directly on the robot.

To set up your laptop or workstation, simply follow the instructions in the :ref:`setup` section.

Jetson Orin
----------------

The Jetson serves as the brain of ToddlerBot—a powerful compact computer capable of up to 100 TOPS 
of inference with 16GB of shared RAM and VRAM. 
Essentially, it's an ARM64 system equipped with an NVIDIA GPU, running Ubuntu 22.04.  
For setup instructions, see :ref:`jetson_orin`.

Remote Controller
-----------------------

We provide two options for remote control: ROG Ally X or Steam Deck. The setup instructions are in the
:ref:`rog_ally_x` and :ref:`steam_deck` sections, respectively.


Codebase
-----------------------

The detailed API documentation is in the :ref:`api` section.