=========
Getting Started
=========

Welcome! This page summarized the process you should follow to build and extend the whole toddlerbot system. In each step, there are pointers to the detailed instructions. Are you ready to make yourself a humanoid? Let's get started! 

Sourcing the Parts
-------------
The first step in building your Toddlerbot is to source the parts.
You can find a list of the parts you will need in the :ref:`bom` section.

.. note::
    You would want to order the PCb parts first! As they takes about 10 days from order to arrive. Check this page for the exact process to order the PCBs!

.. note::
    We highly recommand you get a **Bambulab X1C** to 3D print everything. It's relatively inexpensive compare to the toddlerbot, and it's gonna save you a lot of time and trouble trying to fit the tolerances. We also provided organized plates with all the correct number and orientations so that you can just hit print and go!

.. note::
    We also highly recommand getting an electric screw driver, both the big one and the small one, for your welfare.

Start Printing
-------------
As you wait for the orders and PCBs to arrive, you can start printing the parts! Suggested plate order is:

1. Torso
2. Legs
3. Arms
4. Head & Other parts

.. note::
    Each plate takes about 10 hours to print on Bambu X1C, so plan accordingly! On a regular printer, it might ~30 hours per plate, you might want to split them to get a higher success rate.

Assembly
-------------
It might look scary, but don't worry! We have a detailed assembly manual that will guide you through the process. And the good news is, the left and right side are mirror. So the process should be mostly identical! For the arm, we also reused a lot of the submodules so as soon as you get a hang of one, the rest should come easily.

- We recommand starting with the arms, as they are the easiest to assemble. It should be a good practice of your assembly skills. The instructions are here: :ref:`assembly_manual`

- And then move on to the legs, left side and the right side are mirror, so you can just follow the same instructions for both sides. The instructions are here: :ref:`assembly_manual`

- And then the head and waist of torso. The instructions are here: :ref:`assembly_manual`

.. note::
    Hopefully at this point you should be close to getting the PCBs. For the Jetson, you would want to set it up first before you put it in the torso. Refer to the section below for the jetson setup instructions.

- And then let's wire things up! The instructions are here: :ref:`electronics`. The PCB sits in the torso, along side the battery.

Jetson
-------------
Jetson is the brain of the Toddlerbot. It's a powerful little computer that can inference up to 100 TOPS and has 16GB of shared RAM and VRAM. It's basically a ARM64 computer with a Nvidia GPU running Ubuntu 22.04. The setup instructions are here: :ref:`jetson_orin`. Basically you need to flash in the new Jetpack 6.0, real time kernel, and some necessary packages.

.. note::
    Jetpack 6.0 is based on Ubuntu 22.04, the system is called Linux4Tegra (L4T).

.. note::
    Real time kernel helps with the control frequency and makes the computer respond faster.

- Now install the Jetson to the torso. And start setting up the software. The instructions are here: :ref:`jetson_orin`

.. note::
    At this point you should have a working toddlerbot! Here are the optional parts that would make your life cooler (and easier).

[Optional] Steam Deck
-------------
Simply put, steam deck is a handheld sized desktop computer. It runs Arch Linux and can run all the software stacks that you would run on a desktop. With steam deck, you are truly wireless. You can bring the toddlerbot anywhere and control it from the steam deck. The setup instructions are here: :ref:`steam_deck`

[Optional] DIY battery pack for more battery life
-------------
You can get up to doubled battery life by switching from a LiPo pack to 4x 21700 cells.

[Optional] Test Stand
-------------
For the safety of the toddy (and you), we designed a test stand that you can lift it up in the air and test without requiring it to be on the ground. The instructions are here: :ref:`test_stand`

[Optional] Teleoperation
-------------
You can for sure develop any teleoperation solution you want. We find a simple yet effective solution is to simply build another upper body (as leader) to teleoperate the full body toddlerbot. It's quite intuitive as you can just hold it's hand and instruct it like a baby. The instructions of building are here: :ref:`teleoperation_arms`. The toddlerbot API have everything already written to teleoperate the toddlerbot this way.