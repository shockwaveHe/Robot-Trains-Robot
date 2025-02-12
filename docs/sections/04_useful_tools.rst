.. _useful_tools:

Useful Tools
========================

.. toctree::
   :maxdepth: 1
   :caption: Table of Contents
   :hidden:

   ../tools/01_zero_point_calibration
   ../tools/02_sysID
   ../tools/03_keyframe_animation
   ../tools/04_onshape_to_robot


Zero-point Calibration
------------------------
Since Dynamixel motors lack an inherent zero point, a reliable method is needed to recalibrate after assembly, which
is frequent during repairs or design iterations. We design calibration devices in CAD that quickly align the robot 
to its zero point, defined as standing with both arms besides the body. The process takes less than a minute. 
We show our design and calibration process in the :ref:`zero_point_calibration` section.

System Identification
--------------------------------
Inspired by `Haarnoja et al. <https://www.science.org/doi/10.1126/scirobotics.adi8022>`_, we collect sysID data
by commanding the motors to track a chirp signal and use the resulting position tracking data to fit an actuation model
as described in `Grandia et al. <https://la.disneyresearch.com/publication/design-and-control-of-a-bipedal-robotic-character/>`_ 
We provide the detailed procedure in the :ref:`sysID` section.

Keyframe Animation
--------------------------------
Keyframe animation is a cornerstone of character animation, but it provides only kinematic data, with no guarantee of dynamic
feasibility. To address this, we developed software integrating MuJoCo with a GUI, enabling real-time tuning
and validation of keyframes and motion trajectories generated through linear interpolation with user-defined timings.
We introduce the software and its features in the :ref:`keyframe_animation` section.

Onshape to Robot Descriptions
--------------------------------
We use Onshape to design ToddlerBot and its components. To transfer the design to the robot, we need to convert the
Onshape assembly to URDF and MJCF files. We provide a detailed guide in the :ref:`onshape_to_robot` section.