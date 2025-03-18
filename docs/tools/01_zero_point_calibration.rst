.. _zero_point_calibration:

Zero-point Calibration
================================


Initial Calibration
-------------------
In the assembly manual and assembly video, we have already shown how to calibrate the zero point of ToddlerBot.
But there are some details that we would like to further elaborate on in this section. The calibration process 
becomes vert quick once you're familiar with it.

#. To start with, you need to 3D-print the calibration devices. You can find all the sliced plates for calibration devices in the 
   `MakerWorld <https://makerworld.com/en/models/1068768>`_. 
   and the CAD files in the `Onshape document <https://cad.onshape.com/documents/1370cb70ae00945ee5a1ab36>`_.
   
#. When inserting the calibration devices, you may encounter friction due to the tight fit. Once secured, they should click into place, locking the joints. 
   If the joints still move, check for obstructing cables or incorrect insertion.
   Routing cables around the calibration devices can be tricky when doing it for the first time, but trust us, it's entirely doable.
   For the knee joints, the lower limits align with the zero point, allowing the joint's natural stopper to serve as the calibration reference.
   For the parallel jaw gripper, the zero point is the open position as detailed in the assembly manual.
   
#. When you are ready to calibrate the zero point, you can run the following script:
   ::

      python toddlerbot/tools/calibrate_zero.py --robot <robot_name> --parts <part_names_separated_by_space>

   If no parts are specified, the script will calibrate all the joints. You can find more information in the ``calibrate_zero.py`` script.

#. Next, you can optionally inspect the joint angles in ``config_motors.json`` to double check. Then run the following script to 
   propagate the zero points to the actual configuration file:

   ::

      python toddlerbot/descriptions/add_configs.py --robot <robot_name>


#. Lastly, we recommend running the standing policy to check if the robot is standing upright on the ground. 
   If not, you can further fine-tune the zero point as described in the next section.

   ::

      python toddlerbot/policies/run_policy.py --policy stand --robot <robot_name> --sim real


More Fine-tuning
----------------

If you find the robot is slightly leaning forward or backward after zero-point calibration with the 3D-printed devices, 
which could totally happen due to the backlash in the joints,
you can further fine-tune the zero point by running this script:
::

    python toddlerbot/policies/run_policy.py --policy calibrate --robot <robot_name> --sim real

This script basically runs a PID control with the torso pitch feedback from IMU. 
You can find more information in the ``calibrate.py`` script.

When you run this script, the robot will stand up and try to maintain an upright position. When 
you think the robot is standing upright, you can press ``Ctrl+C`` to stop the script. IMPORTANT:
Please make sure to hold the robot when you stop the script, as the motors will be disabled and 
the robot will fall over.
The script will then save the zero point to the configuration file.