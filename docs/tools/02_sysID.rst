.. _sysID:

System Identification
============================

This section provides a comprehensive guide to perform system identification (SysID) for Dynamixel motors.

Hardware Setup
---------------------------------------------
You will need to purchase the sysID section in the :ref:`bill_of_materials`.

For 3D-printed parts, you can find all the sliced plates in the `MakerWorld <https://makerworld.com/en/models/1068768>`_ 
and the CAD files in the `Onshape document <https://cad.onshape.com/documents/1fb5d9a88ac086a053c4340b>`_.

We use 21700 cells as loads to adjust the motor load weight.

Using the accompanying image as a guide, set up the SysID hardware so that the load arm has sufficient clearance to rotate from -90° to 90°. 
Note that 0° corresponds to the horizontal position, parallel to the ground. You can use the Dynamixel Wizard to verify the setup and 
configure the motor's ID to 0 with a baud rate of 2M.

.. image:: ../_static/sysID_setup.png
   :alt: SysID Setup
   :align: center
   :width: 100%


Set up Optuna Dashboard
---------------------------------------------

To visualize the sysID optimization process, you need to install the following packages:

.. tabs::

   .. group-tab:: Linux

      ::

         sudo apt install libpq-dev postgresql
         sudo systemctl start postgresql

   .. group-tab:: Mac OSX (arm64)

      ::

         brew install postgresql
         brew services start postgresql

Run PostgreSQL:

.. tabs::

   .. group-tab:: Linux

      ::

         sudo -u postgres psql

   .. group-tab:: Mac OSX (arm64)

      ::

         psql postgres

Enter the following commands in the PostgreSQL prompt:

::

   CREATE DATABASE optuna_db;
   CREATE USER optuna_user WITH ENCRYPTED PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;

Exit the PostgreSQL prompt.


SysID Data Collection
---------------------------------------------

Run the following command with the appropriate robot name to collect data.
For motors in the 2XC430, 2XL430, XC330, XC430, and XM430 series, use the corresponding robot names: 
sysID_XC430, sysID_XL430, sysID_XC330, sysID_XC430, and sysID_XM430, respectively.
If your motor is not listed, refer to our `Onshape document <https://cad.onshape.com/documents/1fb5d9a88ac086a053c4340b>`_ to set it up in Onshape. 
Then, use the :ref:`onshape_to_robot` tool to generate the XML file for the motor. For reference, check ``toddlerbot/descriptions/sysID_XC430``
for the expecetd outcome.

::

   python toddlerbot/policies/run_policy.py --policy sysID_fixed --sim real --robot <robot_name>


Run the Optimization
---------------------------------------------

Finally, run the following command to start the optimization process. The ``--time-str`` argument should be set 
to the timestamp of the directory generated when running the SysID policy in the previous section. 
For example, ``20250301_155535``.

::

   python toddlerbot/tools/run_sysID.py --robot <robot_name> --time-str <time_str>


Run the Optuna dashboard in another terminal to visualize the optimization process:

::

   optuna-dashboard postgresql://optuna_user:password@localhost/optuna_db