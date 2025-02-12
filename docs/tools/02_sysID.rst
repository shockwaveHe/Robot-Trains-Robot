.. _sysID:

System Identification (WIP)
============================

This section provides a comprehensive guide to perform system identification (SysID) for Dynamixel motors.

Hardware Setup
---------------------------------------------
TODO


SysID Data Collection
---------------------------------------------
TODO

Set up Optuna Dashboard
---------------------------------------------

For the SysID Optimization tool, you need to install the following packages:

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

Run the Optuna dashboard:

::

   optuna-dashboard postgresql://optuna_user:password@localhost/optuna_db