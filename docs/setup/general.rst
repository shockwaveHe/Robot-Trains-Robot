.. _general_setup:

General
=======

Set up Miniforge
----------------

We recommend installing `Miniforge <https://github.com/conda-forge/miniforge>`__.

Run the following commands to determine your system architecture:

.. code:: bash

   uname -m

Based on your system architecture, download the appropriate Miniforge installer. For example,
for a Linux machine with ``arm64`` architecture, download ``Linux aarch64 (arm64)`` from their website. 
Do **NOT** run the install script with sudo. 
Answer ``yes`` to all the options.

Run ``source ~/.bashrc`` to activate the conda environment.

Set up the repo
---------------

Run the following commands to clone the repo:

.. code:: bash

   mkdir ~/projects
   cd ~/projects
   git clone git@github.com:hshi74/toddlerbot.git
   cd toddlerbot
   git submodule update --init --recursive


Follow the steps on `this page <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`__ to set up SSH keys for GitHub if needed.

Set up the conda environment:

.. tabs::

   .. group-tab:: Linux

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e .[linux]

   .. group-tab:: Mac OSX (arm64)

      ::

         CONDA_SUBDIR=osx-arm64 conda create -n toddlerbot python=3.10
         conda activate toddlerbot
         conda config --env --set subdir osx-arm64
         pip install -e toddlerbot/brax
         pip install -e .[macos]

   .. group-tab:: Windows

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e .[windows]

   .. group-tab:: Jetson

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e .[jetson]

   .. group-tab:: Steam Deck

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e .[steam_deck]

For **Jetson**, you need to follow the information on `this page <https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048>`__
to install ``torch`` and ``torchvision``. For reference, we downloaded 
`the wheel file for PyTorch v2.3.0 with JetPack 6.0 (L4T R36.2 / R36.3) + CUDA 12.2 <https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl>`__.

We find that the ``--content-disposition`` option is useful for downloading the file with the correct name:

.. code:: bash

   wget --content-disposition <link/to/the/wheel>

Assuming the toddlerbot conda environment is activated, install the wheels with:

.. code:: bash

   pip install <path/to/the/wheel>


Last but not least, run the following command to verify the installation of jax and torch:

.. code:: bash
   
   python tests/test_jax_torch.py --platform <linux/macos/windows/jetson/steam_deck>

Dynamixel
---------

Dynamixel motors require a low latency timer. We automated most of the process in `toddlerbot/actuation/dynamixel/dynamixel_control.py`, but you may need to manually set the latency timer.

According to the doc `here <https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/faq/#how-to-change-an-usb-latency-in-dynamixel-sdk>`__, 
set ``LATENCY_TIMER = 1`` in ``/[path_to_env]/toddlerbot/lib/python3.10/site-packages/dynamixel_sdk/port_handler.py``.

.. tabs::

   .. group-tab:: Mac OSX (arm64)

      According to the discussion `here <https://openbci.com/forum/index.php?p=/discussion/3108/driver-latency-timer-fix-for-macos-11-m1-m2>`__ and `in this blog post <https://www.mattkeeter.com/blog/2022-05-31-xmodem/#ftdi>`__, 
      we need to run a small C program each time on Mac to set the latency timer to 1.

      Run the following commands to set it up:
      ::

         brew install libftdi
         cd toddlerbot/actuation/dynamixel/latency_timer_setter_macOS
         cc -arch arm64 -I/opt/homebrew/include/libftdi1 -L/opt/homebrew/lib -lftdi1 main.c -o set_latency_timer
         ./set_latency_timer

(Optional) Set up OnShape to URDF
---------------------------------

Obtain the API key and secret key from the `OnShape developer portal <https://dev-portal.onshape.com/keys>`__.

We recommend storing your API key and secret in environment variables, and you can add something like this to your `.bashrc`:

::

   export ONSHAPE_API=https://cad.onshape.com
   export ONSHAPE_ACCESS_KEY=Your_Access_Key
   export ONSHAPE_SECRET_KEY=Your_Secret_Key


Read the `config doc <https://onshape-to-robot.readthedocs.io/en/latest/config.html>`__ first if you have any issues.

We need to install MeshLab to simplify the meshes downloaded from OnShape in the URDF files.

.. tabs::

   .. group-tab:: Linux

      ::

         sudo apt-get install meshlab


   .. group-tab:: Mac OSX (arm64)

      We recommend you install MeshLab releases older than 2020.12, such as `2020.9 <https://github.com/cnr-isti-vclab/meshlab/releases/tag/Meshlab-2020.09>`__. 
      Later releases removed the support for ``meshlabserver``.

      Add the following line to your ``~/.bashrc``:

      ::

         export PATH="/Applications/meshlab.app/Contents/MacOS:$PATH"

      Then run:

      ::

         source ~/.bashrc

      Go to ``~/anaconda3/envs/toddlerbot/lib/python3.10/site-packages/onshape_to_robot/config.py``. Change line 144 from:

      ::

         if not os.path.exists('/usr/bin/meshlabserver') != 0:

      To:

      ::

         import shutil
         if shutil.which('meshlabserver') is None:

      TODO: Automate this change in the script.

Run the following script and follow the instructions:

::

   bash scripts/onshape_to_robot.sh

(Optional) Set up the SysID Optimization Tool
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

(Optional) Set up the PID Tuner Tool
------------------------------------

For the PID tuner tool, install the following:

.. tabs::

   .. group-tab:: Linux

      ::

         sudo apt-get install libxcb-xkb1 libxkbcommon-x11-0 libxcb-cursor0

   .. group-tab:: Mac OSX (arm64)

      (TODO: Update the following command)
      ::

         brew install libxcb-xkb1 libxkbcommon-x11-0 libxcb-cursor0

(Optional) Visualize with Blender
---------------------------------

Install Blender from the `official website <https://www.blender.org/download/>`__. We use version 4.1.1.

Add Blender to your PATH:

.. tabs::

   .. group-tab:: Linux

      ::

         export PATH="$PATH:/path/to/blender"

   .. group-tab:: Mac OSX (arm64)

      ::

         export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"

Run the following command to open the visualization:

::

   blender toddlerbot/visualization/vis_mujoco.blend

To visualize a different robot, change the variable ``robot_name`` and set ``reimport = True``.

To visualize a different rollout, change the variable ``exp_folder_path`` and run the script.