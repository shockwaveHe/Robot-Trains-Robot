.. _setup:

Setup
============

This is the general software setup process with slight variations on different platforms. 
For additional setup of specific platforms such as :ref:`jetson_orin`, ROG Ally X, and Steam Deck, 
please refer to the corresponding sections.

Set up the Repo
-----------------

Run the following commands to clone the repo:

.. code:: bash

   mkdir ~/projects
   cd ~/projects
   git clone git@github.com:hshi74/toddlerbot.git
   cd toddlerbot
   git submodule update --init --recursive

Follow the steps on `this page <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`_ 
to set up SSH keys for GitHub if needed.


Install Miniforge
------------------

If ``conda`` is not installed yet, we recommend installing `Miniforge <https://github.com/conda-forge/miniforge>`_.

Run the following commands to determine your system architecture:

.. code:: bash

   uname -m

Based on your system architecture, download the appropriate Miniforge installer. For example,
for a Linux machine with ``arm64`` architecture, download ``Linux aarch64 (arm64)`` from their website.
Do **NOT** run the install script with sudo.
Answer ``yes`` to all the options.

Run ``source ~/.bashrc`` to activate the conda environment.


Set up Conda Environment
-----------------------------

.. tabs::

   .. group-tab:: Linux

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e ".[linux]"

   .. group-tab:: Mac OSX (arm64)

      ::

         CONDA_SUBDIR=osx-arm64 conda create -n toddlerbot python=3.10
         conda activate toddlerbot
         conda config --env --set subdir osx-arm64
         pip install -e toddlerbot/brax
         pip install -e ".[macos]"

   .. group-tab:: Mac OSX (x86_64)

      ::

         CONDA_SUBDIR=osx-64 conda create -n toddlerbot python=3.10
         conda activate toddlerbot
         conda config --env --set subdir osx-64
         pip install -e toddlerbot/brax
         pip install -e ".[macos]"

   .. group-tab:: Windows

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e ".[windows]"

   .. group-tab:: Jetson

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e ".[jetson]"

   .. group-tab:: ROG Ally X

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e "."

   .. group-tab:: Steam Deck

      ::

         conda create --name toddlerbot python=3.10
         conda activate toddlerbot
         pip install -e toddlerbot/brax
         pip install -e "."
