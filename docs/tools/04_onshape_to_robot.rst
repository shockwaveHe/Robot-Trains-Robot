.. _onshape_to_robot:

Onshape to Robot Descriptions
==============================

This is a command-line tool that allows you to convert Onshape robots to URDF and MJCF files for simulation and deployment.
The tool is built on top of the `onshape_to_robot <https://github.com/Rhoban/onshape-to-robot>`__ package.

Obtain the API key and secret key from the `Onshape developer portal <https://dev-portal.onshape.com/keys>`__.

We recommend storing your API key and secret in environment variables, and you can add something like this to your `.bashrc`:

::

   export ONSHAPE_API=https://cad.onshape.com
   export ONSHAPE_ACCESS_KEY=Your_Access_Key
   export ONSHAPE_SECRET_KEY=Your_Secret_Key


Read the `config doc <https://onshape-to-robot.readthedocs.io/en/latest/config.html>`__ first if you have any issues.

We need to install MeshLab to simplify the meshes downloaded from Onshape in the URDF files.

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


Run the following script and follow the instructions:

::

   bash scripts/onshape_to_robot.sh --robot <robot_name>


Please carefully read the terminal messages printed from this script.