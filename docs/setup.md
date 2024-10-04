# Setup

## Set up Conda

1. Install Miniforge: Download `Linux aarch64 (arm64)` from [their website](https://github.com/conda-forge/miniforge). Do **NOT** run the install script with sudo. Answer `yes` to all the options.

## Set up the repo

Run the following commands to clone the repo:
```
git clone git@github.com:hshi74/toddlerbot.git
cd toddlerbot
git submodule update --init --recursive
```
Set up the conda environment:
```
conda create --name toddlerbot python=3.10
conda activate toddlerbot
pip install -e .
```
MacOS (arm64)
```
CONDA_SUBDIR=osx-arm64 conda create -n toddlerbot python=3.10
conda activate toddlerbot
conda config --env --set subdir osx-arm64
pip install -e .
```

## Set up MuJoCo

MacOS (arm64):
If there is yourdfpy import error, run this:
```
pip install lxml==4.9.4
```

Run MuJoCo-related scripts with `mjpython` instead of `python`.
Add `"python": "path/to/mjpython"` in `launch.json` for the VSCode debugger.


### Dynamixel

According to the doc [here](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/faq/#how-to-change-an-usb-latency-in-dynamixel-sdk),
Set `LATENCY_TIMER = 1` in `/path_to_env/toddlerbot/lib/python3.10/site-packages/dynamixel_sdk/port_handler.py`.

MacOS users:

According to the discussion [here](https://openbci.com/forum/index.php?p=/discussion/3108/driver-latency-timer-fix-for-macos-11-m1-m2) and [this blog post](https://www.mattkeeter.com/blog/2022-05-31-xmodem/#ftdi), we run a small C program each time to set the latency timer to 1.

```
brew install libftdi
cd toddlerbot/actuation/dynamixel/latency_timer_setter_macOS
cc -arch arm64 -I/opt/homebrew/include/libftdi1 -L/opt/homebrew/lib -lftdi1 main.c -o set_latency_timer
./set_latency_timer
```

## (Optional) Set up OnShape to URDF
Obtain the API key and secret key from the [OnShape developer portal](https://dev-portal.onshape.com/keys).

We recommend you to store your API key and secret in environment variables, you can add something like this in your .bashrc:
```
// Obtained at https://dev-portal.onshape.com/keys
export ONSHAPE_API=https://cad.onshape.com
export ONSHAPE_ACCESS_KEY=Your_Access_Key
export ONSHAPE_SECRET_KEY=Your_Secret_Key
```

Note: The headless mode won't work for mesh simplification with meshlabserver currently.

### Linux Systems
```
sudo apt-get install meshlab
```
[Config doc](https://onshape-to-robot.readthedocs.io/en/latest/config.html)

Run the following script and follow the instructions:
```
bash scripts/onshape_to_robot.sh
```

### MacOS (arm64)
We recommend you to install MeshLab releases older than 2020.12 such as [2020.9](https://github.com/cnr-isti-vclab/meshlab/releases/tag/Meshlab-2020.09) because 2020.12 and later releases removed the support of `meshlabserver`.

Add the following line to your `~/.bashrc`.
```
export PATH="/Applications/meshlab.app/Contents/MacOS:$PATH"
```

Then call
```
source ~/.bashrc
```

Go to `~/anaconda3/envs/toddlerbot/lib/python3.10/site-packages/onshape_to_robot/config.py`
Change line 144 from
```
if not os.path.exists('/usr/bin/meshlabserver') != 0:
```
to
```
import shutil
if shutil.which('meshlabserver') is None:
```

## (Optional) Set up the SysID Optimization Tool
For the SysID Optimization tool, you need to install these:
```
sudo apt install libpq-dev postgresql
sudo systemctl start postgresql
```
MacOS:
```
brew install postgresql
brew services start postgresql
```
Run PostgreSQL:
```
sudo -u postgres psql
```
MacOS:
```
psql postgres
```
Enter the following commands in the PostgreSQL prompt:
```
CREATE DATABASE optuna_db;
CREATE USER optuna_user WITH ENCRYPTED PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;

```
Exit the PostgreSQL prompt.

Run the Optuna dashboard:
```
optuna-dashboard postgresql://optuna_user:password@localhost/optuna_db
```

## (Optional) Set up the PID Tuner Tool
For the PID tuner tool, you need to install these:
```
sudo apt-get install libxcb-xkb1 libxkbcommon-x11-0 libxcb-cursor0
```



## (Optional) Visualize with Blender

1. Install Blender from the [official website](https://www.blender.org/download/). I use the version 4.1.1. 
1. [Optional] Install 
1. Add Blender to your PATH:

        # MacOS
        export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"
        # Linux
        export PATH="$PATH:/path/to/blender"

1. Run the following command to open the visualization:

        blender toddlerbot/visualization/vis_mujoco.blend

1. To visualize a different robot, change the variable `robot_name` and set `reimport = True`.
1. To visualize a different rollout, change the variable `exp_folder_path` and run the script.
