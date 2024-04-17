# toddlerbot

## Installation

```
git clone git@github.com:hshi74/toddlerbot.git
cd toddlerbot
git submodule update --init --recursive
```

Recommended IDE:
VSCode

Recommended VSCode Extenstions:
1. Black Formatter
1. isort
1. Python, Pylance, Python Debugger
1. Serial Monitor
1. URDF
1. vscode-stl-viewer
1. XML Tools

### Linux Systems
```
conda create --name toddlerbot python=3.8
conda activate toddlerbot
pip install -e .
```
Install pygraphviz according to [these instructions](https://pygraphviz.github.io/documentation/stable/install.html).

#### Setting up Optuna
Linux:
```
sudo apt install libpq-dev postgresql
sudo systemctl start postgresql
sudo -u postgres psql
```

MacOS:
```
brew install postgresql
brew services start postgresql
psql postgres
```

Enter the following commands in the PostgreSQL prompt:
```
CREATE DATABASE optuna_db;
CREATE USER optuna_user WITH ENCRYPTED PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;

```
Exit the PostgreSQL prompt.


#### Install legged_gym
Follow [these instructions](https://github.com/leggedrobotics/legged_gym).

It's OK to install IssacGym Preview Release 4.

Change line 135 in `isaacgym/python/isaacgym/torch_utils.py` from 
```
def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
```
to
```
def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
```

Run
```
pip install tensorboard
```

Change `~/anaconda3/envs/toddlerbot/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py` from
```
import tensorboard
from setuptools import distutils
LooseVersion = distutils.version.LooseVersion

# ...

del distutils

# ...
```
to
```
import tensorboard
# from setuptools import distutils
# LooseVersion = distutils.version.LooseVersion
from distutils.version import LooseVersion

# ...

# del distutils

# ... 
```

According to [this issue](https://github.com/tensorflow/tensorboard/issues/6808), run
```
pip install protobuf==4.25
```

### MacOS (Apple Sillicon M1/M2)
```
CONDA_SUBDIR=osx-arm64 conda create -n toddlerbot python=3.8
conda activate toddlerbot
conda config --env --set subdir osx-arm64

brew install graphviz
export C_INCLUDE_PATH=/opt/homebrew/opt/graphviz/include
export LIBRARY_PATH=/opt/homebrew/opt/graphviz/lib
pip install pygraphviz
pip install -e .
```

If there is yourdfpy import error, run this:
```
pip install lxml==4.9.4
```

Run MuJoCo-related scripts with `mjpython` instead of `python`.
Add `"python": "path/to/mjpython"` in `launch.json` for the VSCode debugger.

For matplotlib, make sure you're using a non-interactive backend in `toddlerbot/utils/vis_utils.py`:
```
plt.switch_backend("Agg")
```

## OnShape to URDF
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

### MacOS (Apple Sillicon M1/M2)
We recommend you to install MeshLab releases older than 2020.12 such as [2020.9](https://github.com/cnr-isti-vclab/meshlab/releases/tag/Meshlab-2020.09) because 2020.12 and later releases removed the support of `meshlabserver`.

Add the following line to your `~/.bashrc`.
```
export PATH="/Applications/meshlab.app/Contents/MacOS:$PATH"
```

Then call
```
source ~/.bashrc
```

Go to `~/anaconda3/envs/toddlerbot/lib/python3.8/site-packages/onshape_to_robot/config.py`
Change line 144 from
```
if not os.path.exists('/usr/bin/meshlabserver') != 0:
```
to
```
import shutil
if shutil.which('meshlabserver') is None:
```

## Working with Arduino and ESC boards

### Install Arduino in VSCode on MacOS
```
brew install arduino-cli
```
Install the Arduino VSCode extension.

Open VSCode settings:
1. Set Arduino Command Path: arduino-cli
1. Set Arduino Path: /opt/homebrew/bin
1. Set Use Arduino Cli: yes

Follow this [tutorial](https://learn.adafruit.com/adafruit-feather-m4-can-express/arduino-ide-setup).

```
screen /dev/tty.usbxxxxx 115200
```

### Flush code to the ESC board
```
brew install open-ocd
brew install gcc-arm-embedded
```
Install these VSCode extensions:
1. C/C++
1. C/C++ Extension Pack
1. Cortex-Debug
1. Makefile Tools

Add the following configuration to launch.json:
```
{
    "cwd": "${workspaceRoot}",
    "executable": "toddlerbot/actuation/sunny_sky/knee_esc/Release/motorcontrol.elf",
    "name": "Flush Knee ESC",
    "request": "launch",
    "type": "cortex-debug",
    "servertype": "openocd",
    "configFiles": [
        "toddlerbot/actuation/sunny_sky/knee_esc/daplink.cfg"
    ],
    "searchDir": [],
    "runToEntryPoint": "main",
    "showDevDebugOutput": "none"
}
```
Connect the ESC board to the computer.
Run Flush Knee ESC from the "Run and Debug" panel.

## Actuation
- MightyZap: Update the firmware if you cannot search the motor.

## TODO

### Export the environment
Will remove this later.
```
pigar generate
```

### ESC interface redesign
1. Separate TX, RX
2. Add a new can bus connector
3. Use connector that is easier to work with
4. Smaller footprint

### No belt

### Good coding practices
1. Use dataclass and argparse if possible
1. Write Google style docstring
1. Write type hint
1. Assert, and raise errors if possible
1. Use pure functions if possible
1. Put the magic numbers together in one place
1. Write inline document if possible
1. Use shell scripts
1. Consider writing unit tests
