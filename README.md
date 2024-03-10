# toddlerbot

### Install the environment
```
conda create --name toddlerbot python=3.8
conda activate toddlerbot
pip install -e .
```

Install pygraphviz according to [these instructions](https://pygraphviz.github.io/documentation/stable/install.html).

Apple Sillicon M1/M2 specific commands:
```
CONDA_SUBDIR=osx-arm64 conda create -n toddlerbot python=3.8
conda activate toddlerbot
conda config --env --set subdir osx-arm64

brew install graphviz
export C_INCLUDE_PATH=/opt/homebrew/opt/graphviz/include
export LIBRARY_PATH=/opt/homebrew/opt/graphviz/lib
pip install pygraphviz
pip install -e .

# If there is yourdfpy import error, run this.
pip install lxml==4.9.4
```

Run MuJoCo-related scripts with `mjpython` instead of `python`.
Add `"python": "path/to/mjpython"` in `launch.json` for the VSCode debugger.

```
plt.switch_backend("Agg")
```

### Export the environment
```
pigar generate
```

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


### Get the robot URDF fron OnShape
To go any further, you will need to obtain API key and secret from the [OnShape developer portal](https://dev-portal.onshape.com/keys).

We recommend you to store your API key and secret in environment variables, you can add something like this in your .bashrc:
```
// Obtained at https://dev-portal.onshape.com/keys
export ONSHAPE_API=https://cad.onshape.com
export ONSHAPE_ACCESS_KEY=Your_Access_Key
export ONSHAPE_SECRET_KEY=Your_Secret_Key
```
This is important:
```
sudo apt-get install meshlab
```
[Config doc](https://onshape-to-robot.readthedocs.io/en/latest/config.html)

Use VScode Extension XML Tools to format the XML files.
```
bash scripts/onshape_to_robot.sh
```
Manually replace the body_link_collision.stl.

### Actuation
- MightyZap: Update the firmware if you cannot search the motor.

### Best practices
1. Use dataclass and argparse if possible
1. Write Google style docstring
1. Write type hint
1. Assert, and raise errors if possible
1. Use pure functions if possible
1. Put the magic numbers together in one place
1. Write inline document if possible
1. Use shell scripts
1. Consider writing unit tests
