# toddleroid

### Install the environment
```
conda create --name toddleroid python=3.8
conda activate toddleroid
pip install -e .
```

Install pygraphviz according to [these instructions](https://pygraphviz.github.io/documentation/stable/install.html).

### Export the environment
```
pigar generate
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
```
sudo apt-get install meshlab
```
[Config doc](https://onshape-to-robot.readthedocs.io/en/latest/config.html)

Use VScode Extension XML Tools to format the XML files.

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
