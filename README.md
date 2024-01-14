# toddleroid

### Install the environment
```
conda create --name toddleroid python=3.8
pip install -e .
```

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

### Best practices
1. Write dataclass, yaml, and use argparse
1. Write docstring, assert, and raise errors
1. Use shell scripts
1. Consider writing unit tests
1. Use pure functions if possible
