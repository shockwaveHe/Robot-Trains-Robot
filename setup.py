from setuptools import find_packages, setup

# Read the contents of your requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# TODO: Get rid of requirements.txt
requirements.extend(
    [
        "pigar==2.1.1",
        "onshape-to-robot==0.3.24",
        "pytest==7.4.4",
        "pygraphviz==1.11",
        "pyglet==1.5.28",
        "cmaes==0.10.0",
        "optuna-dashboard==0.15.1",
    ]
)

setup(
    name="toddlerbot",
    packages=find_packages(),
    install_requires=requirements,
)
