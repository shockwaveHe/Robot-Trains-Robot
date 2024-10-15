import platform

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
        "pyglet==1.5.28",
        "cmaes==0.10.0",
        "optuna-dashboard==0.15.1",
        "psycopg2-binary==2.9.9",
        "sphinx-autobuild",
    ]
)

# Determine the operating system and append the appropriate JAX package
if platform.system() == "Linux":  # Check if the system is Linux
    requirements.append("jax[cuda12]")
else:
    requirements.append("jax")  # Use the CPU version for macOS and Windows


setup(
    name="toddlerbot",
    packages=find_packages(),
    install_requires=requirements,
)
