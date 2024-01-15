from setuptools import find_packages, setup

# Read the contents of your requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

requirements.extend(["pigar==2.1.1", "onshape-to-robot==0.3.24", "control==0.9.4"])

setup(
    name="toddleroid",
    packages=find_packages(),
    install_requires=requirements,
)
