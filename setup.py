from setuptools import find_packages, setup

# Read the contents of your requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

requirements.extend(["pigar", "onshape-to-robot"])

setup(
    name="toddleroid",
    packages=find_packages(),
    install_requires=requirements,
)
