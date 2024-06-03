from setuptools import setup

setup(
    name="dynamixel_sdk",
    version="3.8.1",
    packages=["dynamixel_sdk"],
    package_dir={"": "src"},
    license="Apache 2.0",
    description="DYNAMIXEL SDK python package",
    long_description=open("README.txt").read(),
    url="https://github.com/ROBOTIS-GIT/DynamixelSDK",
    author="Leon Jung",
    author_email="rwjung@robotis.com",
    install_requires=["pyserial"],
)
