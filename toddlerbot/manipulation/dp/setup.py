from setuptools import find_packages, setup

setup(
    name="dp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        "numpy<=1.26.4",
        "opencv-python",
        "scikit-image",
        "scikit-video",
        "zarr",
        "pygame",
        "pymunk",
        "torch==2.3",
        "torchvision",
        "diffusers",
        "gdown",
        "lz4",
        "joblib",
        "gym==0.26.2",
        "shapely",
        # Add other dependencies as needed
    ],
    entry_points={
        "console_scripts": [
            # You can define command-line scripts here if needed
        ],
    },
)
