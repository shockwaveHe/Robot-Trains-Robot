# RetargetLib: A library for motion retargeting

This repository contains a collection of tools for retargeting motion between humans and humanoids.

## Installation

```bash
git clone https://github.com/Stanford-TML/retarget_lib.git
cd retarget_lib
pip install [-e] .
```

## Usage

Check the `examples` folder for examples on how to use this library.

**Retarget a human pose from a Vicon motion capture system to a G1 robot in real time**

```
python examples/mink_retarget_example.py --host [Vicon server IP] --xml_file [path to]/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml
```

In MacOS use `mjpython` instead of `python`. This example uses the `ik_configs/vicon2g1_29dof_rev_1_0.json` configuration by default.
