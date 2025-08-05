# Robot Trains Robot

![ToddlerBot](docs/_static/banner.png)

**[Paper](https://arxiv.org/abs/2502.00893)** |
**[Website](https://robot-trains-robot.github.io/)** |
**[Video](https://youtu.be/A43QxHSgLyM)** | 
**[Tweet](https://x.com/HaochenShi74/status/1886599720279400732)** |

Robot-Trains-Robot (RTR) is a novel framework where a robotic arm teacher actively supports and guides a humanoid robot student. The RTR system provides protection, learning
schedule, reward, perturbation, failure detection, and automatic resets. It enables efficient long-term real-world humanoid training with minimal human intervention. 

## Setup
Our setup overall is similar to [todderbot](https://toddlerbot.github.io/). 
Refer to [this page](https://hshi74.github.io/toddlerbot/software/01_setup.html) for instructions to setup.

## Walkthrough
### Pretraining walking policy
The following command execute the training procedure presented in section 3.1 in the paper.
#### Stage 1 training
```
python -m toddlerbot.locomotion.train_mjx --tag <your tag>
```
#### Stage 2 training
```
python -m toddlerbot.locomotion.train_mjx --tag <the same tag as that above> --optimize-z --restore <yyyymmdd_hhmmss> # timestamp output from stage 1
```

### Real-world training
An examplar execution process is presented in the [launch](https://github.com/shockwaveHe/Robot-Trains-Robot/tree/rtr/launch) folder. Please make sure that the robot, computer and remote learner are under the same network.
#### Real-world adaptation for walking policy (pretrain needed)
On computer, run the script to control the arm and treadmill
```
python -m toddlerbot.policies.run_policy --policy at_leader --sim finetune --ip <your robot ip> 
```
On robot, run the finetune script
```
python toddlerbot/policies/run_policy.py --sim real --ip <your computer ip> --policy walk_finetune --robot toddlerbot_2xm  --ckpt <pretrained checkpoint>
```
On remote learner, run the remote learning script
```
python toddlerbot/policies/run_policy.py --policy walk
```
#### Real-world learning from scratch for swing-up policy (no pretrain)
On computer, run the script to control the arm
```
python toddlerbot/policies/run_policy.py --policy swing_arm_leader --ip <your robot ip> --robot toddlerbot_2xm
```
On robot, run the script to learn the swing-up policy
```
python toddlerbot/policies/run_policy.py --sim real --ip <your computer ip> --policy swing --robot toddlerbot_2xm --no-plot
```

## Citation
If you use ToddlerBot for published research, please cite:
```
@article{shi2025toddlerbot,
  title={ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation},
  author={Shi, Haochen and Wang, Weizhuo and Song, Shuran and Liu, C. Karen},
  journal={arXiv preprint arXiv:2502.00893},
  year={2025}
}
```

## License  

- The ToddlerBot codebase (including the documentation) is released under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE).

- The ToddlerBot design (Onshape document, STL files, etc.) is released under the [![License: CC BY-NC-SA](https://img.shields.io/badge/License-CC%20BY--NC--SA-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/), which allows you to use and build upon our work non-commercially.
The design of ToddlerBot is provided as-is and without warranty.
