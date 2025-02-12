# ToddlerBot

![ToddlerBot](docs/_static/banner.png)

**[ArXiv](https://arxiv.org/abs/YOUR_PAPER_ID)** |
**[Website](https://toddlerbot.github.io/)** |
**[Video](https://youtu.be/A43QxHSgLyM)** | 
**[Tweet](https://twitter.com/YOUR_TWEET)** |
**[Documentation](https://github.com/toddlerbot/toddlerbot.github.io/wiki)**

ToddlerBot is a low-cost, open-source humanoid robot platform designed for scalable policy learning and research in robotics and AI.

This codebase includes low-level control, RL training, DP training, real-world deployment and basically EVERYTHING you need to run ToddlerBot in the real world!

Built entirely in Python, it is **fully pip-installable** (python >= 3.10) for seamless setup and usage!


## Setup
Refer to [this page]() for instructions to setup.


## Walkthrough

- Checkout `examples` for some scripts to start with. Many of them run on a real-world instance of ToddlerBot.

- The `motion` folder contains carefully crafted keyframe animations designed for ToddlerBot. For example, you can run

    ```
    python toddlerbot/policies/run_policy.py --policy replay --run-name push_up --vis view
    ```

    to see the push up motion in MuJoCo. You're very welcome to contribute your keyframe animation to our repository by
    submitting a pull request!

- The `scripts` folder contains some utility bash scripts.

- The `tests` folder have some tests that you can run with 

    ```
    pytest tests/
    ``` 

    to verify our installation.

- The `toddlerbot` folder contains all the source code. You can find a detailed API documentation [here]().


## Submitting an Issue
For easier maintenance, we will ONLY monitor GitHub Issues and likely ignore questions from other sources.
We welcome issues related to anything we’ve open-sourced, not just the codebase.

Before submitting an issue, please ensure you have:
- Read the [documentation](), including the [Tips and Tricks]() section.
- Checked the comments in the scripts.
- Carefully reviewed the assembly manual.
- Watched the assembly videos.

If we determine that your issue arises from not following these resources, we are unlikely to respond. 
However, if you have found a bug, need support for any open-sourced component, or want to submit a feature request, 
feel free to open an issue.

We truly appreciate your feedback and will do our best to address it!

## Community

See [our website]() for links to join the Discord or WeChat community!

## Contributing  

We welcome contributions from the community! To contribute, just follow the standard practice:
1. Fork the repo  
2. Create a branch (`feature-xyz`)  
3. Commit & push  
4. Submit a Pull Request (PR)  

## Citation
If you use ToddlerBot for published research, please cite:
```

```

## License  

- The ToddlerBot codebase (including the documentation) is released under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE).

- The ToddlerBot design (Onshape document, STL files, etc.) is released under the [![License: CC BY-NC-SA](https://img.shields.io/badge/License-CC%20BY--NC--SA-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/), which allows you to use and build upon our work non-commercially.
The design of ToddlerBot is provided as-is and without warranty.
