import argparse
import os
import time
from dataclasses import asdict

from toddlerbot.algorithms.ppo.on_policy_runner import OnPolicyRunner
from toddlerbot.envs.humanoid_config import HumanoidCfg
from toddlerbot.envs.humanoid_env import HumanoidEnv
from toddlerbot.envs.ppo_config import PPOCfg
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import get_load_path
from toddlerbot.utils.misc_utils import set_seed


def make_runner(
    env: HumanoidEnv, env_cfg: HumanoidCfg, train_cfg: PPOCfg, exp_folder_path: str
):
    all_cfg = {**asdict(env_cfg), **asdict(train_cfg)}
    runner = OnPolicyRunner(env, all_cfg, exp_folder_path)

    # save resume path before creating a new log_dir
    resume = train_cfg.runner.resume
    if resume:
        # load previously trained model
        resume_path = get_load_path(
            os.path.basename(exp_folder_path),
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        print(f"Loading model from: {resume_path}")
        runner.load(resume_path, load_optimizer=False)

    return runner


def train(robot: Robot, sim: MuJoCoSim, env_name: str):
    exp_name = f"{env_name}_rl_{robot.name}_{sim.name}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"

    os.makedirs(exp_folder_path, exist_ok=True)

    if env_name == "walk":
        from toddlerbot.envs.toddlerbot_config import toddlerbot_cfg, toddlerbot_ppo_cfg
        from toddlerbot.envs.toddlerbot_env import ToddlerbotEnv

        env = ToddlerbotEnv(sim, robot, toddlerbot_cfg)
        runner = make_runner(env, toddlerbot_cfg, toddlerbot_ppo_cfg, exp_folder_path)

        try:
            runner.learn(
                num_learning_iterations=toddlerbot_ppo_cfg.runner.max_iterations,
                init_at_random_ep_len=True,
            )
        except KeyboardInterrupt:
            print("KeyboardInterrupt recieved. Closing...")

        finally:
            if hasattr(sim, "save_recording"):
                sim.save_recording(exp_folder_path)  # type: ignore

            sim.close()

    else:
        raise ValueError("Unknown env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="walk",
        help="The name of the env.",
    )
    parser.add_argument(
        "--vis",
        type=str,
        default="render",
        help="The name of the env.",
    )
    args = parser.parse_args()

    set_seed(0)

    robot = Robot(args.robot)

    if args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot, vis_type=args.vis)
    else:
        raise ValueError("Unknown simulator")

    train(robot, sim, args.env)
