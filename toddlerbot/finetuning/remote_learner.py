import os
import gin
import ipdb
import time
import torch
import argparse
import traceback
import numpy as np
from toddlerbot.sim.robot import Robot
from toddlerbot.finetuning.server_client import RemoteServer
from toddlerbot.policies.mjx_finetune import MJXFinetunePolicy
from toddlerbot.finetuning.finetune_config import get_finetune_config
from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.policies import get_policy_class, dynamic_import_policies

torch.set_float32_matmul_precision("high")

# Call this to import all policies dynamically
dynamic_import_policies("toddlerbot.policies")

NUM_ACTIONS = 12

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="walk")
    args = parser.parse_args()
    robot = Robot("toddlerbot_2xm")
    init_motor_pos = np.zeros(len(robot.motor_ordering), dtype=np.float32)
    current_task = args.task
    env_cfg = get_env_config(current_task)
    finetune_cfg = get_finetune_config(current_task)
    if current_task == "walk":
        current_task += "_finetune"

    # ckpt_folders = ["20250414_192943_latent_torch_film_4e-5_gamma_0.97_ldr_nm"]
    ckpt_folders = ["20250422_030939_latent_torch_film_4e-5_ar_0.3_ldr"]
    # ckpt_folders = ["20250423_074441_torch_ldr_baseline"]
    # ckpt_folders = [
    #     "20250423_074441_torch_ldr_baseline",
    #     "20250424_202645_latent_torch_z_film_5e-5_residual",
    # ]

    PolicyClass = get_policy_class(current_task)
    policy: MJXFinetunePolicy = PolicyClass(
        current_task,
        robot,
        init_motor_pos,
        ckpts=ckpt_folders,
        exp_folder="tests/logging",
        env_cfg=env_cfg,
        finetune_cfg=finetune_cfg,
        is_real=False,
    )
    print(policy.value_net, policy.policy_net)
    policy.replay_buffer.keep_data_after_reset = True
    # policy.replay_buffer.load_compressed("")
    # value_net_path = "tests/value_net_new.pth"
    # if current_task == "walk_finetune":
    #     recalculate_reward = False
    #     if os.path.exists(value_net_path) and not recalculate_reward:
    #         print("existing value net found")
    #         policy.value_net.load_state_dict(torch.load(value_net_path))

    #     elif len(policy.replay_buffer) > 0:
    #         policy.recalculate_reward()
    #         policy.offline_abppo_learner.fit_q_v(policy.replay_buffer)
    #         policy.replay_buffer.reset()
    #         with open(value_net_path, "wb") as f:
    #             torch.save(policy.value_net.state_dict(), f)
    #     print("Replay buffer size:", len(policy.replay_buffer))
    server = RemoteServer(host="172.24.68.176", port=5007, policy=policy)
    server.start_receiving_data()

    while not server.exp_folder:
        time.sleep(0.1)
    if not os.path.exists(server.exp_folder):
        os.makedirs(server.exp_folder)
    policy.exp_folder = server.exp_folder
    policy.logger.set_exp_folder(server.exp_folder)
    with open(os.path.join(server.exp_folder, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())
    # policy.logger.plot_updates()

    try:
        while server.is_running and server.client_thread.is_alive():
            if policy.learning_stage == "offline":
                if (
                    len(policy.replay_buffer) > finetune_cfg.offline_initial_steps
                    and (len(policy.replay_buffer) + 1) % finetune_cfg.update_interval
                    == 0
                ):
                    for _ in range(policy.finetune_cfg.abppo_update_steps):
                        policy.offline_abppo_learner.update(policy.replay_buffer)
                    policy.policy_net.load_state_dict(
                        policy.abppo._policy_net.state_dict()
                    )
                    server.push_policy_parameters(
                        policy.abppo._policy_net.state_dict()  # TODO: No latent_z in offline stage
                    )  # Push latest parameters to agent A.
                    print("Replay buffer size:", len(policy.replay_buffer))
                if len(policy.replay_buffer) >= finetune_cfg.offline_total_steps:
                    policy.switch_learning_stage()

            elif (
                policy.learning_stage == "online"
                and len(policy.replay_buffer) >= finetune_cfg.online.batch_size
            ):
                policy.online_ppo_learner.update(
                    policy.replay_buffer,
                    policy.total_steps - policy.finetune_cfg.offline_total_steps,
                )
                policy.policy_net.load_state_dict(
                    policy.online_ppo_learner._policy_net.state_dict()
                )
                policy.logger.plot_updates()
                # ans = input('push updates?')
                server.push_policy_parameters(
                    {"latent_z": policy.online_ppo_learner.latent_z}
                    if finetune_cfg.optimize_z
                    else policy.online_ppo_learner._policy_net.state_dict()
                )
                policy.logger.plot_rewards()
                print("Replay buffer size:", len(policy.replay_buffer))
            time.sleep(0.01)

    except Exception as e:
        ipdb.post_mortem()
        traceback.print_exc()
    finally:
        print("Server stopped running.")
        policy.logger.plot_rewards()
        policy.logger.plot_updates()
        policy.logger.close()
        save_networks = input("Save networks? y/n:")
        while save_networks not in ["y", "n"]:
            save_networks = input("Save networks? y/n:")
        if save_networks == "y":
            policy.save_networks()
        save_buffer = input("Save replay buffer? y/n:")
        if save_buffer == "y":
            policy.replay_buffer.save_compressed(policy.exp_folder)
