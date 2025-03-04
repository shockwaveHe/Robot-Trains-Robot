import os
import sys
import time

import torch
import numpy as np
from toddlerbot.sim.robot import Robot
from toddlerbot.finetuning.server_client import RemoteServer
from toddlerbot.policies.walk_finetune import WalkFinetunePolicy
from toddlerbot.finetuning.finetune_config import get_finetune_config
from toddlerbot.locomotion.mjx_config import get_env_config
torch.set_float32_matmul_precision('high')


NUM_ACTIONS = 12

if __name__ == "__main__":
    robot = Robot("toddlerbot")
    init_motor_pos = np.zeros(len(robot.motor_ordering), dtype=np.float32)
    finetune_cfg = get_finetune_config("walk")
    finetune_cfg.update_mode = "local"
    env_cfg = get_env_config("walk")
    # ckpt_folder = "results/stored/toddlerbot_walk_finetune_real_world_20250211_101354"
    ckpt_folders = [
        # "results/toddlerbot_walk_finetune_real_world_20250224_222209"
    ]

    # policy.abppo_offline_learner.update(policy.replay_buffer)
    # policy.logger.plot_updates()
    policy = WalkFinetunePolicy(
        "walk_finetune",
        robot,
        init_motor_pos,
        ckpt=ckpt_folders,
        exp_folder="tests/logging",
        env_cfg=env_cfg,
        finetune_cfg=finetune_cfg,
        is_real=False
    )

    server = RemoteServer(host='192.168.0.227', port=5007, policy=policy)
    server.start_receiving_data()
    while not server.exp_folder:
        time.sleep(0.1)
    if not os.path.exists(server.exp_folder):
        os.makedirs(server.exp_folder)
    policy.exp_folder = server.exp_folder
    policy.logger.set_exp_folder(server.exp_folder)
    # policy.logger.plot_updates()

    try:
        while server.is_running and server.client_thread.is_alive():
            # if len(policy.replay_buffer) == 47316 and len(ckpt_folders):
            #     server.push_policy_parameters(policy.abppo._policy_net.state_dict())
            if policy.learning_stage == "offline":
                if len(policy.replay_buffer) > finetune_cfg.offline_initial_steps and (len(policy.replay_buffer) + 1) % finetune_cfg.update_interval == 0:
                    for _ in range(policy.finetune_cfg.abppo_update_steps):
                        policy.offline_abppo_learner.update(policy.replay_buffer)
                    policy.policy_net.load_state_dict(policy.abppo._policy_net.state_dict())
                    server.push_policy_parameters(policy.abppo._policy_net.state_dict())  # Push latest parameters to agent A.
                    print("Replay buffer size:", len(policy.replay_buffer))
                if len(policy.replay_buffer) >= finetune_cfg.offline_total_steps:
                    policy.switch_learning_stage()
            elif policy.learning_stage == "online" and len(policy.replay_buffer) == finetune_cfg.online.batch_size:
                policy.online_ppo_learner.update(policy.replay_buffer, policy.total_steps - policy.finetune_cfg.offline_total_steps)
                policy.policy_net.load_state_dict(policy.online_ppo_learner._policy_net.state_dict())
                server.push_policy_parameters(policy.online_ppo_learner._policy_net.state_dict())
                print("Replay buffer size:", len(policy.replay_buffer))
            time.sleep(0.01)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Exception:", e)
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