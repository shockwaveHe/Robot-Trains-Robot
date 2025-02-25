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
    ckpt_folder = "results/toddlerbot_walk_finetune_real_world_20250224_222209"

    # policy.abppo_offline_learner.update(policy.replay_buffer)
    # policy.logger.plot_updates()
    policy = WalkFinetunePolicy(
        "walk_finetune",
        robot,
        init_motor_pos,
        ckpt=[ckpt_folder],
        exp_folder="tests/logging",
        env_cfg=env_cfg,
        finetune_cfg=finetune_cfg,
        is_real=False
    )
    server = RemoteServer(host='172.24.68.176', port=5007, policy=policy)
    server.start_receiving_data()
    while not server.exp_folder:
        time.sleep(0.1)
    if not os.path.exists(server.exp_folder):
        os.makedirs(server.exp_folder)
    policy.exp_folder = server.exp_folder
    policy.logger.set_exp_folder(server.exp_folder)

    try:
        while server.is_running and server.client_thread.is_alive():
            if len(policy.replay_buffer) > 50000 and (len(policy.replay_buffer) + 1) % 100 == 0:
                print("Replay buffer size:", len(policy.replay_buffer))
                policy.abppo_offline_learner.update(policy.replay_buffer)
                server.push_policy_parameters(policy.abppo._policy_net.state_dict())  # Push latest parameters to agent A.
                print("Pushed policy parameters to agent A.")
                print("Replay buffer size:", len(policy.replay_buffer))
            time.sleep(0.01)
            if (len(policy.replay_buffer) + 1) % 3000 == 0:
                policy.logger.plot_queue.put((policy.logger.plot_rewards, []))
                policy.logger.plot_queue.put((policy.logger.plot_updates, []))
    except Exception as e:
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