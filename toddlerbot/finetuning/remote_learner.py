import os
import time
import torch
from toddlerbot.finetuning import networks
from toddlerbot.finetuning.logger import FinetuneLogger
from toddlerbot.finetuning.finetune_config import FinetuneConfig
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.finetuning.dynamics import DynamicsNetwork, BaseDynamics
from toddlerbot.finetuning.server_client import RemoteServer
from toddlerbot.finetuning.networks import load_jax_params, load_jax_params_into_pytorch
from toddlerbot.finetuning.abppo import AdaptiveBehaviorProximalPolicyOptimization, ABPPO_Offline_Learner
NUM_ACTIONS = 12

if __name__ == "__main__":
    finetune_cfg = FinetuneConfig()
    action_size=NUM_ACTIONS
    observation_size=finetune_cfg.frame_stack * finetune_cfg.num_single_obs
    privileged_observation_size=finetune_cfg.frame_stack * finetune_cfg.num_single_privileged_obs
    value_hidden_layer_sizes=finetune_cfg.value_hidden_layer_sizes
    policy_hidden_layer_sizes=finetune_cfg.policy_hidden_layer_sizes
    activation_fn = torch.nn.ReLU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = networks.GaussianPolicyNetwork(
        observation_size=observation_size,
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=policy_hidden_layer_sizes,
        action_size=action_size,
        activation_fn=activation_fn
    ).to(device)
    policy_net_opt = torch.compile(policy_net)
    # Create value network
    value_net = networks.ValueNetwork(
        observation_size=privileged_observation_size,
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=value_hidden_layer_sizes,
        activation_fn=activation_fn
    ).to(device)

    Q_net_cls = networks.DoubleQNetwork if finetune_cfg.use_double_q else networks.QNetwork
    Q_net = Q_net_cls(
        observation_size=privileged_observation_size,
        action_size=action_size,
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=value_hidden_layer_sizes,
        activation_fn=activation_fn
    ).to(device)

    dynamics_net = DynamicsNetwork(
        observation_size=privileged_observation_size,
        action_size=action_size,
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=value_hidden_layer_sizes,
        activation_fn=activation_fn
    ).to(device)

    dynamics = BaseDynamics(device, dynamics_net, finetune_cfg)
    policy_path = os.path.join(
                "toddlerbot", "policies", "checkpoints", "walk_policy"
            )
    print(f"Loading pretrained model from {policy_path}")
    jax_params = load_jax_params(policy_path)
    load_jax_params_into_pytorch(policy_net, jax_params[1]["params"])
    data_folder = "results/stored/toddlerbot_walk_finetune_real_world_20250211_101354"
    replay_buffer = OnlineReplayBuffer(device, observation_size, privileged_observation_size, action_size, finetune_cfg.buffer_size, enlarge_when_full=finetune_cfg.update_interval * finetune_cfg.enlarge_when_full)
    if os.path.exists(os.path.join(data_folder, "buffer.npz")):
        replay_buffer.load_compressed(data_folder)
        print("Loaded replay buffer from", data_folder)
    server = RemoteServer(host='172.24.68.176', port=5000)
    server.start_receiving_data(replay_buffer)
    while not server.exp_folder:
        time.sleep(0.1)
    if not os.path.exists(server.exp_folder):
        os.makedirs(server.exp_folder)
    logger = FinetuneLogger(server.exp_folder, enable_logging=True)
    abppo = AdaptiveBehaviorProximalPolicyOptimization(device, policy_net, finetune_cfg)
    abppo_offline_learner = ABPPO_Offline_Learner(device, finetune_cfg, abppo, Q_net, value_net, dynamics, logger)
    # TODO: check why update ratio behavior changes when learning on local or remote
    while True:
        if len(replay_buffer) % 500 == 0:
            print("Replay buffer size:", len(replay_buffer))
            abppo_offline_learner.update(replay_buffer)
            logger.plot_queue.put((logger.plot_updates, []))
            server.push_policy_parameters(abppo._policy_net.state_dict())  # Push latest parameters to agent A.
            print("Pushed policy parameters to agent A.")
            print("Replay buffer size:", len(replay_buffer))
        time.sleep(0.01)