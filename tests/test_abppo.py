import os
import numpy as np
import torch
import minari
import gymnasium as gym
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import your toddlerbot modules
from toddlerbot.finetuning import networks
from toddlerbot.finetuning.logger import FinetuneLogger
from toddlerbot.finetuning.finetune_config import FinetuneConfig
from toddlerbot.finetuning.replay_buffer import OfflineReplayBuffer
from toddlerbot.finetuning.dynamics import DynamicsNetwork, BaseDynamics
from toddlerbot.finetuning.abppo import AdaptiveBehaviorProximalPolicyOptimization, ABPPO_Offline_Learner
from toddlerbot.finetuning.bc_ensemble import BehaviorCloningEnsemble

NUM_ACTIONS = 12

# Define a helper function to extract actions from the learned policy.
@torch.no_grad()
def get_action(policy_net, obs, device, deterministic=True):
    obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
    action_dist = policy_net(obs_tensor)
    if deterministic:
        # If available, use the mean (mode) of the Gaussian distribution.
        actions = action_dist.base_dist.mode
        for transform in action_dist.transforms:
            actions = transform(actions)
        log_prob = action_dist.log_prob(actions).sum()
    else:
        actions = action_dist.sample()
        log_prob = action_dist.log_prob(actions).sum()

    return actions.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

# Evaluation function
def evaluate_policy(env, policy, num_episodes=10):
    total_rewards, total_steps = [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward, episode_steps = 0, 0
        while not done:
            action, _ = get_action(policy, obs, device, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            episode_steps += 1
        total_steps.append(episode_steps)
        total_rewards.append(episode_reward)
    return total_steps, total_rewards

def termination_fn_hopper(obs, config = None):

    height = obs[:, 0]
    angle = obs[:, 1]
    not_done =  torch.isfinite(obs).all(axis=-1) \
                * (torch.abs(obs[:,1:]) < 100).all(axis=-1) \
                * (height > .7) \
                * (torch.abs(angle) < .2)
    done = ~not_done
    return done[:,None].cpu().numpy()


# TODO: test dynamics predict termination accuracy
if __name__ == "__main__":
    # --------------------------
    # Load Minari offline dataset and populate replay buffer
    # --------------------------
    # Load the dataset (here, "mujoco/hopper/medium-v0")
    dataset = minari.load_dataset("mujoco/hopper/expert-v0")
    # import ipdb; ipdb.set_trace()
    print("Loaded dataset with total steps:", dataset.total_steps)
    # --------------------------
    # Set up configuration and networks
    # --------------------------
    finetune_cfg = FinetuneConfig()

    finetune_cfg.policy_hidden_layer_sizes = (256, 256)
    finetune_cfg.value_hidden_layer_sizes = (512, 512)
    finetune_cfg.dynamics_hidden_layer_sizes = (256, 256, 256)

    action_size = dataset.action_space.shape[0]
    observation_size = dataset.observation_space.shape[0]
    privileged_observation_size = dataset.observation_space.shape[0]
    value_hidden_layer_sizes = finetune_cfg.value_hidden_layer_sizes
    policy_hidden_layer_sizes = finetune_cfg.policy_hidden_layer_sizes
    activation_fn = torch.nn.ReLU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Policy network (GaussianPolicyNetwork)
    policy_net = networks.GaussianPolicyNetwork(
        observation_size=observation_size,
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=policy_hidden_layer_sizes,
        action_size=action_size,
        activation_fn=activation_fn
    ).to(device)

    # Value network
    value_net = networks.ValueNetwork(
        observation_size=privileged_observation_size,
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=value_hidden_layer_sizes,
        activation_fn=activation_fn
    ).to(device)

    # Q network (or Double Q, based on config)
    Q_net_cls = networks.DoubleQNetwork if finetune_cfg.use_double_q else networks.QNetwork
    Q_net = Q_net_cls(
        observation_size=privileged_observation_size,
        action_size=action_size,
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=value_hidden_layer_sizes,
        activation_fn=activation_fn
    ).to(device)

    # Dynamics network
    dynamics_net = DynamicsNetwork(
        observation_size=privileged_observation_size,
        action_size=action_size,
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=value_hidden_layer_sizes,
        activation_fn=activation_fn
    ).to(device)
    dynamics = BaseDynamics(device, dynamics_net, finetune_cfg, terminate_fn=termination_fn_hopper, extract_obs_fn=lambda x: x)

    exp_folder = "tests/logging"
    logger = FinetuneLogger(exp_folder, enable_logging=True)

    # Instantiate the ABPPO learner and its adaptive policy wrapper
    # general configurations
    use_state_norm = False
    use_reward_norm = None

    # bc configurations
    bc_policies = 4
    bc_seed = 0
    bc_steps = int(4e5)
    bc_log_freq = 20
    bc_eval_freq = int(4e4)
    bc_layers = (256, 256)
    bc_lr = 1e-4
    bc_batch_size = 512
    bc_kl = "data"
    bc_alpha = 0.1
    bc_kl_type = "heuristic"
    # abppo configurations
    finetune_cfg.num_policy = 4
    finetune_cfg.eval_step = 100
    finetune_cfg.value_update_steps = int(4e5)
    finetune_cfg.dynamics_update_steps = int(4e5)
    finetune_cfg.bppo_steps = 10000

    num_iterations, num_eval_episodes = 1, 10
    abppo = AdaptiveBehaviorProximalPolicyOptimization(device, policy_net, finetune_cfg)
    abppo_offline_learner = ABPPO_Offline_Learner(device, finetune_cfg, abppo, Q_net, value_net, dynamics, logger)

    
    # Create a replay buffer with capacity equal to the dataset total steps.
    replay_buffer = OfflineReplayBuffer(
        device,
        obs_dim=observation_size,
        privileged_obs_dim=privileged_observation_size,
        action_dim=action_size,
        max_size=int(dataset.total_steps) + 1
    )
    
    # Iterate over all episodes in the dataset and store transitions.
    # We assume that privileged observations are identical to the observations.
    episode_returns = []
    for episode in dataset.iterate_episodes():
        observations = episode.observations  # shape (T, obs_dim)
        actions = episode.actions            # shape (T, action_dim)
        rewards = episode.rewards            # shape (T,)
        terminations = episode.terminations  # shape (T,) booleans
        truncations = episode.truncations    # shape (T,) booleans
        episode_returns.append(sum(rewards))
        T = len(observations)
        # For each transition, use t and t+1 (so we iterate until T-1)
        for t in range(T - 1):
            s = observations[t]
            s_p = observations[t]  # same as observation
            a = actions[t]
            r = np.array([rewards[t]])
            done = bool(terminations[t])
            truncated = bool(truncations[t])
            a_logprob = np.array([0.0])  # No action log probability in the dataset; set to zero.
            replay_buffer.store(s, s_p, a, r, done, truncated, a_logprob, raw_obs=s)
    print(f"Load dataset with {len(episode_returns)} episodes, average return: {np.mean(episode_returns)}, max return: {np.max(episode_returns)}")
    replay_buffer.compute_return(finetune_cfg.gamma)
    replay_buffer.normalize_reward(finetune_cfg.gamma, use_reward_norm)
    if use_state_norm:
        state_mean, state_std = replay_buffer.normalize_state()
    else:
        state_mean, state_std = 0.0, 1.0
    print("Replay buffer loaded with {} transitions".format(replay_buffer._size))
    preprocess_obs_fn = lambda x, y: x
    bc_ensemble = BehaviorCloningEnsemble(bc_policies, device, observation_size, bc_layers, action_size, preprocess_obs_fn, bc_lr, bc_batch_size, bc_kl, bc_kl_type)

    # Recover the environment from the dataset for evaluation
    eval_env = dataset.recover_environment(eval_env=True)
    print("Recovered evaluation environment:", eval_env)

    # bc training
    bc_folder = os.path.join(exp_folder, f"bc_policy_{bc_alpha}")
    print(f"BC folder: {bc_folder}")
    if not os.path.exists(bc_folder):
        best_bc_scores = np.zeros(bc_policies)
        best_bc_meta_score = 0
        pbar = tqdm(range(int(bc_steps)), desc='bc updating ......')
        for step in pbar:
            bc_losses = bc_ensemble.joint_train(replay_buffer, alpha=bc_alpha)
            if step % bc_log_freq == 0:
                logger.log_update(bc_loss_mean=bc_losses.mean(), bc_loss_std=bc_losses.std())
            if step % int(bc_eval_freq) == 0:
                current_bc_score = bc_ensemble.evaluation(eval_env, bc_seed, state_mean, state_std)
                mean_loss, mean_bc_score = bc_losses.mean(), current_bc_score.mean()
                print(f"Step: {step}, Loss: {mean_loss:.4f}, Score: {mean_bc_score:.4f}")
                
        index = [i for i in range(bc_policies)]
        bc_ensemble.ensemble_save(bc_folder, index)

    # Load the bc policies from the ensemble
    abppo.load_bc(bc_folder, policy_prefix="bc")
    eval_steps, eval_returns = evaluate_policy(eval_env, abppo._policy_net, num_eval_episodes)
    print("Loaded BC policy average return: {:.2f} ± {:.2f}".format(np.mean(eval_returns), np.std(eval_returns)))
    print(f"Evaluating the policy in the {eval_env} environment")
    for _ in range(num_iterations):
        print("Starting offline training...")
        # The update() method internally trains the Q/V networks, dynamics, and then runs bppo joint training.
        joint_loss = abppo_offline_learner.update(replay_buffer)
        print("Offline training complete. Final joint loss:", joint_loss)

        eval_steps, eval_returns = evaluate_policy(eval_env, abppo._policy_net, num_eval_episodes)
        print("Evaluation complete for {} episodes".format(num_eval_episodes)) 
        print("Average return: {:.2f} ± {:.2f}".format(np.mean(eval_returns), np.std(eval_returns)))
        print("Average steps: {:.2f} ± {:.2f}".format(np.mean(eval_steps), np.std(eval_steps)))
        logger.plot_updates()