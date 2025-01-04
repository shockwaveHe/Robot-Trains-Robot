#!/bin/bash

# Define the different configurations for each experiment
robots=("toddlerbot")
envs=("walk")
config_overrides=(
    "PPOConfig.num_timesteps=300000000,PPOConfig.num_evals=3000,PPOConfig.seed=0"
    "PPOConfig.num_timesteps=300000000,PPOConfig.num_evals=3000,PPOConfig.seed=1"
    "PPOConfig.num_timesteps=300000000,PPOConfig.num_evals=3000,PPOConfig.seed=2"
    "PPOConfig.num_timesteps=300000000,PPOConfig.num_evals=3000,PPOConfig.seed=3"
)

# Iterate over all configurations
for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for config_override in "${config_overrides[@]}"; do
            echo "Running experiment with Robot: $robot, Env: $env, Config Override: $config_override"
            
            # Run the Python script with the current configuration
            python toddlerbot/locomotion/train_mjx.py --robot "$robot" --env "$env" --config-override "$config_override"
            
            # Optional: Add a small delay between experiments
            sleep 1
        done
    done
done
