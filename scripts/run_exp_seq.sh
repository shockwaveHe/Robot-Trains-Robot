#!/bin/bash

# Define the different configurations for each experiment
robots=("toddlerbot_OP3")
envs=("walk")
config_overrides=(
    "PPOConfig.batch_size=1024 PPOConfig.num_minibatches=1"
    "PPOConfig.batch_size=512 PPOConfig.num_minibatches=2"
    "PPOConfig.batch_size=256 PPOConfig.num_minibatches=4"
    "PPOConfig.batch_size=128 PPOConfig.num_minibatches=8"
)

# Iterate over all configurations
for robot in "${robots[@]}"; do
    for env in "${envs[@]}"; do
        for config_override in "${config_overrides[@]}"; do
            echo "Running experiment with Robot: $robot, Env: $env, Config Override: $config_override"
            
            # Run the Python script with the current configuration
            python toddlerbot/locomotion/train_mjx.py --robot "$robot" --env "$env" --config_override "$config_override"
            
            # Optional: Add a small delay between experiments
            sleep 1
        done
    done
done