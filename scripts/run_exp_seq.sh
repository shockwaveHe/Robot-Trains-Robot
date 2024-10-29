#!/bin/bash

# Define the different configurations for each experiment
robots=("toddlerbot_OP3")
envs=("walk")
config_overrides=(
    "HangConfig.init_hang_force=2.0 HangConfig.final_hang_force=0.0 HangConfig.hang_force_decay_episodes=150"
    "HangConfig.init_hang_force=2.0 HangConfig.final_hang_force=0.0 HangConfig.hang_force_decay_episodes=100"
    "HangConfig.init_hang_force=2.0 HangConfig.final_hang_force=0.0 HangConfig.hang_force_decay_episodes=200"
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