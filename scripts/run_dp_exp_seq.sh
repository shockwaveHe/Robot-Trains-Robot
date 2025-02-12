#!/bin/bash

# Run DP experiments sequentially

# Define the different configurations for each experiment
robot="toddlerbot"
task="hug"
time_strs=("20250109_220617") 
# robot="toddlerbot_gripper"
# task="pick"
# time_strs=("20250110_194100")
# robot="toddlerbot"
# task="grasp"
# time_strs=("20250106_181537")
configs=(
    "--weights imagenet"
    "--weights imagenet --obs-horizon 5"
    "--weights imagenet --action-horizon 5"
    "--weights imagenet --action-horizon 3"
    "--weights imagenet --obs-horizon 3"
    "--weights imagenet --pred-horizon 8"
)

# Iterate over all configurations
for time_str in "${time_strs[@]}"; do
    for config in "${configs[@]}"; do
        echo "robot: $robot, task: $task, datasets: $time_str, config: $config"
        
        # Run the Python script with the current configuration
        python toddlerbot/manipulation/train.py --robot $robot --task $task --time-str "$time_str" $config
        
        # Optional: Add a small delay between experiments
        sleep 1
    done
done
