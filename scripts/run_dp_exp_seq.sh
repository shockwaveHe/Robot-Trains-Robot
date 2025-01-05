#!/bin/bash

# Define the different configurations for each experiment
# robot="toddlerbot_gripper"
# task="pick"
# time_strs=("20241216_162951 20241216_164545")
robot="toddlerbot"
task="hug"
time_strs=("20241217_162015")
configs=(
    "--weights imagenet"
    "--weights imagenet --obs-horizon 3"
    "--weights imagenet --obs-horizon 7"
    "--weights imagenet --pred-horizon 8"
    "--weights imagenet --action-horizon 3"
    "--weights imagenet --action-horizon 8"
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
