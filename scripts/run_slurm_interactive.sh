#!/bin/bash

srun --account=move --partition=move-interactive --gres=gpu:l40s:1 --time=3-00:00:00 --mem-per-cpu=4G --cpus-per-task=8 --pty bash