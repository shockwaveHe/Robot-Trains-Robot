#!/bin/bash

srun --account=move --partition=move-interactive --gres=gpu:a5000:1 --time=24:00:00 --mem-per-cpu=4G --cpus-per-task=8 --pty bash
