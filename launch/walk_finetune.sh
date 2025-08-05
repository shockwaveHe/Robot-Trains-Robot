#!/usr/bin/env bash

set -e  # Exit on error

pause() {
    # $1 = message (optional)
    local msg="${1:-Press Enter to continue...}"
    read -p "$msg"
}

echo "Press enter to runing the walk finetune policy"
cd ~/projects/toddlerbot_internal
python toddlerbot/policies/run_policy.py --sim real --ip 192.168.110.232 --policy walk_finetune --robot toddlerbot_2xm  --ckpt release_latent_torch_film_4e-5_ar_0.3_ldr
