#!/usr/bin/env bash

set -e  # Exit on error

pause() {
    # $1 = message (optional)
    local msg="${1:-Press Enter to continue...}"
    read -p "$msg"
}

echo "Press enter to runing the arm_treadmill leader"
cd /home/kukabot/Projects/toddlerbot_internal
python -m toddlerbot.policies.run_policy --policy at_leader --sim finetune --ip 192.168.110.216             