#!/usr/bin/env bash

set -e  # Exit on error

pause() {
    # $1 = message (optional)
    local msg="${1:-Press Enter to continue...}"
    read -p "$msg"
}

echo "Press enter to runing the swing arm leader"
cd /home/kukabot/Projects/toddlerbot_internal
python toddlerbot/policies/run_policy.py --policy swing_arm_leader --ip 192.168.110.216 --robot toddlerbot_2xm