#!/usr/bin/env bash

set -e  # Exit on error

pause() {
    # $1 = message (optional)
    local msg="${1:-Press Enter to continue...}"
    read -p "$msg"
}

echo "Press enter to runing the swing policy"
cd ~/projects/toddlerbot_internal
python toddlerbot/policies/run_policy.py --sim real --ip 192.168.110.232 --policy swing --robot toddlerbot_2xm --no-plot