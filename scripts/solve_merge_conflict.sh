git checkout --theirs toddlerbot/descriptions && git add toddlerbot/descriptions
git checkout --theirs toddlerbot/actuation && git add toddlerbot/actuation
git checkout --theirs toddlerbot/sensing && git add toddlerbot/sensing
find toddlerbot/policies -type f | grep -v -E "(run_policy.py|mjx_policy.py)" | xargs -I {} sh -c 'git checkout --theirs "{}" && git add "{}"'
bash scripts/onshape_to_robot.sh --robot toddlerbot