cd toddlerbot_internal
# test teleoperation

python toddlerbot/manipulation/teleoperation.py --view_frame

# test teleoperation + simulator

# in one terminal run:
python toddlerbot/policies/run_policy.py --policy teleop_follower_pd --robot toddlerbot --vis view --task teleop_vr
# in the second terminal run:
python toddlerbot/policies/run_policy.py --policy teleop_vr_leader --robot toddlerbot_active

# test teleoperation + real robot
# on toddy run:
python toddlerbot/policies/run_policy.py --policy teleop_follower_pd --robot toddlerbot --sim real --task teleop_vr --ip 192.168.0.29
# replace the ip with your desktop ip
# on your desktop run:
 python toddlerbot/policies/run_policy.py --policy teleop_vr_leader --robot toddlerbot_active --ip 192.168.0.237
# replace the ip with toddy's ip