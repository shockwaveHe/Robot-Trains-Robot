Steps to fine-tune the model in the real world:
* connect windows machine to the ft sensor via USB
* connect windows machine to the NUC via ethernet
* run the following command on the windows machine:
```
z daqft
ca daq
python .\read_data_plot.py
```
* start the UR5 arm and treadmill
* connect power cable / battery to arya, flip the power switch
* [on the NUC] ssh to arya
* [on the NUC] start ft_application
* [on the NUC] run policy at_leader with finetuning sim
```
python -m toddlerbot.policies.run_policy --ip 10.5.6.248 --policy at_leader --sim finetune
```
* [on arya] run policy walk_finetune with real world sim
```
python toddlerbot/policies/run_policy.py --sim real --ip 10.5.6.243 --policy walk_finetune
```
Test toddler:
```
python toddlerbot/policies/run_policy.py --sim real --policy walk --command "0 0 0 0 0 0.1 0 0"
```