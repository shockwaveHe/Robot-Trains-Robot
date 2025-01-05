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
* [on arya] run policy walk_finetune with real world sim