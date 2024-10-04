# Jetson Orin

Jetson Orin is the on-board compute for the toddlerbot. We recommend Jetson Orin NX 16GB.

## Flash the System

We recommend using JetPack 6.0.

https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html#real-time-kernel-using-ota-update

```bash
sudo apt install dkms
sudo git clone https://github.com/paroj/xpad.git /usr/src/xpad-0.4
sudo dkms install -m xpad -v 0.4
```

- Add yourself to the correct user group [doc](https://github.com/NVIDIA/jetson-gpio)
- Be sure to modify the address of rules: e.g
```bash
sudo groupadd -f -r gpio
sudo usermod -a -G gpio $USER
sudo chown root.gpio /dev/gpiochip0
sudo chmod 660 /dev/gpiochip0

python -m pip install Jetson.GPIO
sudo cp ~/miniforge3/envs/toddlerbot/lib/python3.10/site-packages/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

- also add yourself to i2c and dialout group.
```bash
sudo usermod -aG i2c $USER
sudo usermod -aG dialout $USER
```
The usb-c port on Jetson is only for flashing, which means it's not fully functional. Do not plug cables into this port to transfer data.

Follow [these instructions](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) to install PyTorch. For reference, we install with
```
pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.02-cp38-cp38-linux_aarch64.whl
```

Set up NoMachine from [this page](https://downloads.nomachine.com/download/?id=118&distro=ARM) for remote desktop access.

Edit sudoers safely:
```
sudo visudo
```
Add a line for specific commands:
```
youruser ALL=(ALL) NOPASSWD: /bin/echo, /usr/bin/tee
```
This allows the user `youruser` to run echo and tee without a password. Ensure you replace youruser with the actual user that the script runs under.

Install miniforge: Download `Linux aarch64 (arm64)` from [their website](https://github.com/conda-forge/miniforge). Do NOT run the install script with sudo. Answer `yes` to all the options.

For the accuracy of teleoperation and logging over network, we need to install ntp package to sync time of the jetson to server.
```
sudo apt install ntp
sudo systemctl enable ntp
sudo service ntp start
```
