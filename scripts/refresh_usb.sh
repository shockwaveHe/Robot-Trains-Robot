#!/bin/bash

# Force to refresh the USB devices
# Therefore, you don't need to unplug and plug the USB devices again
# This script is useful when the USB devices are not working properly
for i in /sys/bus/usb/devices/*/authorized; do
  if [ -f "$i" ]; then  # Check if the file exists
    echo "Resetting the usb device at $i"
    echo 0 | sudo tee "$i" > /dev/null
    echo 1 | sudo tee "$i" > /dev/null
  fi
done
