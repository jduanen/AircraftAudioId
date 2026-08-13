#!/bin/bash
# Script to sweep the SPI bus frequencies to find the max working speed
#
# Make sure the boot-time overlay (in config.txt) isn't also loaded, so it doesn't
#  conflict with the live one we're about to apply
# Comment out, or remove, the `dtoverlay=cm4-sdspi` line in /boot/firmware/config.txt
#  for now, or just reboot once first so nothing's currently claiming SPI1 CE0

# Sweep loop -- no reboot, no recompile, each iteration is seconds
for freq in 400000 1000000 4000000 8000000 12000000; do
    sudo dtoverlay -r cm4-sdspi 2>/dev/null
    sleep 1
    sudo dtoverlay cm4-sdspi freq=$freq
    sleep 1
    echo "=== $freq Hz ==="
    dmesg | tail -5
    ls /dev/mmcblk* 2>&1
    echo
done
