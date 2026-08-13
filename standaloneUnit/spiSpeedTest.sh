#!/bin/bash
#
# Check that the current SPI frequency results in reliable reads/writes


SPI_PART="/dev/mmcblk3p2"

# after picking a frequency and it successfully enumerated
if ??format??; then
    sudo mkfs.ext4 -F ${SPI_PART}
fi

if ??unmounted??; then
    sudo mount ${SPI_PART} /mnt/sdcard
fi

# make test file and checksum it
dd if=/dev/urandom of=/tmp/testfile bs=1M count=200
ORIG_CHKSUM=`sha256sum /tmp/testfile`
sudo cp /tmp/testfile /mnt/sdcard/testfile
sync  # make write complete
sudo umount /mnt/sdcard && sudo mount ${SPI_PART} /mnt/sdcard  # force a fresh read, not cached
NEW_CHKSUM=`sha256sum /mnt/sdcard/testfile`

# compare both checksums -- must match exactly
if ??comp??; then
    echo "GOOD"
else
    echo "BAD"
fi

