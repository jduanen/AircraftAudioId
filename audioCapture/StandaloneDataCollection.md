# StandaloneDataCollection.md

Documentation for the Standalone Data Collection System — a sealed, self-contained field unit (RasPi CM4) that can be placed in different locations, retrieved at a later point in time, and its data harvested and added to the local dataset. See [LocalDataCollection.md](LocalDataCollection.md) for the WLAN-connected two-machine setup used within reach of the WLAN.

---

## Overview

This is a sealed, completely self-contained, data collection unit that can be left unattended (and not connected to the internet), for extended periods of time and then retrieved and the data harvested, post-processed, and added to the dataset.

In addition the using the same microphone as used in the local system (to gather the audio clips), this system includes an ADS-B receiver and its associated antenna (used to provide ground truth for the audio samples), and a GPS receiver and antenna (used to get the location where the audio samples are made). There is also a WiFi radio and antenna that can be used to interact with the (normally-sealed) unit.

## Hardware

This unit is based on a RasPi CM4 module with a base board that provides connectors for four USB ports, five serial ports, an I2C bus, and a SPI bus. The CM4 modules is connected to the same type of USB microphone as the Remote Audio Capture Unit, as well as a (FlightAware) USB-SDR ADS-B receiver dongle, and a GPS receiver (with a serial interface).

The USB-SDR dongle is attached via a USB-A connector to the USB1 connector and the microphone is connected via a USB-C connector to the USB2 connector.

A micro-SD socket for the bulk storage is connected to the SPI1 connector.

The Ochin daughterboard schematics must be consulted to determine how to wire the JST-GH-1.25 4-pin (USB) and 6-pin (UART and SPI) connectors.

The unit has a water-resistant power connector for the 12VDC (@?A) power supply required to operate the unit. This switch is lighted and is connected at one end to the 3V3 pin on UART0/1, and the other side is connected to the RTS pin on USART4.

A separate momentary pushbutton (not the illuminated power switch above) is wired to USART4's RXD pin, for triggering a graceful shutdown without needing to pull power.

The enslosure is a cast aluminium case that serves as a heat-sink for the CM4 module (and, if possible, the USB-SDR dongle).
The three (ADS-B 1090MHz, WiFi, and GPS) antennas are mounted onto the enclosure in such a manner as to resist water infiltration.

## Software

The software for this unit builds on that developed for the Local Data Collection system.

The (software-controlled) illuminated power switch is the only indication that the unit provides that it is functioning correctly. It can signal correct operation by staying on continuously, and it can indicate something is wrong by blinking the switch's LED. If the LED is off, this indicates that the system either has no power, or has a hard failure and the sofware isn't running. In practice: solid on = healthy (GPS fix acquired, audio capturing, storage OK); blinking = starting up (still waiting for a GPS fix) or a software-detected problem (most commonly low storage); off = unpowered or crashed.

It is expected that, in order to gather sufficient data, this unit will remain in a location for days to months at a time. This means that the unit must contain sufficient storage space to contain all of the samples and metadata generated before the unit is retreived and the information dumped. Once free storage drops below a threshold, new recordings are halted (never auto-deleted) and the LED switches to its error-blink pattern, signaling that the unit needs retrieval.

Because the unit is moved by power-cycling it (there's no way to relocate a sealed, running unit), it reads its position from the on-board GPS receiver exactly once at startup — blocking until a fix is acquired — and uses that fixed position for the entire run. There is no runtime location-change detection or mid-run restart; a fresh fix is simply acquired the next time the unit boots.

A separate shutdown button (see "Hardware" above) triggers a full, graceful `systemctl poweroff` — not just stopping `standaloneRecorder`. It runs as its own independent service (`shutdownButton.service`) so it keeps working even if the recording service has crashed. Combined with the CM4 bootloader's `POWER_OFF_ON_HALT=1` EEPROM setting (see "Installation" below), a press brings the unit to genuine low-power state rather than just an idle halt, safe to physically power off at that point.

### Installation

#### Install 64-bit Trixie Lite OS on the CM4's eMMC

* on the daughtercard: hold boot mode button, apply power to pads, then release boot button
* connect the CM4 to host via the daughtercard's USB-C connector and run `rpiboot` on the host
  - should see a raw memory device on the host
* run `rpi-imager` to flash the CM4's on-board eMMC
* do update and global installs
  - `sudo apt update`
  - `sudo apt upgrade`
  - `sudo apt install -y python3-pip python3-venv`
* add select linuxTools
  - `mkdir ~/bin`
  - `scp jdn@gpuServer1.lan:Code/linuxTools/scripts/{maxTemp,rssi,volts}.sh ~/bin/`

#### Enable the status LED GPIO (see standaloneUnit/cm4-led-gpio.dts)

* the LED is wired between the 3V3 pin on UART0/1 and RTS4 (USART4)
  - one leg fixed high, the other pulled by this GPIO line, making it active-low
  - StatusLed already sets LineSettings(active_low=True) to handle this
* RTS4 is BCM GPIO11 on the Ochin baseboard, matching `--ledLine`'s default
* `sudo apt install -y device-tree-compiler`
* `dtc -@ -I dts -O dtb -o cm4-led-gpio.dtbo cm4-led-gpio.dts`
* `sudo install -m 0644 cm4-led-gpio.dtbo /boot/firmware/overlays/cm4-led-gpio.dtbo`
* add `dtoverlay=cm4-led-gpio` to `/boot/firmware/config.txt`
* after reboot, confirm the line appears: `sudo gpiodetect` / `sudo gpioinfo gpiochip0`
* sanity-check polarity with the LED itself before trusting it as a status indicator
  - **confirmed:** `sudo gpioset --active-low -c gpiochip0 11=1` lights the LED — verifies the RTS4/3V3 wiring and active-low polarity match `StatusLed`'s `LineSettings(active_low=True)`
  - **found:** releasing a driven GPIO line on the BCM2711 does *not* return it to floating — the pin keeps driving its last-set level until explicitly changed or the board is power-cycled (confirmed: `Ctrl-C` on `gpioset ...11=1` left the LED on; `Ctrl-C` on `...11=0` left it off). This means "off" only reliably means "never touched since boot" — a hard crash (`SIGKILL`, panic) after the LED has been driven can leave it stuck in whatever state it was last in, not necessarily off
    * mitigated: `scripts/standaloneRecord.py` now converts `SIGTERM` (what `systemd stop`/`restart` sends) into the same graceful-shutdown path `Ctrl-C` already used, so the LED is explicitly driven off on any orderly stop — confirmed via a mocked-GPIO test that this runs even if `SIGTERM` arrives mid-startup (before the main recording loop)
    * still open: this doesn't cover `SIGKILL`/kernel panics/hangs — a systemd watchdog (`WatchdogSec=` + periodic `sd_notify` from the health-loop thread) would close that gap but hasn't been added yet

#### Enable the shutdown button GPIO (see standaloneUnit/cm4-shutdown-button.dts)

* a separate momentary switch (not the illuminated power switch/LED above) is wired to RXD4 (USART4) and GND
  - with RTS4 claimed by the LED and CTS4 by GPS PPS, RXD4 was the last free USART4 pin — chosen over TXD4 since it's natively an input, matching the button's direction
* RXD4 is BCM GPIO9 on the Ochin baseboard, matching `shutdownButtonWatch.py`'s `--line` default
* pull-up, edge detection, and debounce are all configured in software (`gpiod.LineSettings` in `standalone/shutdownButton.py`), not in the overlay — the overlay only handles pin muxing, same separation of concerns as the LED
* `dtc -@ -I dts -O dtb -o cm4-shutdown-button.dtbo cm4-shutdown-button.dts`
* `sudo install -m 0644 cm4-shutdown-button.dtbo /boot/firmware/overlays/cm4-shutdown-button.dtbo`
* add `dtoverlay=cm4-shutdown-button` to `/boot/firmware/config.txt`
* after reboot, confirm the line appears: `sudo gpioinfo gpiochip0`
* set the CM4 bootloader EEPROM's `POWER_OFF_ON_HALT` so a `systemctl poweroff` actually drops the module to low power rather than idling until the next reboot:
  - `sudo rpi-eeprom-config --edit`
  - add/change: `POWER_OFF_ON_HALT=1`
  - save and exit — the update is staged and applied on next reboot
  - `sudo reboot`
  - confirm it took: `sudo rpi-eeprom-config | grep POWER_OFF_ON_HALT`

#### Enable GPS PPS on CTS4 (USART4)

* with RTS4 claimed by the status LED and USART4's TX/RX unused, CTS4 is free
  - we repurpose it as the GPS receiver's PPS (pulse-per-second) input
    * for tighter chrony time discipline than NMEA-only timing alone
* unlike the status LED, this needs no custom overlay
  - Linux already ships a stock `pps-gpio` overlay that turns any GPIO into a PPS source device
* CTS4 is BCM GPIO10 on the Ochin baseboard schematic
* add to `/boot/firmware/config.txt`:
  - `dtoverlay=pps-gpio,gpiopin=10`
* verify after reboot:
  - `sudo apt install pps-tools`
  - `ppstest /dev/pps0`  # should print a timestamp once per second
* GPS PPS outputs are typically a 3.3V TTL pulse, directly compatible with a CM4 GPIO input
  - this was confirmed against the GPS module's datasheet
* the GPU module is a GT-U7, and its pinout is:
  - VCC: power input 3.3V
  - GND: ground
  - TXD: GPS serial data, defaults to NMEA and UBX responses
  - RXD: (UBX/NMEA format) configuration commands from the host
  - PPS: one pulse-per-second time reference
* see "Set up GPS time discipline" below for wiring `/dev/pps0` into chrony via gpsd

#### Enable the micro-SD on the SPI Bus

* describe the SPI SD socket in the device tree in the file 'cm4-sdspi.dts' (standaloneUnit/cm4-sdspi.dts)
  - create Device Tree overlay for the SPI bus and CS wired to the socket
    * SPI_0, CE_1, 3.3V-only, no card-detect switch
    * CS: GPIO18, SPI1_CE0_N
    * MISO: GPIO19, SPI1_MISO
    * MOSI: GPIO20, SPI1_MOSI
    * SCLK: GPIO21, SPI1_SCLK
  - the SPI CE signal is connected to the CM4's GPIO18 pin -- i.e., CE_1
* compile and install the Device Tree overlay
  - `sudo apt update`
  - `sudo apt install -y device-tree-compiler`
  - `dtc -@ -I dts -O dtb -o cm4-sdspi.dtbo cm4-sdspi.dts`
  - `sudo install -m 0644 cm4-sdspi.dtbo /boot/firmware/overlays/cm4-sdspi.dtbo`
  - check it
    * `ls -l /boot/firmware/overlays/cm4-sdspi.dtbo`
  - enable it by adding 'dtoverlay=cm4-sdspi' to '/boot/firmware/config.txt'
    * do not use 'dtparam=spi=on' or the 'spi1-1cs/spi1-2cs/spi1-3cs' overlay
  - check if generic spidev device gets created on SPI1 CE0
    * if so, it conflicts with the SD-card driver
    * `ls -l /dev/spidev*`
      - should show nothing for spi1
      - if spidev1.0 exists, something is still claiming the bus generically
    * `dmesg | grep -Ei 'spi|mmc'`
      - look for mmc_spi binding to spi1.0 and a new mmcblkN device
    * `ls /dev/mmcblk*`
      - this should be the card itself, once mmc_spi has claimed it
  - for testing, comment out the dtoverlay line in config.txt
    * to test:
      - `echo spi1.0 | sudo tee /sys/bus/spi/drivers/mmc_spi/bind`
        * this is a manual, no-overlay sanity check
      - temporarily use the stock 'spi1-1cs' overlay instead (which mux's the pins correctly and creates /dev/spidev1.0), unbind 'spidev' from it, and hand-bind 'mmc_spi' to the same device path at runtime
      - this is a reasonable way to confirm the wiring and the card itself both work before trusting the custom overlay's DT syntax
        * worth doing first, since it isolates "is the hardware/wiring good" from "is my custom .dts correct"
* **confirmed working:** `mmc_spi` binds to `spi1.0` and registers an MMC host with `cm4-sdspi.dts`
  - **`spi-max-frequency` is card-specific, not just wiring-specific — always sweep and verify per-card, don't trust a number from a different card.** A first card failed even 12 MHz with `mmc_spi spi1.0: no support for card's volts` on every boot; 400 kHz (the SD spec's own mandated power-up handshake speed — `mmc_spi` doesn't auto-negotiate a slower probe clock, it uses `spi-max-frequency` throughout, including CMD0/CMD8/OCR) was the only thing that worked for that card. A second (currently in-use) card, on the identical wiring/overlay, swept cleanly all the way to 24 MHz — 60x the first card's ceiling. Don't assume either number transfers to a new card.
  - **fast, no-reboot way to sweep speeds**: `spi-max-frequency` is exposed as a live-overridable `freq` parameter (see `cm4-sdspi.dts`'s `__overrides__` block) via Raspberry Pi's configfs-based live overlay loading — no recompile, no reboot per test:
    ```bash
    # unmount first — the overlay can't be removed/reapplied under a live filesystem
    sudo systemctl stop standaloneRecorder 2>/dev/null
    sudo umount /home/jdn/Code/AircraftAudioId/recordings 2>/dev/null

    for freq in 400000 1000000 4000000 8000000 12000000 16000000 20000000 24000000; do
        sudo dtoverlay -r cm4-sdspi 2>/dev/null
        sleep 1
        sudo dtoverlay cm4-sdspi freq=$freq
        sleep 1
        echo "=== $freq Hz ==="; dmesg | tail -5; ls /dev/mmcblk3* 2>&1; echo
    done
    ```
    watch for `no support for card's volts` or a missing `/dev/mmcblkN` — that marks where it broke
  - **live overlay churn degrades after repeated cycles** — expect `OF: overlay: WARNING: memory leak will occur if overlay removed` after several add/remove cycles, and eventually `Failed to apply overlay '0_cm4-sdspi' (kernel)` with no new dmesg output at all (state corruption, not a new hardware problem). `sudo reboot` clears it. Past ~5-8 live cycles in one session, prefer testing one value at a time via `config.txt` (`dtoverlay=cm4-sdspi,freq=<value>` + reboot) over continued live cycling.
  - **enumeration succeeding isn't proof the speed is safe for sustained transfer** — it only confirms the init handshake worked, not that data written at that clock won't get corrupted by marginal signal integrity. Verify with an actual write+read-back checksum before trusting a value:
    ```bash
    dd if=/dev/urandom of=/tmp/testfile bs=1M count=200
    sha256sum /tmp/testfile
    sudo cp /tmp/testfile /mnt/testfile   # mount point used during testing
    sync
    sudo umount /mnt && sudo mount /dev/mmcblk3p1 /mnt   # force a fresh read, not cached
    sha256sum /mnt/testfile   # must match exactly
    ```
  - **committed value: 16 MHz** — deliberately short of the tested-clean 24 MHz ceiling, to leave real margin for temperature/voltage/connector-wear drift over months of unattended field deployment rather than running at the edge of what merely passed once on the bench. Actual throughput needs for this workload are modest regardless (mono 44.1kHz int16 WAV is ~86 KB/s raw) — the point of testing higher speeds is safety margin, not because more speed was actually needed.

#### Mount the micro-SD as recordings storage

The micro-SD card is the unit's bulk storage (see "Hardware" above) — mount it at `recordings/` inside the repo checkout so `standaloneRecorder.service`'s existing `--outputDir` default needs no changes.

* identify the device (distinct from the CM4's boot eMMC)
  - `lsblk` or `dmesg | grep -i mmc` — the SD-over-SPI card registers separately from the eMMC (which is normally `/dev/mmcblk0`)
* partition and format — **destructive**, skip if the card already has data you need
  - `sudo parted /dev/mmcblkX --script mklabel gpt mkpart primary ext4 0% 100%`
  - `sudo mkfs.ext4 /dev/mmcblkXp1`
  - ext4 over exFAT/FAT32: native Linux permissions/symlinks, no 4 GB file-size ceiling, and journaling matters for a card that can lose power ungracefully in the field
* get a stable identifier for `/etc/fstab` (device names can shift across boots with multiple mmc devices)
  - `sudo blkid /dev/mmcblkXp1`
* if `recordings/` already has data on the eMMC from earlier bring-up testing, move it onto the new card first — mounting over the directory hides (does not delete) its previous contents, but don't rely on that
  - `mkdir -p /home/jdn/Code/AircraftAudioId/recordings`
* add to `/etc/fstab` (`nofail` so a missing/faulty card doesn't hang boot on an unattended unit)
  - `UUID=<uuid-from-blkid>  /home/jdn/Code/AircraftAudioId/recordings  ext4  defaults,noatime,nofail  0  2`
* mount and verify without rebooting
  - `sudo mount -a`
  - `df -h /home/jdn/Code/AircraftAudioId/recordings`
* **fix ownership** — `mkfs.ext4` leaves the filesystem root owned `root:root`; `standaloneRecorder.service` runs as `User=jdn`, so without this it fails with `PermissionError: [Errno 13] Permission denied` trying to create `recordings/audio`/`recordings/metadata` (confirmed on hardware)
  - `sudo chown jdn:jdn /home/jdn/Code/AircraftAudioId/recordings`
* make `standaloneRecorder.service` wait for the mount rather than starting and writing straight into the underlying eMMC root filesystem if the card isn't mounted yet — already added: `RequiresMountsFor=/home/jdn/Code/AircraftAudioId/recordings` in the service's `[Unit]` section

#### Install the Application Software on the Standalone Device

* Create venv (first time only)
  - `mkdir -p Code/AircraftAudioId/standaloneUnit`
  - `cd Code/AircraftAudioId/standaloneUnit`
  - `python3 -m venv venv`

* Install the native PortAudio library (sounddevice's pip package is only a
  ctypes wrapper around it — without this, `import sounddevice` fails with
  `OSError: PortAudio library not found`)
  - `sudo apt install libportaudio2`

* Activate venv and install packages
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`

* Sync selected parts of host repo to the device
  - **`../scripts/syncToStandalone.sh`**: run on the server, not the CM4
    * `bash scripts/syncToStandalone.sh <standalone-hostname-or-ip> [--skipFaaDb]`
    * syncs `src/aircraftAudio/`, `scripts/standaloneRecord.py`, `standaloneUnit/` (including `cm4-sdspi.dts`/`cm4-led-gpio.dts`), and `data/ReleasableAircraft/` (skip with `--skipFaaDb` once already on the device — it's ~500 MB on first sync)
    * does not create the venv (see Installation steps above) or restart `standaloneRecorder.service`

* Set up local ADS-B capture (readsb/dump1090-fa against the FlightAware USB-SDR dongle)
  - `sudo apt update`
  - **`sudo apt install readsb` is not enough on this image** — confirmed on hardware: the generic Trixie-archive package is compiled *without* RTL-SDR support at all. `readsb --device-type rtlsdr ...` fails immediately with `SDR type '0' not recognized`, and the SDR types listed in `readsb --help` (`modesbeast`, `gnshulc`, `ifile`, `none`) don't include `rtlsdr` — the package wasn't built against `librtlsdr`. FlightAware's own apt repo builds `readsb`/`dump1090-fa` with RTL-SDR support baked in (that's their whole product line), but as of this bring-up it doesn't yet have Trixie (Debian 13) packages published:
    * `wget https://flightaware.com/adsb/piaware/files/packages/pool/piaware/f/flightaware-apt-repository/flightaware-apt-repository_1.2_all.deb`
    * `sudo dpkg -i flightaware-apt-repository_1.2_all.deb && sudo apt update`
    * if this has a Trixie package by the time you're reading this, `sudo apt install readsb` from this repo should just work — check `apt-cache policy readsb` first
  - **build from source instead** (`wiedehopf/readsb`, which is what the confirmed-working build below is based on):
    * `sudo apt install -y build-essential librtlsdr-dev libusb-1.0-0-dev pkg-config git debhelper help2man libzstd-dev libncurses-dev zlib1g-dev`
    * `git clone https://github.com/wiedehopf/readsb.git && cd readsb`
    * **RTL-SDR support is gated behind a Debian *build profile*, not a plain `make` flag** — `debian/rules` only appends `RTLSDR=yes` to the build when `DEB_BUILD_PROFILES` contains `rtlsdr` (see its `CONFIG_SWITCH` logic). A bare `dpkg-buildpackage -b` silently produces the same RTL-SDR-less binary as the apt package:
      - `dpkg-buildpackage -b -Prtlsdr`
    * if you rebuild after an earlier attempt, run `make clean` first — debhelper tracks build completion via stamp files under `debian/.debhelper/` and can silently skip re-running the compile step (reusing the previous, RTL-SDR-less build) even when the build profile changes
    * install the result: `cd .. && sudo dpkg -i readsb_*.deb && sudo apt install -f` (ignore the `-dbgsym` package, that's debug symbols only)
    * **verify with a real invocation, not `--help`**: the `readsb --help` "supported SDR types" listing turned out to be stale/generic text that didn't reflect actual compiled support either way in testing — confirm with `sudo readsb --device 0 --device-type rtlsdr --gain -10 --ppm 0 --net` instead; getting past the "SDR type not recognized" error (even into a different, device-specific error) confirms it's genuinely compiled in
  - grant the (unprivileged) `readsb` system user permission to open the RTL-SDR USB device
    * without this, the service fails fast with `FATAL: rtlsdr: error querying device #0: Permission denied` — a manual `sudo readsb ...` test working is not enough to confirm this, since root bypasses the permission check the service account doesn't
    * `sudo apt install rtl-sdr` (ships the udev rule granting the `plugdev` group access to known RTL-SDR vendor/product IDs — `librtlsdr-dev` alone, installed earlier for the build, does not include this)
    * `sudo usermod -aG plugdev readsb`
    * `sudo udevadm control --reload-rules && sudo udevadm trigger`
  - configure it to read from the USB-SDR dongle: `/etc/default/readsb` sets four env vars the systemd unit's `ExecStart` references (`$RECEIVER_OPTIONS $DECODER_OPTIONS $NET_OPTIONS $JSON_OPTIONS`) — confirmed working values for the FlightAware dongle used here:
    ```
    RECEIVER_OPTIONS="--device 0 --device-type rtlsdr --gain auto --ppm 0"
    DECODER_OPTIONS="--max-range 450 --write-json-every 1"
    NET_OPTIONS="--net --net-ri-port 30001 --net-ro-port 30002 --net-sbs-port 30003 --net-bi-port 30004,30104 --net-bo-port 30005"
    JSON_OPTIONS="--json-location-accuracy 2 --range-outline-hours 24"
    ```
  - enable and start service
    * `sudo systemctl enable --now readsb`
    * `sudo journalctl -u readsb -n 50 --no-pager` — confirm no `Permission denied`/`sdrOpen() failed` errors
  - **readsb only writes `aircraft.json` to disk (`/run/readsb/`, per the unit's `--write-json /run/readsb`) — it does not serve HTTP itself.** Something else has to serve that directory for `curl`/`ReadsbClient` to reach it at all; `--net` alone (enabling readsb's own TCP ports for raw/Beast/SBS feeds) does not cover this
    * lean option (no web UI needed on a sealed, air-gapped unit — just the raw JSON): `lighttpd` with a symlink into readsb's output directory
      - `sudo apt install lighttpd`
      - `sudo ln -s /run/readsb /var/www/html/data` (the symlink itself is on the persistent root filesystem; its target, `/run/readsb`, is tmpfs and gets recreated fresh by systemd's `RuntimeDirectory=readsb` on every readsb start, so this survives reboots without needing to be redone)
      - `sudo systemctl enable --now lighttpd`
      - if `curl` still 404s, lighttpd's default config may have `server.follow-symlink` disabled — use an explicit `alias.url += ( "/data/" => "/run/readsb/" )` in its config instead
    * fuller option: install `tar1090` for a full map-based web UI, which configures lighttpd and serves at `/tar1090/data/aircraft.json` instead — not needed just to reach the JSON endpoint, but an option if the web UI itself is useful for field debugging
    * whichever path, it must match `--readsbUrl` passed to `scripts/standaloneRecord.py` (`standaloneRecorder.service`'s default is `http://localhost/data/aircraft.json`)
  - verify: `curl http://localhost/data/aircraft.json`

* Set up GPS time discipline (gpsd + chrony, since this unit has no internet for NTP)
  - the GPS receiver is wired to UART0/1 (for wiring/routing reasons, not USART4 — no conflict with the LED/PPS pins, which are on a separate peripheral either way)
  - `enable_uart=1` in `/boot/firmware/config.txt` is enough to bring up this UART — no additional `dtoverlay=uartN` needed, since it's the SoC's primary UART
  - use `/dev/serial0` rather than a specific `ttyAMA*`/`ttyS0` name: the firmware/udev symlink `/dev/serial0` always points at whichever device is actually the primary UART, which otherwise depends on Bluetooth presence/config
  - `sudo apt install gpsd chrony`
  - **give `/dev/pps0` read/write group access** — it's created `root:root 660` by default, and gpsd's service account can't open it otherwise (silently — no error, PPS just never links):
    * `sudo tee /etc/udev/rules.d/99-pps.rules <<< 'SUBSYSTEM=="pps", GROUP="dialout", MODE="0660"'`
    * `sudo udevadm control --reload-rules && sudo udevadm trigger`
    * `gpsd`'s system user is already in `dialout` (needed for `/dev/serial0`), so no separate group-membership step is needed
  - **point gpsd at both the serial device and the PPS device explicitly** — they're electrically/logically unrelated (unlike a USB GPS dongle where both come through one device the kernel can auto-link), so gpsd has no way to know they belong to the same physical unit unless told:
    * `sudo ex /etc/default/gpsd`
      - `DEVICES="/dev/serial0 /dev/pps0"` (serial device first — gpsd links a bare `/dev/pps*` entry to the most recently opened serial device)
      - `GPSD_OPTIONS="-n"` (poll immediately at startup rather than waiting for a client to connect — see below for why that matters here)
  - **bypass gpsd's socket activation** — Debian's `gpsd.socket`/`gpsd.service` pair only starts the actual daemon on-demand when a client connects to gpsd's own protocol port. Nothing in this system ever does that (`GpsClient` reads `/dev/serial0` directly, bypassing gpsd entirely — gpsd here exists solely for the chrony SHM bridge), so left as shipped, gpsd would sit dormant and never poll the device at all:
    * `sudo systemctl disable --now gpsd.socket`
    * `sudo systemctl enable --now gpsd.service`
  - verify raw NMEA output before wiring in GpsClient: `cat /dev/serial0`
  - **confirmed SHM unit mapping** (verified via `gpsd -N -D5 -n /dev/serial0 /dev/pps0`, watching `journalctl -u gpsd` for `ntpshm_put` lines — do not assume the "obvious" numbering, verify it on the actual hardware/gpsd version, since it wasn't what gpsd's own allocation-time log first suggested):
    * `SHM(0)` — coarse time from `/dev/serial0`'s NMEA sentences, updated once/sec
    * `SHM(1)` — allocated but unused once `/dev/pps0` is given as its own separate device (this is gpsd's *default* pairing for a serial-only setup with no external PPS device; giving it one explicitly redirects the real PPS data elsewhere, per the next line)
    * `SHM(2)` — the real hardware PPS assertions from `/dev/pps0`, confirmed via repeating `ntpshm_put(NTP2, ...) /dev/pps0 ... accepted` log lines once/sec
  - discipline chrony from gpsd's shared-memory (SHM) time reference
    * `sudo systemctl stop systemd-timesyncd`
    * `sudo systemctl disable systemd-timesyncd`
    * `sudo ex /etc/chrony/chrony.conf`
      - comment out default network pools (no internet on this unit)
      - add: `refclock SHM 0 offset 0.0 delay 0.2 refid GPS` (coarse, from NMEA)
      - add: `refclock SHM 2 offset 0.0 refid PPS precision 1e-7` (precise, PPS-disciplined — **not `SHM 1`**, see mapping above)
    * `sudo systemctl restart chronyd`
  - **`SHM(0)` needs a capability grant to be readable at all** — gpsd hard-codes `SHM(0)`/`SHM(1)` as `root:root 0600` by design (reserved for a co-resident *root*-privileged consumer, historically old-style `ntpd`); chrony deliberately drops root privileges and can't read a `0600` root-owned segment through any `chrony.conf` setting. `SHM(2)` and up are `0666` (world-readable) so don't need this, but the `GPS`/`SHM(0)` line does:
    * `sudo systemctl edit chronyd`
    * add:
      ```
      [Service]
      AmbientCapabilities=CAP_IPC_OWNER
      ```
    * `sudo systemctl restart chronyd`
  - check status
    * `chronyc sources` — both `GPS` and `PPS` should show climbing `Reach` (nonzero) within a few poll cycles once the GPS has a satellite fix
    * `chronyc tracking`
  - if `GPSD_OPTIONS` was temporarily set to `-n -D5` for debugging via `journalctl -u gpsd`, remove the `-D5` afterward and restart gpsd — no reason to run an unattended field unit at debug log verbosity indefinitely

* install and enable the standaloneRecorder service
  - `cd ${HOME}/Code/AircraftAudioId/standaloneUnit`
  - `sudo cp etc/systemd/system/standaloneRecorder.service /etc/systemd/system/`
  - `sudo systemctl daemon-reload`
  - `sudo systemctl enable standaloneRecorder`
  - `sudo systemctl start standaloneRecorder`
  - `sudo journalctl -u standaloneRecorder -f`
    ==> confirm this waits for/acquires a GPS fix, then starts recording

* install and enable the shutdownButton service
  - `cd ${HOME}/Code/AircraftAudioId/standaloneUnit`
  - `sudo cp etc/systemd/system/shutdownButton.service /etc/systemd/system/`
  - `sudo systemctl daemon-reload`
  - `sudo systemctl enable --now shutdownButton`
  - `sudo journalctl -u shutdownButton -f`
    ==> confirm it logs "Watching gpiochip0 line 9 for button press", then press the button and confirm the unit shuts down cleanly (LED off, then power drops per `POWER_OFF_ON_HALT=1`)

## Workflow

The Standalone system reuses the Local system's `AircraftRecordingSystem` core (flyover trigger/save logic, `RecordingMetadata` schema) unchanged, injecting a local-capture audio stream and a local ADS-B client in place of the Pi's TCP stream and the networked readsb endpoint. This makes the retrieved data directory-compatible with the existing dataset tools with no changes.

### Step 1: GPS Fix (Startup)

**Module:** `src/aircraftAudio/standalone/gps.py` — `GpsdClient`

Blocks until gpsd reports a valid fix (`TPV` message, `mode >= 2`), via gpsd's own TCP JSON protocol (`--gpsdHost`/`--gpsdPort`, default `127.0.0.1:2947`) rather than reading the GPS's serial device directly. This fix's latitude/longitude become `observerLat`/`observerLon` for the entire run — see "Software" above for why there's no runtime re-acquisition.

`--gpsMinSatellites` (default `0`) can additionally require a minimum satellite count from gpsd's `SKY` reports, but defaults off: `SKY` and `TPV` are separate, asynchronously-arriving gpsd message types, so gating on a satellite count that may not be freshly updated at the exact moment a good `TPV` lands **can reject an otherwise-valid fix indefinitely** — confirmed on hardware with a GPS hovering around 3-4 satellites against a default of 4, which stalled `waitForFix()` well past when gpsd already had a good, chrony-confirmed fix. `mode >= 2` is gpsd's own authoritative fix-quality signal; treat `--gpsMinSatellites` as an optional stricter filter, not something to rely on by default.

**Why `GpsdClient` and not a direct serial read:** `gps.py` also provides `GpsClient`, which reads `/dev/serial0` directly via NMEA parsing — but gpsd already holds that device open with an exclusive lock (`TIOCEXCL`) for the chrony PPS/SHM time bridge (see "Set up GPS time discipline" above), so a second process opening it directly fails immediately with `OSError: [Errno 16] Device or resource busy` — confirmed on hardware. `GpsdClient` queries gpsd's already-running fix instead of competing for the raw device; `GpsClient` remains available for standalone use where gpsd isn't already running against the same port.

### Step 2: Local Audio Capture

**Module:** `src/aircraftAudio/record/audioStream/localStream.py` — `LocalAudioStream`

Same 6-method interface `RemoteAudioStream` provides to `AircraftRecordingSystem` (`isStreamHealthy`, `getBuffer`, `getBufferStartTime`, `getClockSkewSecs`, etc.), but captures directly via `sounddevice.InputStream` on the CM4 itself — no TCP layer, since capture and recording run in the same process here. `getClockSkewSecs()` always returns `0.0`: audio and ADS-B polling share one wall clock, so there's no cross-machine skew to estimate.

### Step 3: Local ADS-B Monitoring

**Module:** `src/aircraftAudio/record/adsb/readsb.py` — `ReadsbClient` (unchanged from the Local system)

Polls a `readsb`/`dump1090-fa` process running locally against the FlightAware USB-SDR dongle, serving its own `aircraft.json` on `localhost` (see "Installation" above for the OS-level install). No code differs from the Local system — only the URL changes, from `adsbrx.lan` to `localhost`.

### Step 4: Flyover Detection, Recording, and Health Gating

**Module:** `src/aircraftAudio/standalone/standaloneRecorder.py` — `StandaloneRecorder`

**Script:** `scripts/standaloneRecord.py` (run on the CM4)

```bash
python scripts/standaloneRecord.py \
    --outputDir ./recordings \
    --faaDatabaseDir /path/to/ReleasableAircraft \
    [--readsbUrl http://localhost/data/aircraft.json] \
    [--gpsdHost 127.0.0.1] [--gpsdPort 2947] [--gpsMinSatellites 0] \
    [--ledChip gpiochip0] [--ledLine 11] \
    [--minFreeGb 2] \
    [--nullSampleInterval 210] [--nullSampleDuration 15]
```

`StandaloneRecorder.start()` acquires the GPS fix (Step 1), then constructs `LocalAudioStream`, `ReadsbClient`, a `StorageGuard` (halts saves below `--minFreeGb`), and an offline aircraft-type adapter backed by `FaaDatabase` (see below), and injects all four into `AircraftRecordingSystem` — the same trigger/save/null-sampling logic documented under the Local system's Step 4 applies unchanged. `--faaDatabaseDir` is **required** here (unlike the Local system's `record.py`): without it, aircraft-type lookups would fall back to a live OpenSky HTTPS call, which stalls on every save on this unit's air-gapped network.

A background health-loop thread polls `audioStream.isStreamHealthy()` and `storageGuard.hasSpace()` and drives the status LED (`src/aircraftAudio/standalone/statusLed.py` — `StatusLed`, `gpiod`-controlled) accordingly: solid while both are healthy, blinking otherwise (including during the Step 1 GPS-fix wait).

Sending `SIGUSR1` to the running process writes a session summary snapshot, same as the Local system.

### Step 5: Retrieval and Dataset Construction

Once physically retrieved, the unit's storage contains the same `recordings/audio/<id>.wav` + `recordings/metadata/<id>.json` layout the Local system produces. **No new tooling is needed** — run the existing Local-system scripts directly against the retrieved directory:

```bash
python scripts/inspectDataset.py --recordingsDir /path/to/retrieved/recordings

python scripts/buildDataset.py \
    --recordingsDir /path/to/retrieved/recordings \
    --outputDir ./dataset \
    --faaDatabaseDir /path/to/ReleasableAircraft \
    --maxDistanceKm 15 --dropUnknown --balanceClasses --stratifyPhase
```

See "Installation" above for CM4 OS bring-up (venv, local readsb/dump1090-fa install, GPS-disciplined `gpsd`+`chrony` time sync, the device-tree overlay exposing the status-LED GPIO line, and the `standaloneRecorder` systemd service).
