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

The enslosure is a cast aluminium case that serves as a heat-sink for the CM4 module (and, if possible, the USB-SDR dongle).
The three (ADS-B 1090MHz, WiFi, and GPS) antennas are mounted onto the enclosure in such a manner as to resist water infiltration.

## Software

The software for this unit builds on that developed for the Local Data Collection system.

The (software-controlled) illuminated power switch is the only indication that the unit provides that it is functioning correctly. It can signal correct operation by staying on continuously, and it can indicate something is wrong by blinking the switch's LED. If the LED is off, this indicates that the system either has no power, or has a hard failure and the sofware isn't running. In practice: solid on = healthy (GPS fix acquired, audio capturing, storage OK); blinking = starting up (still waiting for a GPS fix) or a software-detected problem (most commonly low storage); off = unpowered or crashed.

It is expected that, in order to gather sufficient data, this unit will remain in a location for days to months at a time. This means that the unit must contain sufficient storage space to contain all of the samples and metadata generated before the unit is retreived and the information dumped. Once free storage drops below a threshold, new recordings are halted (never auto-deleted) and the LED switches to its error-blink pattern, signaling that the unit needs retrieval.

Because the unit is moved by power-cycling it (there's no way to relocate a sealed, running unit), it reads its position from the on-board GPS receiver exactly once at startup — blocking until a fix is acquired — and uses that fixed position for the entire run. There is no runtime location-change detection or mid-run restart; a fresh fix is simply acquired the next time the unit boots.

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
* the GPU module is a GT-UZ, and its pinout is:
  - VCC: power input 3.3V
  - GND: ground
  - TXD: GPS serial data, defaults to NMEA and UBX responses
  - RXD: (UBX/NMEA format) configuration commands from the host
  - PPS: one pulse-per-second time reference
* see "Set up GPS time discipline" below for wiring `/dev/pps0` into chrony via gpsd

#### Enable the micro-SD on the SPI Bus

* describe the SPI SD socket in the device tree in the file 'cm4-sdspi.dts' (audioCapture/cm4-sdspi.dts)
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
* **confirmed working:** `mmc_spi` binds to `spi1.0` and registers an MMC host with `cm4-sdspi.dts` as shipped, at `spi-max-frequency = <400000>` (400 kHz)
  - the originally-specified 12 MHz produced `mmc_spi spi1.0: no support for card's volts` on every boot — not a `voltage-ranges` misconfiguration, but the SD spec's power-up handshake (CMD0/CMD8/OCR query) needing to run at <=400 kHz; `mmc_spi` doesn't auto-negotiate a slower probe clock, it uses `spi-max-frequency` throughout, so 12 MHz corrupted that initial exchange
  - if higher throughput is needed later, this can be stepped up from 400 kHz and retested, but hand-wired JST-GH cabling is unlikely to hold up much past the low single-digit MHz range

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
* make `standaloneRecorder.service` wait for the mount rather than starting and writing straight into the underlying eMMC root filesystem if the card isn't mounted yet — already added: `RequiresMountsFor=/home/jdn/Code/AircraftAudioId/recordings` in the service's `[Unit]` section

#### Install the Application Software on the Standalone Device

* Create venv (first time only)
  - `mkdir -p Code/AircraftAudioId/standaloneUnit`
  - `cd Code/AircraftAudioId/standaloneUnit`
  - `python3 -m venv venv`

* Activate venv and install packages
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`

* Sync selected parts of host repo to the device
  - **`../scripts/syncToStandalone.sh`**: run on the server, not the CM4
    * `bash scripts/syncToStandalone.sh <standalone-hostname-or-ip> [--skipFaaDb]`
    * syncs `src/aircraftAudio/`, `scripts/standaloneRecord.py`, `standaloneUnit/`, `audioCapture/cm4-sdspi.dts`, and `data/ReleasableAircraft/` (skip with `--skipFaaDb` once already on the device — it's ~500 MB on first sync)
    * does not create the venv (see Installation steps above) or restart `standaloneRecorder.service`

* Set up local ADS-B capture (readsb/dump1090-fa against the FlightAware USB-SDR dongle)
  - `sudo apt update`
  - install readsb (or dump1090-fa -- either serves an aircraft.json endpoint)
    * `sudo apt install readsb`
  - configure it to read from the USB-SDR dongle and serve JSON on localhost
    * edit `/etc/default/readsb` (package-dependent; confirm during bring-up):
      - enable `--net`
      - confirm the served aircraft.json path (commonly `/data/aircraft.json`)
      - for a plain readsb install, or `/tar1090/data/aircraft.json` if tar1090 is also installed for a local web UI
        * whichever it is, it must match `--readsbUrl` passed to `scripts/standaloneRecord.py` (`standaloneRecorder.service`'s default is `http://localhost/data/aircraft.json` -- update it to match once confirmed)
  - enable and start service
    * `sudo systemctl enable --now readsb`
  - verify the service is running correctly
    * `curl http://localhost/data/aircraft.json`

* Set up GPS time discipline (gpsd + chrony, since this unit has no internet for NTP)
  - the GPS receiver is wired to UART0/1 (for wiring/routing reasons, not USART4 — no conflict with the LED/PPS pins, which are on a separate peripheral either way)
  - `enable_uart=1` in `/boot/firmware/config.txt` is enough to bring up this UART — no additional `dtoverlay=uartN` needed, since it's the SoC's primary UART
  - use `/dev/serial0` rather than a specific `ttyAMA*`/`ttyS0` name: the firmware/udev symlink `/dev/serial0` always points at whichever device is actually the primary UART, which otherwise depends on Bluetooth presence/config
  - `sudo apt install gpsd chrony`
  - point gpsd at the GPS receiver's serial device
    * `sudo ex /etc/default/gpsd`
      - `DEVICES="/dev/serial0"`
    * `sudo systemctl enable --now gpsd`
  - verify raw NMEA output before wiring in GpsClient: `cat /dev/serial0`
  - if the PPS overlay is enabled (see "Enable GPS PPS on CTS4" above), gpsd auto-detects `/dev/pps0` alongside the serial NMEA device and exposes a second, PPS-corrected SHM segment (SHM(1))
    * this is in addition to the coarse NMEA-only one (SHM(0)); no extra gpsd config needed
  - discipline chrony from gpsd's shared-memory (SHM) time reference
    * `sudo systemctl stop systemd-timesyncd`
    * `sudo systemctl disable systemd-timesyncd`
    * `sudo ex /etc/chrony/chrony.conf`
      - comment out default network pools (no internet on this unit)
      - add: `refclock SHM 0 offset 0.0 delay 0.2 refid GPS` (coarse, from NMEA)
      - if the PPS overlay is enabled, also add: `refclock SHM 1 offset 0.0 refid PPS precision 1e-7` (precise, PPS-disciplined)
    * `sudo systemctl restart chronyd`
  - check status
    * `chronyc sources`
    * `chronyc tracking`

* install and enable the standaloneRecorder service
  - `cd ${HOME}/Code/AircraftAudioId/standaloneUnit`
  - `sudo cp etc/systemd/system/standaloneRecorder.service /etc/systemd/system/`
  - `sudo systemctl daemon-reload`
  - `sudo systemctl enable standaloneRecorder`
  - `sudo systemctl start standaloneRecorder`
  - `sudo journalctl -u standaloneRecorder -f`
    ==> confirm this waits for/acquires a GPS fix, then starts recording

## Workflow

The Standalone system reuses the Local system's `AircraftRecordingSystem` core (flyover trigger/save logic, `RecordingMetadata` schema) unchanged, injecting a local-capture audio stream and a local ADS-B client in place of the Pi's TCP stream and the networked readsb endpoint. This makes the retrieved data directory-compatible with the existing dataset tools with no changes.

### Step 1: GPS Fix (Startup)

**Module:** `src/aircraftAudio/standalone/gps.py` — `GpsClient`

Reads NMEA sentences (GGA/RMC) off the GPS receiver's serial UART and blocks until a valid fix with at least `--gpsMinSatellites` satellites is obtained. This fix's latitude/longitude become `observerLat`/`observerLon` for the entire run — see "Software" above for why there's no runtime re-acquisition.

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
    [--gpsPort /dev/serial0] [--gpsMinSatellites 4] \
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
