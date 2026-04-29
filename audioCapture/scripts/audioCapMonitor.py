#!/usr/bin/env python3
"""
Health monitor for the AircraftAudio audio capture on a RasPi

Publish HA discovery message at startup

Periodically publishes a JSON MQTT message with:
  cpuTempC, wifiRssiDbm, loadAvg{1,5,15}m, uptimeSecs, captureRunning

Usage:
    python scripts/piMonitor.py --broker <mqtt-host>
                                [--port 1883]
                                [--interval 30]
                                [--iface wlan0]
                                [--user <user>] [--password <password>]

Run this on the Pi Zero W (running Trixie) with the audio capture HW
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt


STATE_TOPIC = "sensors/AudioCapture/state"

_DEVICE = {
    "identifiers": ["audio_capture"],
    "name": "AudioCapture (Pi Zero W)",
}
_ORIGIN = {"name": "audioCapMonitor.py"}

DISCOVERY_CONFIGS = [
    (
        "homeassistant/binary_sensor/audiocap_running/config",
        {
            "name": "Capture Running",
            "unique_id": "audiocap_running",
            "state_topic": STATE_TOPIC,
            "value_template": "{{ 'ON' if value_json.captureRunning else 'OFF' }}",
            "device": _DEVICE,
            "origin": _ORIGIN,
        },
    ),
    (
        "homeassistant/sensor/audiocap_cpu_temp/config",
        {
            "name": "CPU Temperature",
            "unique_id": "audiocap_cpu_temp",
            "state_topic": STATE_TOPIC,
            "value_template": "{{ value_json.cpuTempC }}",
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "state_class": "measurement",
            "device": _DEVICE,
            "origin": _ORIGIN,
        },
    ),
    (
        "homeassistant/sensor/audiocap_wifi_rssi/config",
        {
            "name": "WiFi RSSI",
            "unique_id": "audiocap_wifi_rssi",
            "state_topic": STATE_TOPIC,
            "value_template": "{{ value_json.wifiRssiDbm }}",
            "unit_of_measurement": "dBm",
            "device_class": "signal_strength",
            "state_class": "measurement",
            "device": _DEVICE,
            "origin": _ORIGIN,
        },
    ),
    (
        "homeassistant/sensor/audiocap_load_1m/config",
        {
            "name": "Load Average 1m",
            "unique_id": "audiocap_load_1m",
            "state_topic": STATE_TOPIC,
            "value_template": "{{ value_json.loadAvg1m }}",
            "state_class": "measurement",
            "device": _DEVICE,
            "origin": _ORIGIN,
        },
    ),
    (
        "homeassistant/sensor/audiocap_load_5m/config",
        {
            "name": "Load Average 5m",
            "unique_id": "audiocap_load_5m",
            "state_topic": STATE_TOPIC,
            "value_template": "{{ value_json.loadAvg5m }}",
            "state_class": "measurement",
            "device": _DEVICE,
            "origin": _ORIGIN,
        },
    ),
    (
        "homeassistant/sensor/audiocap_load_15m/config",
        {
            "name": "Load Average 15m",
            "unique_id": "audiocap_load_15m",
            "state_topic": STATE_TOPIC,
            "value_template": "{{ value_json.loadAvg15m }}",
            "state_class": "measurement",
            "device": _DEVICE,
            "origin": _ORIGIN,
        },
    ),
    (
        "homeassistant/sensor/audiocap_uptime/config",
        {
            "name": "Uptime",
            "unique_id": "audiocap_uptime",
            "state_topic": STATE_TOPIC,
            "value_template": "{{ value_json.uptimeSecs }}",
            "unit_of_measurement": "s",
            "device_class": "duration",
            "state_class": "total_increasing",
            "device": _DEVICE,
            "origin": _ORIGIN,
        },
    ),
]

LOG_LEVEL = logging.WARNING

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------

def cpuTempC() -> float | None:
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return round(int(raw) / 1000.0, 1)
    except Exception:
        return None


def wifiRssiDbm(iface: str) -> int | None:
    # Primary: iw dev <iface> link  →  "signal: -62 dBm"
    try:
        result = subprocess.run(
            ["iw", "dev", iface, "link"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            m = re.search(r"signal:\s*(-?\d+)\s*dBm", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass

    # Fallback: /proc/net/wireless (signal column, index 3, may have trailing ".")
    try:
        for line in Path("/proc/net/wireless").read_text().splitlines():
            if line.strip().startswith(iface):
                parts = line.split()
                return int(float(parts[3].rstrip(".")))
    except Exception:
        pass

    return None


def loadAvg() -> tuple[float, float, float]:
    return os.getloadavg()


def uptimeSecs() -> int:
    raw = Path("/proc/uptime").read_text().split()[0]
    return int(float(raw))


def isCaptureRunning() -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "capture.py"],
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def collectMetrics(iface: str) -> dict:
    load1, load5, load15 = loadAvg()
    return {
        "timestamp": round(time.time(), 3),
        "cpuTempC": cpuTempC(),
        "wifiRssiDbm": wifiRssiDbm(iface),
        "loadAvg1m": round(load1, 3),
        "loadAvg5m": round(load5, 3),
        "loadAvg15m": round(load15, 3),
        "uptimeSecs": uptimeSecs(),
        "captureRunning": isCaptureRunning(),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def buildArgParser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish Pi health metrics via MQTT")
    p.add_argument("--broker",   required=True,              help="MQTT broker hostname or IP")
    p.add_argument("--port",     type=int, default=1883,     help="MQTT broker port (default: 1883)")
    p.add_argument("--interval", type=int, default=30,       help="Publish interval in seconds (default: 30)")
    p.add_argument("--iface",    default="wlan0",            help="WiFi interface name (default: wlan0)")
    p.add_argument("--user",     default=None,               help="MQTT username")
    p.add_argument("--password", default=None,               help="MQTT password")
    return p


def run(args: argparse.Namespace) -> None:
    client = mqtt.Client(client_id="trixie-health-monitor")

    if args.user:
        client.username_pw_set(args.user, args.password)

    def _on_connect(c, userdata, flags, rc):
        if rc == 0:
            log.info("Connected to MQTT broker %s:%d", args.broker, args.port)
        else:
            log.warning("MQTT connect failed, rc=%d — will retry", rc)

    def _on_disconnect(c, userdata, rc):
        if rc != 0:
            log.warning("MQTT disconnected unexpectedly (rc=%d), reconnecting…", rc)

    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect

    client.reconnect_delay_set(min_delay=5, max_delay=60)

    log.info("Connecting to %s:%d, state_topic=%s, interval=%ds",
             args.broker, args.port, STATE_TOPIC, args.interval)
    try:
        client.connect(args.broker, args.port, keepalive=60)
    except Exception as exc:
        log.error("Initial MQTT connect failed: %s — will keep retrying", exc)

    client.loop_start()

    for disc_topic, disc_payload in DISCOVERY_CONFIGS:
        payload = json.dumps(disc_payload)
        result = client.publish(disc_topic, payload, qos=1, retain=True)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            log.info("Discovery published: %s", disc_topic)
        else:
            log.warning("Discovery publish failed (rc=%d) for %s — broker may be unreachable",
                        result.rc, disc_topic)
            sys.exit(1)

    while True:
        metrics = collectMetrics(args.iface)
        payload = json.dumps(metrics)
        result = client.publish(STATE_TOPIC, payload, qos=1, retain=False)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            log.info("Topic: %s; Published: %s", STATE_TOPIC, payload)
        else:
            log.warning("Publish failed (rc=%d), broker may be unreachable", result.rc)
        time.sleep(args.interval)


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    run(args)
