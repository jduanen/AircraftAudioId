#!/usr/bin/env python3
"""
Health monitor for the Trixie audio capture Pi.

Periodically publishes a JSON MQTT message with:
  cpuTempC, wifiRssiDbm, loadAvg{1,5,15}m, uptimeSecs, captureRunning

Usage:
    python scripts/piMonitor.py --broker <mqtt-host>
                                [--port 1883]
                                [--topic aircraftaudioid/trixie/health]
                                [--interval 30]
                                [--iface wlan0]
                                [--user <user>] [--password <password>]
                                [--no-retain]

Run this on the Pi Zero W (Trixie).
"""

import argparse
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import paho.mqtt.client as mqtt


logging.basicConfig(
    level=logging.INFO,
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
    p.add_argument("--topic",    default="aircraftaudioid/trixie/health",
                   help="MQTT topic to publish to")
    p.add_argument("--interval", type=int, default=30,       help="Publish interval in seconds (default: 30)")
    p.add_argument("--iface",    default="wlan0",            help="WiFi interface name (default: wlan0)")
    p.add_argument("--user",     default=None,               help="MQTT username")
    p.add_argument("--password", default=None,               help="MQTT password")
    p.add_argument("--no-retain", dest="retain", action="store_false", default=True,
                   help="Disable MQTT retained flag")
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

    log.info("Connecting to %s:%d, topic=%s, interval=%ds",
             args.broker, args.port, args.topic, args.interval)
    try:
        client.connect(args.broker, args.port, keepalive=60)
    except Exception as exc:
        log.error("Initial MQTT connect failed: %s — will keep retrying", exc)

    client.loop_start()

    while True:
        metrics = collectMetrics(args.iface)
        payload = json.dumps(metrics)
        result = client.publish(args.topic, payload, qos=1, retain=args.retain)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            log.info("Published: %s", payload)
        else:
            log.warning("Publish failed (rc=%d), broker may be unreachable", result.rc)
        time.sleep(args.interval)


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    run(args)
