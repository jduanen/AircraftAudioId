#!/usr/bin/env python3
"""
PCF8574 I2C I/O expander reader for the Standalone unit's WiFi gate switch.

The PCF8574's inputs are quasi-bidirectional and open-drain: each pin reads
1 (pulled up internally) unless something external pulls it low. Wire a
switch between the pin and GND to read 0 when closed.
"""

from typing import Callable, Optional

from smbus2 import SMBus


class Pcf8574:
    """
    Args:
        address:    7-bit I2C address (depends on the PCF8574's A0-A2
                    address-pin strapping — confirm with `i2cdetect -y 1`).
        busNum:     Linux I2C bus number (/dev/i2c-N; CM4 default is 1).
        busFactory: Injectable factory `busNum -> SMBus`-like object with a
                    `read_byte(address)` method — defaults to
                    `smbus2.SMBus`. Lets tests replay a fake bus without
                    real hardware.
    """

    def __init__(
        self,
        address: int = 0x20,
        busNum: int = 1,
        busFactory: Optional[Callable[[int], object]] = None,
    ):
        busFactory = busFactory or SMBus
        self.address = address
        self._bus = busFactory(busNum)

    def readInputs(self) -> int:
        """Return all 8 pins as one byte (bit0 = P0 ... bit7 = P7)."""
        return self._bus.read_byte(self.address)

    def readInput(self, bit: int) -> bool:
        """Return the state of a single pin (True = high/not pulled low)."""
        return bool(self.readInputs() & (1 << bit))
