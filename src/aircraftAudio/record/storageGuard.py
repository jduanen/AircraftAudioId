#!/usr/bin/env python3
"""
Disk-space guard used to halt new recordings before storage fills up.

Used by the Standalone Data Collection unit, which runs unattended for
days-to-months with no operator present to notice a full disk. Recordings
are never auto-deleted to make room — once free space drops below the
threshold, AircraftRecordingSystem stops saving new recordings until the
unit is retrieved.
"""

import shutil
from pathlib import Path


class StorageGuard:
    """
    Checks free disk space against a minimum threshold.

    Args:
        path:         Any path on the filesystem to check (typically outputDir).
        minFreeBytes: Minimum free bytes required to permit new recordings.
    """

    def __init__(self, path: str | Path, minFreeBytes: int):
        self.path = Path(path)
        self.minFreeBytes = minFreeBytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def freeBytes(self) -> int:
        return shutil.disk_usage(self.path).free

    def hasSpace(self) -> bool:
        return self.freeBytes() >= self.minFreeBytes
