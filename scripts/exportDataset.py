#!/usr/bin/env python3
"""
Export recorded sessions to a training CSV for use with toolchain.py.

Usage:
    python scripts/exportDataset.py --recordingsDir ./recordings [--output dataset.csv]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aircraftAudio.export import createTrainingDataset, buildArgParser


if __name__ == "__main__":
    args = buildArgParser().parse_args()
    createTrainingDataset(args.recordingsDir, args.output)
