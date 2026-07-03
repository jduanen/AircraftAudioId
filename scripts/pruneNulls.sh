#!/bin/bash

# Preview: how many nulls you have
find recordings/metadata -name "*_null.json" | wc -l

# Keep 300 random nulls, delete the rest (run from project root)
python3 - <<'EOF'
import random, os
from pathlib import Path

keep = 300
nulls = list(Path("recordings/metadata").glob("*_null.json"))
random.shuffle(nulls)
to_delete = nulls[keep:]

for meta in to_delete:
    wav = Path("recordings/audio") / meta.name.replace(".json", ".wav")
    meta.unlink()
    if wav.exists():
        wav.unlink()

print(f"Deleted {len(to_delete)} null recordings, kept {min(keep, len(nulls))}.")
EOF
