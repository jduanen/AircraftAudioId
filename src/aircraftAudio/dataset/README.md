# dataset

Offline processing pipeline. Aligns ADS-B state snapshots to audio sample
positions and extracts fixed-length clips for model training.

## Modules

**`align.py`**  
Maps each ADS-B state in a recording's metadata JSON to its position in the WAV
file using `audioStartTime` (Pi-side timestamp of sample 0) and `capturedAt`
(server-side timestamp of each ADS-B poll).

- `alignStates(metadataPath)` — returns all states annotated with
  `timeOffsetSecs`, `sampleIndex`, and `inWindow`
- `alignedWindows(metadataPath, windowSecs)` — returns only in-window states,
  each with `windowStart`/`windowEnd` sample indices

Alignment math: `sampleIndex = (capturedAt - audioStartTime) * sampleRate`

Requires recordings made with the current `recorder.py` (needs `audioStartTime`
in metadata and `capturedAt` on each state). Older recordings are skipped.

**`clipExport.py`**  
Cuts each recording into fixed-length clips centred on each in-window ADS-B
state and writes a `dataset.csv` with one row per clip.

- `buildClipDataset(recordingsDir, outputDir, clipSecs)` — extracts clips and
  writes `<outputDir>/clips/*.wav` + `<outputDir>/dataset.csv`
- `splitByEvent(df, trainFrac)` — splits by `recordingId` (flyover event) so
  all clips from one flyover stay in the same split, preventing data leakage

CSV columns: `filepath`, `recordingId`, `vehicle_types` (JSON list),
`directionClass` (0–7), `velocityKts`, `altitudeFt`, `distanceKm`,
`bearingDeg`, `headingDeg`, `clipOffsetSecs`

## Entry point

```bash
python scripts/buildDataset.py \
    --recordingsDir ./recordings \
    --outputDir ./dataset \
    --clipSecs 5.0 \
    --maxDistanceKm 10.0
```

Produces `dataset/train.csv` and `dataset/val.csv` ready for `training/`.

## Dependencies

`soundfile`, `numpy`, `pandas`.
