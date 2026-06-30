# Evaluation Guide

## Training Monitoring (TensorBoard)

### Key Metrics

| Metric | What to watch |
|---|---|
| Training Loss | Should decrease steadily. Plateau early = optimization stuck |
| Validation Loss | Should track training loss. Rising while train loss falls = overfitting |
| val_f1 (macro) | Primary early-stopping signal. Flat for 10 epochs → EarlyStopping fires |
| Learning Rate | Explains sudden behavior changes (e.g., cosine schedule decay, backbone unfreeze) |

### Reading the Loss Curves

| Pattern | Diagnosis |
|---|---|
| Both losses fall together | Training is healthy |
| Train loss falls, val loss rises | Overfitting. Model is memorizing the training set |
| Both losses high or barely moving | Underfitting, bad learning rate, or pipeline bug |
| Metrics noisy batch-to-batch | Look at epoch averages, not individual steps |
| Train/val loss ratio > 10× | Severe overfitting. Consider freezing backbone, adding dropout, or reducing data imbalance |

### TensorBoard Dashboards

- **Scalars**: loss, val_f1, learning rate; compare multiple runs side-by-side
- **Graphs**: verify network structure matches intent; check early before wasting training time
- **Histograms**: watch for weight/activation collapse or saturation
- **Profiler**: use when training is unexpectedly slow or I/O-bound

### Backbone Freezing Notes

- `--freezeBackbone` restricts training to layer4 + classifier head (~2M vs ~11M parameters). Strongest single lever for small datasets.
- `--unfreezeEpoch N`: at epoch N the full backbone resumes training. The cosine LR schedule has decayed by then, so fine-tuning is gentle. Typical value: 25–35. Too early (epoch 20) resumes overfitting before the classifier head is stable. Too late (epoch 40+) with a very low LR underfits.
- If train/val loss ratio spikes after unfreeze, the dataset is not large enough for full fine-tuning. Consider removing `--unfreezeEpoch` entirely.

---

## Model Evaluation Metrics

### F1 (per-class)

Harmonic mean of precision and recall. Range [0, 1]; 1 = perfect

- Useful when classes are imbalanced. Accuracy alone can look good while minority classes fail
- Penalises imbalance: high precision + low recall still gives a low F1

### mAP (mean Average Precision)

Summarises the precision-recall trade-off across all classes and confidence thresholds. Range [0, 1]; higher is better.

- Integrates area under the precision-recall curve per class, then averages
- Useful for multi-label tasks where a single threshold doesn't capture the full trade-off

### Macro-F1

Unweighted average of per-class F1 scores. Range [0, 1]; higher is better

- Each class counts equally regardless of clip count. Minority classes are not swamped by dominant ones
- Primary early-stopping metric (`val_f1` in training logs)
- **Class with zero val clips**: if a class passes `--minClipsPerClass` but all its recordings land in train during the session split, torchmetrics returns F1=0 for that class and drags macro-F1 down. The training log prints which class has 0 val positives; exclude it or collect more recordings

### mAP vs Macro-F1

| Situation | Interpretation |
|---|---|
| mAP higher, Macro-F1 lower | Better at common classes; worse on rare ones |
| Macro-F1 higher, mAP lower | More balanced across classes; slightly lower detection quality |

**The per-class AP table is more informative than the scalar val_f1**, especially when rare classes drag the macro average. A checkpoint with lower Macro-F1 may be preferable if it handles the classes you care about most.

---

## Input Data Quality

Two evaluation tiers are available via `scripts/evalClipQuality.py`.

### Fast Path (CSV only, no audio reads)

Uses the pre-computed `clipRms` column and ADS-B metadata already in the dataset CSV.

| Output | Description |
|---|---|
| Per-class table | Mean/median/P10 RMS dBFS, % clips below quality threshold, avg distance and altitude |
| RMS histogram | Distribution of signal levels for the selected class |
| Phase breakdown | RMS stats split by approach / closest / departure |

### Deep Path (`--deepAnalysis`, reads WAV files)

Computes seven additional metrics per clip using soundfile + librosa.

| Metric | What it measures |
|---|---|
| `silenceFrac` | Fraction of samples with |x| < 0.005 |
| `clippingFrac` | Fraction of samples with |x| > 0.99 (ADC saturation) |
| `frameEnergyStd` | Std dev of per-0.1s-frame RMS across the clip |
| `edgeCenterRatio` | RMS of first+last 1 s ÷ RMS of middle 3 s |
| `spectralFlatness` | 0 = tonal/structured, 1 = broadband noise |
| `spectralCentroid` | Frequency center of mass in Hz |
| `lowFreqRatio` | Fraction of spectral energy below 200 Hz |

---

## dBFS Threshold Guidance

The script uses **dBFS** (decibels full-scale, relative to the ADC clipping point), not dBm. A clip is useful only when the aircraft signal is clearly above the ambient noise floor (aim for 10-15 dB headroom).

| Range | Assessment |
|---|---|
| −20 to −35 dBFS | Loud, close aircraft; ideal |
| −35 to −50 dBFS | Moderate distance; usually usable |
| −50 to −60 dBFS | Distant/quiet; borderline, may be mostly noise |
| below −60 dBFS | Almost certainly inaudible over ambient noise |

**To calibrate for your setup**: run the fast path (`--datasetCsv` only) and inspect the P10 column. Set `--rmsThresholdDb` just above the level where clips stop sounding like aircraft and start sounding like wind or silence. Listening to a handful of clips at −50, −55, and −60 dBFS takes only a few minutes.

---

## Per-Metric Interpretation

### Silence fraction (`silenceFrac`)

| Value | Meaning |
|---|---|
| < 10% | Normal; continuous audio |
| 10–30% | Some gaps; monitor |
| > 30% | Large gaps; aircraft may have been brief or already gone when clip was cut |

### Clipping fraction (`clippingFrac`)

| Value | Meaning |
|---|---|
| ≈ 0% | Normal |
| > 0.1% | ADC saturation beginning; mic gain too high or aircraft too close |
| > 1% | Serious saturation; spectral detail destroyed; these clips confuse the model |

### Frame energy std dev (`frameEnergyStd`)

- **High**: energy rises and falls across the clip; distinct aircraft event is present
- **Near 0**: temporally flat; ambient noise with no flyover event, regardless of RMS level
- Interpret relatively across clips; absolute scale depends on mic gain

### Edge/center energy ratio (`edgeCenterRatio`)

| Value | Meaning |
|---|---|
| < 1.0 | Center louder than edges; correct flyover shape |
| ≈ 1.0 | Flat energy; aircraft at constant distance throughout |
| > 1.5 | Inverted envelope; clip misaligned or capturing only the tail/head of a flyover |

### Spectral flatness (`spectralFlatness`)

| Value | Meaning |
|---|---|
| < 0.2 | Tonal, structured signal; aircraft engines produce harmonic content |
| 0.3–0.6 | Mixed; some aircraft, some noise |
| > 0.6 | Broadband noise dominant; clip likely does not sound like an aircraft |

### Spectral centroid (`spectralCentroid`)

| Value | Meaning |
|---|---|
| 600–2000 Hz | Typical for piston aircraft (propeller harmonics) |
| 300–1000 Hz | Typical for jets (engine/fan noise) |
| < 300 Hz | Wind or handling rumble dominant |
| > 4000 Hz | Unusual; possibly interference or very noisy mic |

### Low-frequency energy ratio (`lowFreqRatio`)

| Value | Meaning |
|---|---|
| < 25% | Normal for aircraft audio |
| 25–50% | Elevated wind content; monitor |
| > 50% | Sub-200 Hz energy dominant; almost always wind noise |

---

## Common Failure Patterns

Read metrics together, not in isolation.

| Pattern | Likely cause |
|---|---|
| Low RMS + high flatness + low frame-energy std | Ambient noise; no aircraft event captured |
| High silence % + low frame-energy std | Aircraft too brief or already gone when clip was cut |
| Low RMS + high LF% + low centroid | Wind noise dominating the clip |
| Good RMS + high clipping % | Mic gain too high or aircraft too close; reduce gain |
| ECR > 1.5 + good RMS | Clip misaligned; aircraft was loudest at the edges of the window |
| Low RMS + low flatness + centroid 500–1500 Hz | Distant but real aircraft, but may still be trainable |

**Filtering recommendation**: discard clips with *at least two* flags simultaneously (e.g. RMS < −55 dBFS AND flatness > 0.5 AND LF% > 50%). Single-metric outliers are often just distant aircraft and may still contribute useful training signal.

---

## Composite Quality Score

Used by `evalClipQuality.py --keepBestN --deepAnalysis` and `buildQualityDataset*.py --deepAnalysis`. Each metric is normalised to [0, 1] (1 = best) and combined with the following weights:

| Weight | Metric | Direction |
|---|---|---|
| 0.35 | RMS dBFS | higher = louder aircraft |
| 0.15 | Silence fraction | lower = fewer silent gaps |
| 0.10 | Clipping fraction | lower = no ADC saturation |
| 0.10 | Spectral flatness | lower = more tonal / structured |
| 0.10 | Edge/center ratio | lower = energy centred on flyover apex |
| 0.10 | Low-freq energy ratio | lower = less wind/rumble |
| 0.10 | Spectral centroid | penalise < 300 Hz (wind) and > 4000 Hz (noise) |

---

## Class Imbalance Pitfalls

- **Zero val clips**: if a class has few recordings and all end up in train during the session-based split, torchmetrics returns F1=0 for that class. Use `--minClipsPerClass` to exclude classes below a threshold from training entirely rather than letting them poison macro-F1.
- **"No positive samples" warning**: the smoking gun for this situation. Check which class the warning names, then either collect more recordings or raise `--minClipsPerClass`.
- **Capping dominant classes** (`--maxPerClass`): reduces total training signal, which can lower macro-F1 even while improving balance. The `pos_weight` in BCEWithLogitsLoss already compensates for imbalance at the loss level — try that first before capping.
- **`--stratifyPhase`**: evens out approach/departure within each class without reducing total clip count. Prefer this over `--balanceClasses` when the dataset is small.
