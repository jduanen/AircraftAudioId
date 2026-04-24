# augmentation

Audio augmentation pipelines and clip windowing.

## Files

**`audioAug.py`** — `buildAugPipeline(bgNoiseDir)`  
Audiomentations pipeline applied to raw waveforms (CPU, numpy). Includes
Gaussian noise, time stretch, pitch shift, gain, and shift. Most impactful
addition: pass a `bgNoiseDir` folder of ambient recordings (no aircraft) to
enable `AddBackgroundNoise` — this substantially improves robustness to
real-world recording conditions.

**`gpuAug.py`** — `buildGpuAugPipeline()`  
torch-audiomentations pipeline applied to GPU tensor batches. Faster than
`audioAug.py` for large batch sizes. Requires `pip install torch-audiomentations`.

**`waveformAug.py`** — `applyWaveformAug`, `SpecAugment`  
Waveform-level augmentation (noise + gain) using pure PyTorch. `SpecAugment`
applies frequency and time masking after spectrogram conversion — most
effective for AST and transformer-based models.

**`windowing.py`** — `SlidingWindowExtractor`  
Sliding-window clip extraction with configurable window size and hop. Use
to generate denser training samples from long flyover recordings by
extracting overlapping clips rather than just one clip per ADS-B state.
Relevant when the dataset is large enough to benefit from higher clip density.

## Priority order

1. `AddBackgroundNoise` (`audioAug.py`) — highest impact, add first
2. `SpecAugment` (`waveformAug.py`) — good for all model types
3. `SlidingWindowExtractor` (`windowing.py`) — more data from same recordings
4. `gpuAug.py` — only if CPU augmentation is the training bottleneck
