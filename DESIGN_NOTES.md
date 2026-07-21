* Goal
  - determine to what degree it is possible to reliably classify information about aircraft based on sound signatures

* Progression
  - start with just aircraft type (e.g., propulsion: jet/turbine/piston, number of engines: 1/2/4, wing type: rotary/fixed, etc.)
  ==> maybe skip this and go straight to the multiple case?
  - start with just a single aicraft in audio range, then work on classifying multiple simultaneously audible aircraft
  - attempt to determine additional attributes of aircraft (e.g., engine/airframe model, distance, direction, etc.)

* Characteristics of the Data
  - ground vehicle sounds have important characteristics in different ranges:
    * Engine rumble: 50-500 Hz (need good low-freq resolution)
    * Tire noise: 500-2000 Hz
    * Exhaust: varies widely by vehicle type
    * Wind/aero noise: broadband, speed-dependent
  - a higher n_fft gives better frequency resolution for separating these
    --> answered 2026-07-03, see Experiment Log below: aircraft flyover energy is
        almost entirely <200 Hz; old config (n_fft=1024, fmax=22050) only resolves
        it into 5-6 bins. Switched to n_fft=2048, fmax=8000.
  - multi-task loss balancing still needs tuning
    * see multiTaskLoss.py
  - segmentation/windowing strategy still matters a lot
    * i.e., how you cut up audio recordings into model-sized clips still matters
    * see windowing.py
  - evaluation/test data set design still matters a lot
    * with large dataset need to be much more careful about data leakage
      - random split is bad
        * clips from same vehicle pass-by event ends up in both train and test sets
      - split by recording session or event ID so that all clips from one pass-by stay together
    * easier to overlook leakage with large datasets
  - handling multi-vehicle audio samples
    * don't exclude them, overlapping sounds occur commonly in the real world
    * this changes the task definition
      - single vehicle:
        * "this clip contains a vehicle of this type" 
        * multi-class classification
      - multiple vehicles:
        * "this clip contains this type of vehicle and this type of vehicle"
        * multi-label classification
    * multi-vehicle task changes several things:
      - output layer activation
      - loss function
      - labels format
      - evaluation matrix
      - how the model reasons about the problem
    * need to restructure the model to handle multi-label classification
      - use sigmoid and not softmax in the output layer
        * see multiLabel.py
      - use BCE as loss function, instead of Cross-Entropy
        * see computeLoss.py
      - change label format
        * see encodeLabels.py
    * determining direction and speed with multiple vehicles is a hard problem
      - with one vehicle, direction and speed have clear targets
      - need to decide how to handle multiple vehicle case
        * Approach A: set prediction (per-vehicle output)
          - predict a fixed max number of vehicle "slots", each with type, direction, and speed
            * see setPredict.py
          - training requires Hungarian Matching (like DETR) to assign groud truth vehicles to prediction slots optimally
            * see hungarianLoss.py
        * Approach B: just predict what you can
          - pragmatic approach -- start with this
            * see pragmaticMultiVehicle.py

* Data Acquisition
  - mic placement
    * outdoors with clear sky view
    * away from buildings (cause echos/reflections)
    * windscreen to reduce noise
    * 2-3 meters above ground if possible
  - best location: ~5-15km from airport
  - peak times: 6-9am, 5-8pm weekdays
  - avoid windy days, prefer dry weather
    * need some days with cloud cover, thermal inversion layers, etc.
  - record several hours to get diverse samples
    * record day and night, over the course of a year
  - filter out noisy samples
    * build a tool to detect noisy samples and remove them from the dataset
  - ensure that the audio capture path doesn't introduce electrical noise into the sound signals

* Training/Testing Dataset
  - capture audio signals and use ADS-B telemetry data as ground truth
    * use either Audio Moth or Raspberry Pi Zero 2W with single microphone to capture audio
  - come up with reliable means of synchronizing audio sample capture times with ADS-B telementry data
  - build tool to get audio clips with associated telemetry data for training/testing
  - come up with format for training data and make tool generate this data
    * first, gather enough data to get sufficient examples of each type of aircraft in isolation, across different times of day and weather conditions
      - then gather sufficient data with multiple aircraft
      - later, gather sufficient data with different aircraft attributes (e.g., engine/aircraft model, distance, direction, altitude, ascending/decending rates, etc.)
  - do good design of train/test subsets
  - work hard to get a large and varied dataset
    * don't have to depend on a pre-trained model, can use a simplier model architecture
    * data can regularize the model so I can run lighter regularization and train longer
    * make sure that even rare classes have enough samples to train on
      - what matters is the ratios among classes, not just the absolute numbers
    * reduce dependency on data augmentation
      - still helps to augment data, but get marginal (not essential) gains from augmentation
    * audio preprocessing is critical
      - no amount of data makes up for bad spectrogram configuration
    * mel spectrogram parameters define the information ceiling of the model
      - these choices matter enormously, even with large data sets
  - experiment with different configurations
    * e.g.,
'''
configs_to_try = [
    {"n_mels": 64,  "n_fft": 512,  "hop_length": 256},  # Higher time resolution
    {"n_mels": 128, "n_fft": 1024, "hop_length": 512},  # Balanced
    {"n_mels": 256, "n_fft": 2048, "hop_length": 512},  # Higher freq resolution
]
'''
  - priorities with large dataset
    1) Audio Preprocessing Quality
    2) Data pipeline speed (bottlenecked by I/O and GPU utilization)
    3) Train/test split integrity
    4) Model capacity (go bigger)
    5) multi-task loss balancing
  - Decision Framework: what % of dataset has multiple vehicles
    * <20%
      - use Pragmatic Approach
        * train on everything and Direction/speed on single vehicle only
    * 20-80%
      - use Pragmatic Approach but train vehicle type on all the data
      - consider using set-based for direction/speed
    * \>80%
      - set-based prediction (Approach A) or Sound source separation as pre-processing step

* Spectrogram Config Is the Highest-Leverage Decision
  - the current pipeline computes spectrograms inside the model (resNetCNN.py, customCNN.py, etc.)
    * this makes the config a model hyperparameter, which means:
      - To change n_mels or n_fft, you reinstantiate the model
      - You can't precompute spectrograms (pretrained/precompute.py) without code changes
      - You can't run fast spectrogram sweep experiments
  - recommended changes:
    * move spectrogram computation into the Dataset before Phase 2 begins
    * use pretrained/precompute.py to precompute to .pt tensors
    * treat (n_mels, n_fft, hop_length) as an outer hyperparameter loop above
 training

* selecting sample size is another critical decision
  - generally, 2sec for sharp events, 4sec for GP classification, and 8-10sec for slowly evolving sounds
  - fix sample length, train once, then sweep window size
    * try 1, 2, 4, and 8sec samples and compare validation accuracy
    * pick smallest one that preserves accuracy
  - initial code uses 10sec audio clips

* Open-source tools that could be useful for this project
  - Audio loading and preprocessing
    * torchaudio (recommended)
      - GPU-accelerated, use for trianing pipelines
      - install with: 'pip install torchaudio'
      - see torchAudio.py
    * librosa (alternative)
      - richer audio analysis features, great for exploration
      - install: 'pip install librosa'
      - see libRosa.py
  - Data Augmentation
    * audiomentations (highly recommended)
      - install: 'pip install audiomentations'
        * see audioAug.py
      - killer feature for this application: add background noise from folder
        * see bgAugNoise.py
    * torch-audiomentations (GPU-accelerated)
      - install: 'pip install torch-audiomentations'
      - see torchAug.py
  - Pre-built Audio Classification Models
    * PyTorch Audio Models (PANNs)
      - install: 'pip install panns-inference'
        * see panns.py
      - use case: extract pretrained embeddings, then train a small classifier on top
    * Transformers (HuggingFace)
      - install: 'pip install transformers datasets'
      - see trans.py
    * HEAR Benchmark Models
      - install: 'pip install hearbaseline'
      - see hear.py
  - Multi-Label Classification Helpers
    * scikit-learn (Metrics)
      - install: 'pip install scikit-learn'
      - see skl.py
    * pytorch-lightning (Training Framework)
      - dramatically simplifies the training loop
      - install: 'pip install pytorch-lightning torchmetrics'
      - see lightning.py
  - Dataset Management
    * datasets (HuggingFace)
      - handles audio loading, caching, and pre-processing elegantly
      - install: 'pip install datasets'
      - see hfDatasets.py
    * webdataset (for very large datasets)
      - optimized for streaming large datasets from disk or cloud
      - install: 'pip install webdataset'
      - see webDataset.py
  - Experiment Tracking
    * Weights & Biases
      - install: 'pip install wandb'
      - see wab.py
    * MLflow (self-hosted alternative)
      - install: 'pip install mlflow'

* open source datasets for aircraft flyover noise
  - FSD50K
    * ~500-1000 aircraft-related clips
    * aicraft and other sounds, good quality
    * https://zenodo.org/records/4060432
  - ESC-50
    * 80 aircraft clips, high quality
    * mostly helicopters, some airplane
    * https://github.com/karolpiczak/ESC-50
  - AudioSet (Google) - aircraft subset
    * ~100,000 aicraft clips
    * coarse labels
      - aircraft ontology: https://www.google.com/url?sa=i&source=web&rct=j&url=https://research.google.com/audioset/ontology/aircraft_1.html&ved=2ahUKEwiV-qX8oqCTAxXmDTQIHQcmOhUQy_kOegQIDxAB&opi=89978449&cd&psig=AOvVaw2VXj2mU4pmUiiiisMGPOjZ&ust=1773608043077000
    * https://research.google.com/audioset/dataset/aircraft.html
    * git clone https://github.com/speedyseal/audiosetdl
      - Run the script: ./download_subset_files.sh /path/to/save
  - AeroSonicDB (YPAD-0523) Kaggle database
    * low-flying aircraft sounds
    * 625 aircraft recordings, 18-60 secs, 8.87hrs in total
    * background: 3.52 hrs of ambient/silent audio, urban soundscape with aicraft annotaitons
    * includes ADS-B data
    * https://www.google.com/url?sa=i&source=web&rct=j&url=https://www.kaggle.com/datasets/gray8ed/audio-dataset-of-low-flying-aircraft-aerosonicdb/data&ved=2ahUKEwi8n9Wio6CTAxVfIjQIHYGSOhcQy_kOegQIBBAB&opi=89978449&cd&psig=AOvVaw11qX86VwD-IngLeX9ht9vz&ust=1773608123543000
    * https://zenodo.org/records/12775560
    * https://github.com/aerosonicdb/AeroSonicDB-YPAD0523
  - UrbanSound8K (subset)
    * ~100 aircraft clips
    * coarse labels, mixed with other sounds
    * https://urbansounddataset.weebly.com/urbansound8k.html
  - TAU Urban Acoustic Scenes 2020 - aircraft subset
    * ~1,500 recordings
    * very detailed information
    * https://data.nasa.gov/dataset/Aircraft-Noise-Events-Database
  - TAU Urban Acoustic Scenes datasets and Sony-Tec Sound Event Dataset
    * contain many flyovers mixed with traffic

* Model Architecture
  - choose appropriate model for each task (maybe same one for multiple tasks?)
  - evaluate CNN and Spectrogram with multi-task heads as a candidate model architecture
    * convert audio to mel/log-mel spectrograms and use image classification-like CNN model, with multiple heads -- one for each task
  - model architecture choice continuum:
    * <100 clips/type  → PANNs embeddings + small head (pretrained/panns.py)
    * 100–500/type     → AST fine-tuning (pretrained/ast.py)  ← likely current regime
    * 500–2000/type    → ResNet-34 with pretrained weights (models/resNetCNN.py)
    * >2000/type       → ResNet-34 or DeepVehicleCNN from scratch
  - starting with ResNet-34 with ImageNet pretrained weights is reasonable but not optimal
    * ImageNet features are RGB photos, not spectrograms
  - recommended Architecture: AST Fine-Tuning for Phase 1
    * pretrained/ast.py — MIT AST fine-tuned on AudioSet
    * why AST beats ResNet-34 in Phase 1:
      1. AudioSet explicitly includes aircraft sounds (jet engine, helicopter, propeller).
         The pretrained features are already audio-domain, not image-domain
      2. AST fine-tuning converges in 5–15 epochs with small datasets where ResNet needs 50+
      3. The HuggingFace Trainer handles checkpointing, evaluation, and early stopping cleanly
   * when to switch to ResNet-34 / DeepVehicleCNN:
     - Once you have >500 labeled clips per class
     - Once AST inference latency is a problem
     - Once you want tighter control over the spectrogram config (AST has its own feature extractor)

* Project Phases
  - Phase 1: Classify by vehicle type (multi-label, single-aircraft clips only)
    * Vehicle types
      - high-level category
        * Propulsion: jet, turbine, piston
        * Engine count: [1-N]
        * Wing type: rotary (helicopter) vs. fixed-wing
      - detailed category
        * from the FAA's Releaseable Aircraft Registration database
          - do lookup in database based on ICAO24 number
        * ICAO24 → category using FAA TYPE-ACFT + TYPE-ENG + NO-SEATS fields
          - falls back to typeToCategory(aircraftType) for foreign aircraft
        * doesn't work for non-US, military, special, or excluded aircraft
    * Dataset
      - ?

  - Phase 2: Classify by Direction of travel (single-aircraft clips only)
    * one of eight cardinal classes (N/NE/E/SE/S/SW/W/NW)

  - Phase 3: Classify by speed (single-aircraft clips only)
    * scalar regression (knots)

  - Phase 4: Multi-aircraft handling
    * detect/classify overlapping aircraft in the same clip

  - Phase 5 (longer-term): Additional attributes 
    * distance, altitude, specific engine/airframe model

* Model Training Process
  - **TBD**

* Experiment Log — Backbone & Spectrogram Investigation (2026-07-03)
  - Context: 8-class categorical Phase 1, ~1000 curated clips/class (dataset_best1000),
    frozen-backbone ResNet-18 plateaued at mAP 0.405 / tuned Macro-F1 0.449-0.460.
    Three classes stayed weak across every run: turboprop (AP ~0.28), narrowbody_jet
    (AP ~0.22-0.26), business_jet (AP ~0.42-0.44).
  - PANNs embeddings test (frozen AudioSet CNN14, 2048-dim, MLP head):
    * hypothesis: AudioSet pretraining (includes aircraft/helicopter/jet-engine/propeller
      classes) should transfer better than ImageNet features on spectrograms
    * result: mAP 0.387-0.406, Macro-F1 0.429 — statistically a wash with ResNet-18,
      not the expected 0.50+. No systematic per-class advantage; helicopter and
      turboprop (the classes AudioSet models most directly) showed no real lift.
      One class (widebody_jet) was clearly worse under PANNs (AP 0.31 vs 0.45),
      suggesting the pooled whole-clip embedding loses information (duration,
      low-frequency harmonics) that the full 5 s spectrogram retains.
    * conclusion: backbone choice is not the bottleneck. Two architecturally very
      different feature extractors land in the same mAP band and fail on the same
      classes — that consistency points at the input representation or the data,
      not model capacity.
  - Clip quality investigation (`scripts/evalClipQuality.py`) on the actual curated
    train+val set (not the raw candidate pool):
    * RMS/loudness: weak classes are NOT quieter. turboprop (-37.1 dBFS mean) and
      narrowbody_jet (-35.5 dBFS) are louder than piston_twin (-45.5 dBFS), which is
      the best-performing class (AP 0.81). Rules out SNR as the explanation.
    * Silence fraction: inversely correlated with performance (best class has the
      MOST silence, 73%; worst classes have the least, 43-55%) — not a quality flag.
    * Clipping: 0% everywhere. Spectral flatness: near-zero (tonal) everywhere.
      No noise-dominance or saturation issue on any class.
    * Low-frequency energy ratio: 92-99% of energy sits below 200 Hz, universal
      across ALL classes (not a differentiator between weak/strong classes) — but
      revealed the representational issue below.
  - Spectrogram resolution check: with the config in place at the time
    (n_fft=1024, hop=512, n_mels=128, sr=44100, fmax=22050 default), computed how
    many of the 128 mel bins actually cover the 0-200 Hz band where the signal lives:
    * only 6/128 mel bins peak below 200 Hz
    * the underlying STFT itself has only 5 frequency bins below 200 Hz (43.1 Hz
      bin width) — the mel filterbank can't resolve finer than that regardless
    * business_jet/narrowbody_jet/regional_jet are all recorded from similar
      altitude (~4300-4700 ft) and distance (~2.8-3.0 km); the acoustic differences
      between these jet subtypes plausibly live in subtle low-frequency engine
      harmonic spacing — exactly the band this config resolves worst
  - Decision: switch mel spectrogram config to **n_fft=2048, hop_length=512,
    n_mels=128, fmax=8000** (was n_fft=1024, fmax=22050/default).
    * fmax=8000 reallocates all 128 mel bins to the band that actually contains
      aircraft signal (spectral centroids across classes topped out ~3800-3900 Hz)
      instead of spending most of them above it
    * n_fft=2048 halves the underlying FFT bin width (43.1 Hz -> 21.5 Hz),
      giving finer resolution within the low-frequency band specifically
    * time-frame count is unaffected (hop_length unchanged), so model input shape
      and ResNet conv1 dims don't change
    * requires regenerating all `.spec.npy` sidecars — old ones were computed
      under the previous config and are silently wrong for the new one
  - Result (fmax=8000, retrained frozen-backbone ResNet-18, epoch=16/val_f1=0.451
    checkpoint): mAP 0.405 -> 0.430, Macro-F1 0.449 -> 0.473. Real net gain, but
    per-class pattern didn't match the hypothesis cleanly:
    * helicopter +0.095 AP, piston_single +0.063, regional_jet +0.045,
      turboprop +0.039, narrowbody_jet +0.018, piston_twin +0.029 (6/8 improved)
    * business_jet -0.041, widebody_jet -0.048 (regressed)
    * narrowbody_jet still weakest overall (AP 0.273) — jet-subtype confusion
      not resolved
    * interpretation: fmax=8000 helped classes whose signal is genuinely
      sub-1kHz (rotor blade-passage frequency, piston combustion rate) most,
      but was too aggressive for the largest jets — business/widebody plausibly
      have real discriminative content in the 8-12 kHz range (APU noise,
      high-bypass fan harmonics) that fmax=8000 discarded outright
  - Follow-up: raised fmax to 12000 (n_fft=2048, n_mels=128 unchanged) to give
    the jets back some headroom while keeping most of the low-frequency
    resolution gain.
  - Result (fmax=12000, epoch=14/val_f1=0.452 checkpoint): mAP 0.409, Macro-F1
    0.462 — WORSE than fmax=8000 (0.430 / 0.473) in aggregate, though still
    better than the original no-cap config (0.405 / 0.449):
    * widebody_jet +0.089, piston_single +0.100 vs fmax=8000 (recovered, as
      hoped — real content in 8-12 kHz)
    * but helicopter -0.155, turboprop -0.136, regional_jet -0.088 vs fmax=8000
      (collapsed — these had been the biggest fmax=8000 winners)
    * net: not diminishing returns, a genuine tug-of-war. Widening the band
      helped the classes that need 8-12 kHz content but diluted resolution
      for the classes whose signal is concentrated low, because both groups
      draw from the same fixed 128-mel-bin budget under one global cutoff.
    * conclusion: no single global fmax can serve both class groups — this is
      the empirical case for a genuinely different representation, not just
      more tuning of one scalar.
  - Decision: replace the single global fmax with a **dual-channel, non-
    overlapping mel spectrogram** — channel 0 covers 0-8000 Hz (full 128-mel
    budget, the band that won for helicopter/turboprop/piston_twin), channel 1
    covers 8000 Hz-Nyquist (full 128-mel budget, the band that won for
    widebody_jet/piston_single/business_jet). Implemented as a 2-channel input to the same
    ResNet-18 (conv1 changed from Conv2d(1,64,...) to Conv2d(2,64,...) — conv1
    is replaced from scratch regardless of channel count, so no pretrained
    weight remapping issue). Rejected full parallel/duplicate backbones (2x
    11M params) as too much added capacity for ~800-1000 clips/class given
    the overfitting fights already underway (freezeBackbone, patience, weight
    decay). `_dualBandMelDb()` in toolchain.py is now the single source of
    truth for spectrogram computation, imported by precomputeSpecs.py,
    evalModel.py, and vizSpecs.py.
  - Result (dual-channel Conv2d(2,64,...), epoch=16/val_f1=0.449 checkpoint):
    mAP 0.424, Macro-F1 0.475 (best Macro-F1 so far; mAP second-best behind
    fmax=8000's 0.430). NOT a clean best-of-both-worlds:
    * narrowbody_jet 0.295 — best-ever score for the weakest class
    * piston_twin 0.789 — regressed BELOW every prior config (0.806-0.855),
      the biggest single-class casualty of this change
    * helicopter 0.392, turboprop 0.278 — better than fmax=12000 but short of
      fmax=8000's peak (0.426, 0.316); widebody_jet 0.415, business_jet 0.402
      — better than fmax=8000 but short of fmax=12000's peak (0.491, 0.431)
    * every "recovered" class landed at a midpoint between its two single-fmax
      extremes rather than at (or above) its own best — a blend, not a
      resolution of the tug-of-war
    * likely cause: `Conv2d(2,64,...)` applies joint filters that sum across
      both input channels starting at the very first layer — low-band and
      high-band information gets blended immediately rather than each band
      developing independent features first. A single-channel conv1
      inherently cannot keep the bands' information separate no matter how
      the input is stacked.
  - Decision: replace the single fused conv1 with a genuine **dual-stem
    architecture** — `stemLow`/`stemHigh`, each a fresh
    Conv2d(1,64,7x7,stride=2)+BatchNorm2d+ReLU+MaxPool mirroring ResNet's own
    stem, one per band. Outputs concatenated (64+64=128ch) then reduced by a
    new learned `fuse` 1x1 Conv2d(128,64) back to the width the pretrained
    `layer1-4` trunk expects, so the shared trunk (~11.2M of the 11.3M total
    params) is untouched and stays pretrained/ImageNet-initialized. This lets
    each band develop independent low-level features before merging, instead
    of blending them in the first layer. `--freezeBackbone` now freezes both
    stems + fuse + layer1-3 (mirrors the original design's precedent of
    freezing the freshly-initialized conv1 alongside pretrained layer1-3);
    `--unfreezeEpoch` unfreezes all four modules together. Trainable/frozen
    param counts are unchanged (~8.5M / ~2.8M) — the new stem+fuse add only
    ~14.7K params, folded into the frozen bucket.
  - Result (dual-stem, epoch=37/val_f1=0.449 checkpoint): mAP 0.441 (best of
    all 5 configs tried), Macro-F1 0.474 (ties the previous best, 0.475):
    * turboprop 0.397 — best-ever score by a wide margin (previous best 0.316)
    * widebody_jet 0.469 — nearly matches its single-fmax peak (0.491)
    * piston_twin 0.846 — substantially recovered from the fused-conv1
      regression (0.789), close to its own peak (0.855)
    * but helicopter 0.376 and piston_single 0.341 dipped below where they'd
      been under fused-conv1 (0.392, 0.424) — separating the stems reduced
      cross-class competition but didn't eliminate it; the trunk and
      classifier head are still shared across all 8 classes
    * narrowbody_jet 0.291 — still under 0.30 across every one of the 5
      configs tried (0.255/0.273/0.245/0.295/0.291). Same ceiling regardless
      of spectrogram or architecture change is a strong signal this class is
      data-limited, not representation-limited.
  - **BANKED as the production config (2026-07-03).** Dual-stem architecture
    + dual-band spectrogram (FMAX_LOW=8000/FMIN_HIGH=8000) is the current
    best result across mAP, Macro-F1, and per-class breadth. Progression
    across the 5 experiments: mAP 0.405 -> 0.430 -> 0.409 -> 0.424 -> 0.441;
    gains per iteration are shrinking while engineering cost per iteration is
    flat or rising (dual-stem was the most invasive change so far). Treating
    this as the point of diminishing returns for spectrogram/architecture
    tuning specifically.
  - Next lever: data, not architecture. narrowbody_jet's flat ceiling across
    every representation tried is the strongest evidence in this log that a
    specific class is capped by data (volume, quality, or genuine acoustic
    ambiguity with regional_jet/business_jet — the three classes recorded
    from similar altitude/distance, see "Clip quality investigation" above)
    rather than by anything fixable through more architecture work.

* Data Volume Rebuild (2026-07-04): 1000/class -> 3000/class, and the
  regional_jet collapse
  - Rebuilt the curated dataset via `buildQualityDatasetFromRecordings.py
    --bestN 3000 --deepAnalysis --faaDatabaseDir data/ReleasableAircraft`
    (previous build used `--bestN 1000`, ranking method and FAA-DB usage
    unconfirmed). Every class capped at 3000 except none needed to fall back
    (piston_twin's raw pool grew to 3386, comfortably clearing 3000).
  - Result (dual-stem architecture, unchanged): mAP 0.441 -> 0.5244,
    Macro-F1 0.474 -> 0.5322 — by far the largest single jump of the whole
    investigation. 6/8 classes improved substantially (piston_single +0.22,
    business_jet +0.21, widebody_jet +0.16, helicopter +0.04 to +0.14
    depending on run, turboprop +0.11, narrowbody_jet +0.06 — finally broke
    its flat 0.30 ceiling from every prior config). Confirms data volume, not
    architecture, was the remaining bottleneck.
  - **regional_jet collapsed: 0.40 -> 0.18-0.20 AP**, now by far the worst
    class (previous floor across all 5 earlier configs was 0.245). Reproduced
    across two independent training runs on the same rebuilt dataset
    (0.204, 0.180) — not a fluke.
  - Diagnostic process (see EVAL.md "Confusion Breakdown" for the tool this
    added, `evalModel.py --confusionFor <class>`):
    1. Hypothesis: FAA-based relabeling (first build to use
       `--faaDatabaseDir`) reassigned regional_jet clips to business_jet
       under seat-count rules. Checked directly against the raw pool:
       **ruled out** — 100% identical category labels for all 30,433 clips
       common to both the old and new raw pools.
    2. Hypothesis: multi-label co-occurrence with confusable classes
       increased. **Ruled out** — 0% multi-label overlap for regional_jet in
       both old and new curated sets.
    3. Hypothesis: going 3x deeper into the quality-ranked pool pulled in
       disproportionately worse regional_jet clips specifically. **Ruled
       out** — the RMS gap between original-selection and newly-added clips
       (7.9 dB) is unremarkable; every class shows a similar 7-10 dB gap,
       including classes that improved substantially (business_jet 9.2 dB,
       piston_twin 9.8 dB).
    4. **Confirmed via `--confusionFor regional_jet`**: among 497 true
       regional_jet val clips, narrowbody_jet has both the highest mean
       predicted probability (0.513 vs regional_jet's own 0.481) and the
       highest top-1 "best guess" rate (27.8% vs regional_jet's own 18.9%).
       The model isn't merely uncertain about regional_jet — it's
       systematically absorbing it into narrowbody_jet specifically.
  - Root cause: regional_jet (20-100 seats) and narrowbody_jet (101-220
    seats) are adjacent FAA seat-count bins (`faaDatabase.py::_deriveCategory`)
    on what is acoustically closer to a continuum than a hard boundary —
    aircraft near the 100-seat line (e.g. larger E-Jet variants) plausibly
    sound genuinely similar regardless of which side of the threshold their
    registry seat count places them on. narrowbody_jet's 3x larger, more
    confident training signal this round likely sharpened its decision
    boundary enough to claim adjacent acoustic territory that used to be
    ambiguous/shared.
  - **Decision (2026-07-04): merged regional_jet into narrowbody_jet.**
    Chose this over collecting more regional_jet data because the confusion
    is systematic and one-directional (narrowbody_jet wins the model's
    top-1 guess even more often than regional_jet does for its own true
    clips) — that pattern points at the ground-truth boundary itself being
    too fine for audio to resolve, not at a fixable data-volume gap. Category
    count: 8 -> 7.
    * Code: `faaDatabase.py::_deriveCategory` — seat-count branches merged
      (`<20`->business_jet, `20-220`->narrowbody_jet, `>220`->widebody_jet).
      `typeCategories.py` — `_EXPLICIT` entries and the `_KEYWORD_RULES`
      regional-jet block (crj/erj/e170-e195/atr keywords) now resolve to
      narrowbody_jet; `regional_jet` removed from `CATEGORIES`.
    * Data: relabeled `type_categories` in-place in the already-built CSVs
      (train.csv/val.csv locally and at dataset_best3000) rather than
      re-extracting from raw recordings — a label-only change, no audio
      content differs. Result: narrowbody_jet's curated count doubled to
      6000 (3000 original + 3000 former regional_jet, no overlap since
      regional_jet had 0% multi-label co-occurrence), all other classes
      unchanged at 3000. Total clip count unchanged (24000).
    * Result (epoch=26/val_f1=0.553 checkpoint): mAP 0.588, Macro-F1 0.572
      (7 classes). Isolating genuine gain from the mechanical effect of
      averaging over 7 classes instead of 8: dropping regional_jet from the
      OLD 8-class results (no merge) would already average to mAP ~0.574;
      the merge gets to 0.588, a real **+0.014** beyond just removing the
      broken class. That gain is concentrated exactly where predicted:
      narrowbody_jet 0.401 -> 0.552 (+0.151, now the largest class at 6000
      clips), helicopter 0.420 -> 0.485 (+0.065). piston_single/turboprop/
      piston_twin held flat; widebody_jet -0.035 and business_jet -0.063
      dipped modestly (within the run-to-run variance already observed
      elsewhere this session — e.g. helicopter swung 0.42-0.51 across
      identical reruns — not treated as a new problem unless it persists).
      Per-class spread is now 0.485-0.799, much tighter than the 0.18-0.86
      spread before the merge. **Closed** — this thread is resolved; the
      merge decision is validated.

* business_jet Investigation (2026-07-05)
  - business_jet dipped -0.063 (0.592 -> 0.529) right after the regional_jet
    merge — suspicious timing, checked with `--confusionFor business_jet`
    rather than assumed to be variance.
  - Confusion breakdown (n=679 true business_jet val clips): business_jet
    wins its own top-1 guess 48.9% of the time (mean prob 0.623) — clearly
    ahead of any single competitor. This is qualitatively different from
    regional_jet's pattern (where the wrong class won outright): not a
    single dominant systematic absorption, but a two-sided split —
    narrowbody_jet (20.5% top-1) and turboprop (17.8% top-1).
  - Investigated the narrowbody/widebody-direction share of the confusion
    directly (not just inferred): scanned business_jet's raw `vehicle_types`
    for large-airframe model designators and found `767-3S2F` (38 clips) and
    `777-FS2` (84 clips) — 122/3000 (4.1%). Keyword rules explicitly list
    "777"/"767" as widebody indicators, so these were reaching business_jet
    via the FAA structural path (`typeAcft=5`, `typeEng=4/5`, `noSeats<20`),
    not the keyword fallback — i.e. genuinely privately-registered/VIP-
    converted 767s and 777s with a low registered seat count, sounding
    exactly like any other 767/777 despite the seat-count-driven label.
  - Fix: `faaDatabase.py::categoryForIcao24` now overrides business_jet to
    the model-string keyword category when `_deriveCategory` says
    business_jet but the model string matches widebody_jet/narrowbody_jet —
    airframe/model family is a more reliable acoustic signal than a private
    cabin's seat count. Narrowly scoped: only fires when the seat-count path
    already said business_jet, verified it does not affect genuine bizjets
    (e.g. `BD-100-1A10`/Challenger 300 unaffected). Existing dataset CSVs
    relabeled in place: 122 clips business_jet -> widebody_jet (business_jet
    3000 -> 2878, widebody_jet 3000 -> 3122).
  - This explains and fixes the narrowbody/widebody-direction share of the
    confusion (~4% contamination) but not the larger turboprop-direction
    share (17.8% top-1) — that's likely genuine acoustic heterogeneity
    within business_jet itself (light jets/VLJs like the Eclipse 500 at the
    small end plausibly sound closer to a large turboprop than to a
    midsize/large-cabin business jet). Not addressed here; business_jet's
    own-class top-1 rate (48.9%) suggests this is a "hard but learnable"
    class rather than a structurally broken one, so left as-is rather than
    forcing another category split without stronger evidence.
  - Result (2 runs on identical post-fix data, epoch=37/val_f1=0.555 and
    epoch=40/val_f1=0.558): business_jet 0.529 -> ~0.557 avg (real
    improvement, confirmed reproducible), but widebody_jet 0.647 -> ~0.586
    avg (real regression, tightly reproducible: 0.587 and 0.584 across the
    two runs — much tighter than this pipeline's normal ~0.03-0.09 run-to-run
    noise, so not attributable to variance). Net mAP -0.012 (0.588 -> 0.576,
    identical both runs). The 122 reclassified VIP-jet clips are correct
    labels but atypical training examples (privately-flown 767/777s likely
    have different altitude/speed/distance profiles than commercial
    widebody traffic), diluting an otherwise more homogeneous widebody_jet
    class. Kept the fix (correct ground truth over a small metric hit) and
    moved to testing the dilution theory directly rather than reverting.

* widebody_jet Dilution Test (2026-07-05)
  - Hypothesis: the atypical VIP-jet minority (122/3122, ~4%) hurt
    widebody_jet because it was a large fraction of a still-small class;
    adding more typical widebody_jet clips should dilute it back down and
    let the class recover.
  - Implementation: rather than re-running the full 8-class
    extraction+deep-analysis pipeline, ran `_runDeepAnalysis`/
    `_compositeScore` (from `evalClipQuality.py`) directly on just
    widebody_jet's ~44K-clip raw pool, selected the new top 6000 (doubling
    it, matching narrowbody_jet's earlier doubling), and merged into the
    existing dataset (swap old widebody_jet rows for the new selection,
    copy new WAVs, precompute their `.spec.npy` with `--skipExisting`).
    widebody_jet: 3122 -> 6000. Total dataset: 24000 -> 26878.
  - Result: **dilution theory confirmed for widebody_jet** — AP recovered
    past even the pre-business_jet-fix peak, to a new best (0.770, then
    0.807 -> 0.828 across repeated runs on this data, i.e. consistently
    the strongest class after piston_twin).
  - But narrowbody_jet regressed (~0.536 avg before -> ~0.435 avg across two
    widebody-boost runs, with high run-to-run variance: 0.388 and 0.482 on
    identical data — wider than this pipeline's usual noise band).
    Positive-rate shift explains the mechanism: widebody_jet went from
    14.5% -> 25.4% of the dataset, becoming the single largest class
    (bigger than narrowbody_jet's 19.5%) — the same "bigger adjacent class
    wins the shared boundary" dynamic seen with regional_jet/narrowbody_jet,
    just one category up the seat-count ladder (narrowbody_jet 20-220 seats
    vs widebody_jet >220).
  - **Checked via `--confusionFor narrowbody_jet` before concluding a merge
    was warranted** (properly converged checkpoint, epoch=10/val_f1=0.544):
    narrowbody_jet narrowly WINS its own top-1 (39.2% vs widebody_jet's
    37.0%, mean prob 0.617 vs 0.607) — qualitatively different from the
    regional_jet case, where narrowbody_jet had decisively beaten regional_jet
    on both metrics. This is a genuinely hard, close boundary (largest
    narrowbody variants like a 737 MAX 10/A321 aren't acoustically far from
    the smallest widebodies like a 767), but the model hasn't given up on
    separating them — **no merge recommended** here.
  - Plateau/grokking check (prompted by "does training bump after a long
    plateau" question): ran `--maxEpochs 150 --patience 60` on this dataset
    to rule out a hidden late-training improvement. Best checkpoint landed
    at **epoch 10** — earlier than the established 26-42 range, not later.
    val_loss was climbing while train_loss kept falling by epoch 70 (plain
    overfitting). Confirms this setup converges early; no evidence of
    delayed generalization ("grokking") at this data scale.
  - **BANKED as the production checkpoint (2026-07-05).** mAP 0.591 —
    best of the entire investigation (from 0.405 at the start: dual-stem
    architecture, 3x data, regional_jet merge, business_jet mislabel fix,
    widebody_jet dilution, in that order). Full progression:
    0.405 -> 0.441 -> 0.524 -> 0.588 -> 0.576 -> 0.591.
  - Remaining weak classes, not yet individually investigated the way
    regional_jet/business_jet/widebody_jet were: turboprop (~0.47-0.50) and
    helicopter (~0.43-0.49). Natural next targets if continuing this line
    of work.

* Incremental-Growth Regression via addNewRecordings.py (2026-07-21)
  - Context: since the 2026-07-05 banked checkpoint (mAP 0.591, 26,878 clips,
    sized by `buildQualityDatasetFromRecordings.py --bestN`: 3000/class except
    narrowbody_jet/widebody_jet at 6000 post-merge/dilution),
    `scripts/addNewRecordings.py` has been run repeatedly to ingest new
    recordings — it has no per-category cap, admitting every new clip that
    clears the current per-category quality bar (fast/RMS mode). Curated
    dataset grew 26,878 -> 87,114 clips (70,147 train + 16,967 val) by the
    time of the next full eval, run with `--freezeBackbone --weightDecay 0.05
    --maxEpochs 150 --patience 60`.
  - Result: mAP fell 0.591 -> 0.460 — a broad regression, not a single-class
    issue. Per-class AP: piston_single 0.780 (only class that held up/
    improved), widebody_jet 0.598 (down from ~0.828 post-dilution-test),
    narrowbody_jet 0.499, piston_twin 0.446 (down from ~0.81-0.85, previously
    the best class of the whole investigation), business_jet 0.313 (down from
    ~0.557), turboprop 0.313 (down from ~0.47-0.50), helicopter 0.274 (down
    from ~0.43-0.49).
  - Two contributing mechanisms found in the data (diagnosis, not yet
    confirmed via `--confusionFor` — see next step below):
    1. **Severe growth imbalance.** Per-category raw-pool size in
       `dataset.csv` varies by >50x (piston_single 210k vs piston_twin 4.1k),
       so an uncapped "admit anything at least as good as the current worst"
       gate lets the biggest classes grow fastest every run. Counts before ->
       after the 2026-07-06 cutoff: piston_single 6835 -> 31,805 (+365%),
       business_jet 3145 -> 4886 (+55%), helicopter 3039 -> 4399 (+45%),
       piston_twin 3226 -> 3750 (+16%, essentially static). piston_single is
       now 37.3% of the val positive rate, >8x piston_twin's 5.3% — the same
       "bigger class wins the shared acoustic boundary" dynamic already
       documented above for regional_jet/narrowbody_jet and
       narrowbody_jet/widebody_jet, plausibly recurring here between the two
       piston classes (same engine family, differing mainly by engine count).
    2. **Quality drift toward the floor.** Comparing average `clipRms` of
       clips added before vs. after 2026-07-06 (proxy for how deep into each
       category's quality-ranked pool the newer clips sit): business_jet
       0.0068 -> 0.0043, helicopter 0.0075 -> 0.0052, turboprop
       0.0079 -> 0.0049, widebody_jet 0.0099 -> 0.0037, narrowbody_jet
       0.0090 -> 0.0058, piston_single 0.0113 -> 0.0083 — all substantially
       lower, consistent with newly-admitted clips sitting closer to the
       quality floor than the hand-curated best-N originally did.
       **piston_twin is the exception**: avgRms rose 0.0034 -> 0.0054 after
       the cutoff, so its regression is not explained by quality drift —
       points at mechanism (1) instead for this class specifically.
  - **Update: mechanism (1) tested via `--confusionFor piston_twin` and
    ruled out for this class.** piston_twin still solidly wins its own top-1
    (42.1% of 895 true clips, mean prob 0.579) — the next-highest confusor
    (piston_single) only takes 12.7%, and the rest spreads nearly evenly
    across every other class (turboprop 11.4%, narrowbody_jet 12.5%,
    widebody_jet 10.4%, business_jet 6.6%, helicopter 4.2%). This is a
    diffuse pattern, qualitatively unlike regional_jet's single-dominant-
    absorber case — closer to the business_jet "hard but learnable" pattern
    above. piston_single is not specifically absorbing piston_twin.
  - **Update: mechanism (2) also tested directly and ruled out for
    piston_twin.** Ran the full 7-metric `_runDeepAnalysis`/`_compositeScore`
    pipeline (not just RMS) on a 150-clip random sample each side of the
    2026-07-06 cutoff: composite score 0.5403 (before) vs 0.5463 (after) —
    flat to fractionally better, not worse. So piston_twin's own newly-added
    clips are not lower quality by any of the 7 metrics tracked.
  - **Net: piston_twin's AP collapse (~0.81-0.85 -> 0.446-0.482) is not
    explained by its own data quality or by a single absorbing class.**
    Remaining candidates, not yet tested: (a) a genuinely global/systemic
    effect of the 3.2x dataset growth and resulting imbalance on training
    dynamics (e.g. pos_weight/decision-threshold calibration shifting for
    every class at once, not a pairwise boundary fight — consistent with
    every class except piston_single regressing, not just piston_twin);
    (b) the current best checkpoint converged much earlier (epoch 11,
    val_f1 0.461) than the banked run (epoch 26-42 range across earlier
    stages) — possibly reflects the frozen-backbone MLP head hitting a
    harder optimization landscape sooner with this much more data/imbalance,
    rather than a data-quality problem per se. The most direct causal test
    of (a): rebuild the curated set with per-category caps close to the
    banked composition (e.g. cap piston_single well below its current
    31,805) and retrain, to see if performance recovers toward 0.591 —
    isolates "did uncapped growth cause this" without needing to identify
    the exact mechanism first.
  - Implication for `scripts/addNewRecordings.py`: as currently designed (no
    per-category cap), every future run will keep widening this imbalance,
    since piston_single's raw pool so vastly exceeds the rare classes'.
    Before the next incremental run, consider a per-category ceiling
    (mirroring `--maxPerClass` in `buildDataset.py`) and/or retraining with
    `--balanceClasses`.
