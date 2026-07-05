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
  - Retrain pending to confirm the fix recovers business_jet's dip and
    whether widebody_jet absorbs the reclassified clips cleanly.
