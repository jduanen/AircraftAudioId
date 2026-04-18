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
    --> TODO need to come up with the same for aircraft flyover noise
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
