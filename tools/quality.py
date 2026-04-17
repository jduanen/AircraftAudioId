import librosa
import numpy as np
import sys

WAV_FILE_PATH = sys.argv[1]

# Load silence recording
y, sr = librosa.load(WAV_FILE_PATH, sr=48000)

# Calculate noise floor metrics
rms_db = 20 * np.log10(librosa.feature.rms(y=y)[0].mean() + 1e-10)
noise_floor_db = 20 * np.log10(np.std(y) + 1e-10)

# A-weighted noise estimate (rough)
a_weighted = librosa.feature.rms(y=y, center=False).mean()
a_noise_db = 20 * np.log10(a_weighted + 1e-10)

print(f"RMS Noise Floor: {rms_db:.1f} dBFS")
print(f"Peak-to-Peak Noise: {noise_floor_db:.1f} dBFS") 
print(f"A-weighted Noise: {a_noise_db:.1f} dBFS")

# Spectral flatness (1.0 = white noise, 0.0 = tonal)
flatness = librosa.feature.spectral_flatness(y=y)[0].mean()
print(f"Spectral Flatness: {flatness:.3f}")
