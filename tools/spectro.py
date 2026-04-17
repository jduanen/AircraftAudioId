import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys

WAV_FILE_PATH = sys.argv[1]

# Load audio
y, sr = librosa.load(WAV_FILE_PATH, sr=48000)

# Compute mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram - 1kHz Tone Test')
plt.tight_layout()
plt.show()
