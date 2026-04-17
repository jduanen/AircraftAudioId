import librosa
import numpy as np

WAV_FILE_PATH = '/home/jdn/1.wav'

y, sr = librosa.load(WAV_FILE_PATH, sr=48000)

# STFT - take MAGNITUDE (abs) to get real power
D = np.abs(librosa.stft(y))**2  # Power spectrum (real numbers)
freqs = librosa.fft_frequencies(sr=sr)

# Find fundamental (~1000 Hz)
fundamental_bin = np.argmin(np.abs(freqs - 1000))
fundamental_power = np.mean(D[fundamental_bin])

# Calculate harmonics (2nd, 3rd, etc.)
harmonics = []
for h in range(2, 11):
    harm_bin = np.argmin(np.abs(freqs - 1000 * h))
    if harm_bin < len(freqs):
        harmonics.append(np.mean(D[harm_bin]))

# THD = sqrt(sum(harmonic powers)) / fundamental
harmonic_power = np.sqrt(sum(np.square(harmonics)))
thd_percent = 100 * harmonic_power / fundamental_power
thd_db = 20 * np.log10(harmonic_power / fundamental_power + 1e-10)  # Avoid log(0)

print(f"Peak frequency: {freqs[fundamental_bin]:.1f} Hz")
print(f"THD: {thd_percent:.3f}% ({thd_db:.1f} dB)")
print(f"1kHz RMS: {20*np.log10(np.sqrt(fundamental_power)):.1f} dBFS")