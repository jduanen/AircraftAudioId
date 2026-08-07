#!/home/jdn/.virtualenvs/aircraftAudioId/bin/python3
#
# Script to animate real-time FFT of WAV file

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import soundfile as sf
import pyaudio
import threading
import time

# Load audio
data, sr = sf.read('yourfile.wav')
if len(data.shape) > 1: data = data[:, 0]
fft_size = 2048
freqs = np.fft.rfftfreq(fft_size, 1 / sr)
CHUNK = fft_size  # Audio block size matches FFT

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, 
                output=True, frames_per_buffer=CHUNK)

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Audio Playback + Realtime FFT')
ax.set_xlim(20, sr / 2)
ax.set_ylabel('Magnitude (dBFS)')
ax.set_xlabel('Frequency (Hz)')
ax.grid(True)
line, = ax.plot(freqs, np.zeros(len(freqs)), color='cyan', lw=1)

# Playback + FFT in background thread
playing = True
pos = 0

def audio_thread():
    global pos
    while playing and pos < len(data):
        chunk = data[pos:pos + CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
        
        # Play audio
        stream.write(chunk.astype(np.float32).tobytes())
        
        # Update position for FFT sync
        pos += CHUNK // 2  # 50% overlap
        time.sleep(0.01)  # ~100 FPS sync
    stream.stop_stream()
    stream.close()
    p.terminate()

def update(frame):
    global pos
    if pos + fft_size > len(data): 
        return line,
    
    chunk = data[int(pos):int(pos + fft_size)]
    fft = np.abs(np.fft.rfft(chunk))
    fft_db = 20 * np.log10(np.maximum(fft, 1e-12))
    
    line.set_data(freqs, fft_db)
    peak = np.max(fft_db)
    ax.set_ylim(-100, max(peak + 6, 6))
    
    return (line,)

# Start audio playback thread
threading.Thread(target=audio_thread, daemon=True).start()

# Animation (visual sync)
n_frames = len(data) // (CHUNK // 2)
ani = anim.FuncAnimation(fig, update, frames=n_frames, interval=20, 
                        blit=True, repeat=False)
plt.tight_layout()
plt.show(block=True)