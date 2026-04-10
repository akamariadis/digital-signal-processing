import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

filename = '91003005.wav'  # and 63019002.wav
y, sr = librosa.load(filename, sr=None)

GF = 0.9
sensitivity = -165
denominator = 10 ** (sensitivity / 20)
pressure = (1.5 * y * GF) / denominator
time = np.arange(len(pressure)) / sr
N = 3
cutoff = 200
b, a = butter(N, cutoff, btype='highpass', fs=sr)
filtered_pressure = filtfilt(b, a, pressure)
plt.figure(figsize=(12, 5))
plt.plot(time, pressure, color='teal', alpha=0.5, label='Αρχικό Σήμα Πίεσης (με θόρυβο)')
plt.plot(time, filtered_pressure, color='darkred', alpha=0.9, linewidth=1, label='Φιλτραρισμένο Σήμα (>200 Hz)')
plt.title(f'Butterworth Filter(200Hz) - {filename}')
plt.xlabel('Χρόνος (Δευτερόλεπτα)')
plt.ylabel('Πίεση (μPa)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()