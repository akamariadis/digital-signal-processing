import librosa
import numpy as np
import matplotlib.pyplot as plt

filename = '91003005.wav'
whale_type = 'Sperm Whale'
GF = 0.9
SENSITIVITY = -165
y, sr = librosa.load(filename, sr=None)
denominator = 10 ** (SENSITIVITY / 20)
pressure = (1.5 * y * GF) / denominator
time = np.arange(len(pressure)) / sr
window_duration = 0.02
window_size = int(sr * window_duration)
ste = np.convolve(pressure**2, np.ones(window_size), mode='same')
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(time, pressure, color='teal', alpha=0.5, label='Σήμα Πίεσης')
ax1.set_xlabel('Χρόνος (Δευτερόλεπτα)')
ax1.set_ylabel('Πίεση (μPa)', color='teal')
ax1.tick_params(axis='y', labelcolor='teal')
ax2 = ax1.twinx()
ax2.plot(time, ste, color='darkorange', linewidth=1.5, alpha=0.8, label='Short-Time Energy')
ax2.set_ylabel('Ενέργεια', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
plt.title(f'Σήμα Πίεσης & Ενέργεια Βραχέος Χρόνου - {whale_type}')
fig.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()