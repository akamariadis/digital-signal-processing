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
p_minus_1 = np.roll(pressure, 1)
p_plus_1 = np.roll(pressure, -1)
p_minus_1[0] = pressure[0]
p_plus_1[-1] = pressure[-1]
tkeo = (pressure ** 2) - (p_minus_1 * p_plus_1)
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(time, pressure, color='teal', alpha=0.5, label='Σήμα Πίεσης')
ax1.set_xlabel('Χρόνος (Δευτερόλεπτα)')
ax1.set_ylabel('Πίεση (μPa)', color='teal')
ax1.tick_params(axis='y', labelcolor='teal')
ax2 = ax1.twinx()
ax2.plot(time, tkeo, color='purple', linewidth=1.5, alpha=0.8, label='Teager-Kaiser Energy')
ax2.set_ylabel('Ενέργεια Teager-Kaiser', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
plt.title(f'Ερώτημα 3.7: Σήμα Πίεσης & Teager-Kaiser Energy - {whale_type}')
fig.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()