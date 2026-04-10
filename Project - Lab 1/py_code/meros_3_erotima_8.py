import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def load_and_get_pressure(filename):
    y, sr = librosa.load(filename, sr=None)
    GF = 0.9
    SENSITIVITY = -165
    denominator = 10 ** (SENSITIVITY / 20)
    pressure = (1.5 * y * GF) / denominator
    return pressure, sr
p_hump, sr_hump = load_and_get_pressure('63019002.wav')
p_sperm, sr_sperm = load_and_get_pressure('91003005.wav')
f_hump, psd_hump = welch(p_hump, fs=sr_hump, nperseg=4096)
f_sperm, psd_sperm = welch(p_sperm, fs=sr_sperm, nperseg=4096)
psd_hump_db = 10 * np.log10(np.maximum(psd_hump, 1e-12))
psd_sperm_db = 10 * np.log10(np.maximum(psd_sperm, 1e-12))
plt.figure(figsize=(12, 6))
plt.semilogx(f_hump, psd_hump_db, label='Humpback Whale', color='teal', linewidth=1.5, alpha=0.8)
plt.semilogx(f_sperm, psd_sperm_db, label='Sperm Whale', color='darkorange', linewidth=1.5, alpha=0.8)
plt.title('Ερώτημα 3.8: Εκτίμηση Φασματικής Πυκνότητας')
plt.xlabel('Συχνότητα (Hz) - Λογαριθμική Κλίμακα')
plt.ylabel('PSD (dB re $1 \mu Pa^2/Hz$)')
plt.legend(loc='upper right')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()