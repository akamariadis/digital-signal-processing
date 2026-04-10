import librosa
import numpy as np
from scipy.signal import butter, filtfilt

filename = 'Pile driving.wav'
y, sr = librosa.load(filename, sr=None)
GF = 0.16
SENSITIVITY = -125
P_REF = 1.0
denominator = 10 ** (SENSITIVITY / 20)
pressure = (1.5 * y * GF) / denominator
p_rms_sperm = np.sqrt(np.mean(pressure**2))
spl_rms_sperm = 20 * np.log10(p_rms_sperm / P_REF)
lowcut = 300.0
highcut = 3000.0
order = 3
b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=sr)
pressure_filtered = filtfilt(b, a, pressure)
p_rms_humpback = np.sqrt(np.mean(pressure_filtered**2))
spl_rms_humpback = 20 * np.log10(p_rms_humpback / P_REF)
print(f"{'='*50}\nΑΝΑΛΥΣΗ ΕΠΙΠΤΩΣΕΩΝ ΘΟΡΥΒΟΥ: {filename}\n{'='*50}")

print("\n1. Sperm Whales:")
print(f"SPL_rms: {spl_rms_sperm:.2f} dB re 1 μPa")
if spl_rms_sperm > 100:
    print("ΣΥΜΠΕΡΑΣΜΑ: Η συμπεριφορά τους ΕΠΗΡΕΑΖΕΤΑΙ (> 100 dB).")
else:
    print("ΣΥΜΠΕΡΑΣΜΑ: Δεν επηρεάζονται.")

print("\n2. Humpback Whales:")
print(f"SPL_rms: {spl_rms_humpback:.2f} dB re 1 μPa")
if spl_rms_humpback > 100:
    print("ΣΥΜΠΕΡΑΣΜΑ: Η συμπεριφορά τους ΕΠΗΡΕΑΖΕΤΑΙ (> 100 dB).")
else:
    print("ΣΥΜΠΕΡΑΣΜΑ: Δεν επηρεάζονται.")