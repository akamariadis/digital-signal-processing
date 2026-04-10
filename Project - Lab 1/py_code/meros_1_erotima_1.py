import numpy as np
from scipy.io import wavfile

N = 1000
fs = 8192
n = np.arange(N)

frequencies = {
    '1': (0.5346, 0.9273),
    '2': (0.5346, 1.0247),
    '3': (0.5346, 1.1328),
    '4': (0.5906, 0.9273),
    '5': (0.5906, 1.0247),
    '6': (0.5906, 1.1328),
    '7': (0.6535, 0.9273),
    '8': (0.6535, 1.0247),
    '9': (0.6535, 1.1328),
    '0': (0.7217, 1.0247)
}

print("Ξεκινάει η δημιουργία και αποθήκευση των αρχείων .wav\n")

for key, (omega_row, omega_col) in frequencies.items():
    signal = np.sin(omega_row * n) + np.sin(omega_col * n)
    normalized_signal = np.float32(signal / np.max(np.abs(signal)))
    filename = f"tone_{key}.wav"
    wavfile.write(filename, fs, normalized_signal)
    print(f"✔️ Αποθηκεύτηκε επιτυχώς το αρχείο: {filename}")

print("\nΗ διαδικασία ολοκληρώθηκε! Μπορείς να βρεις τα αρχεία στον φάκελο του project σου.")







