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
AM1 = 3122674
AM2 = 3123434
total_sum = AM1 + AM2
digits_str = str(total_sum).zfill(8)

print(f"Το άθροισμα των ΑΜ που θα μεταφραστεί σε τόνους είναι: {digits_str}")

silence = np.zeros(100)
sequence_parts = []

for i, digit in enumerate(digits_str):
    omega_row, omega_col = frequencies[digit]
    tone = np.sin(omega_row * n) + np.sin(omega_col * n)
    sequence_parts.append(tone)
    if i < len(digits_str) - 1:
        sequence_parts.append(silence)

final_signal = np.concatenate(sequence_parts)

final_signal_normalized = np.float32(final_signal / np.max(np.abs(final_signal)))

wavfile.write("tone_sequence.wav", fs, final_signal_normalized)
print("Το αρχείο 'tone_sequence.wav' δημιουργήθηκε επιτυχώς στον φάκελο του project!")