import numpy as np
import soundfile as sf

file_path = 'speech_utterance.wav'

signal, fs = sf.read(file_path)

win_len_ms = 30
win_len_samples = int((win_len_ms / 1000) * fs)
num_frames = len(signal) // win_len_samples

ste = np.zeros(num_frames)
zcr = np.zeros(num_frames)

for i in range(num_frames):
    start = i * win_len_samples
    end = start + win_len_samples
    frame = signal[start:end]
    ste[i] = np.sum(frame ** 2)

    zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * win_len_samples)

print(f"Χωρίστηκε σε: {num_frames} παράθυρα (των {win_len_ms} ms)")

print("Στατιστικά Ενέργειας Βραχέος Χρόνου (STE)")
print(f"Μέγιστη Ενέργεια: {np.max(ste):.4f}")
print(f"Ελάχιστη Ενέργεια: {np.min(ste):.4f}")
print(f"Μέση Ενέργεια του σήματος : {np.mean(ste):.4f}\n")

print("Στατιστικά Ρυθμού Εναλλαγής Προσήμου (ZCR)")
print(f"Μέγιστο ZCR: {np.max(zcr):.4f}")
print(f"Ελάχιστο ZCR: {np.min(zcr):.4f}")
print(f"Μέσο ZCR του σήματος: {np.mean(zcr):.4f}\n")