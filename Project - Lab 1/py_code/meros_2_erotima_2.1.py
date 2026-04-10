import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

file_path = 'speech_utterance.wav'
signal, fs = sf.read(file_path)

N_total = len(signal)
time_axis = np.arange(N_total) / fs
win_len_ms = 30 # 20 < 30 < 50
win_len_samples = int((win_len_ms / 1000) * fs)
num_frames = N_total // win_len_samples
ste = np.zeros(num_frames)
zcr = np.zeros(num_frames)
frame_time_axis = np.zeros(num_frames)

for i in range(num_frames):
    start = i * win_len_samples
    end = start + win_len_samples
    frame = signal[start:end]
    frame_time_axis[i] = (start + end) / 2 / fs
    ste[i] = np.sum(frame ** 2)
    zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * win_len_samples)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Voice Signal
ax1.plot(time_axis, signal, color='teal', linewidth=0.8)
ax1.set_title('Αρχικό Σήμα Φωνής')
ax1.set_ylabel('Πλάτος')
ax1.grid(True, linestyle='--', alpha=0.7)

# STE
ax2.plot(frame_time_axis, ste, color='darkorange', linewidth=1.5, drawstyle='steps-mid')
ax2.set_title(f'Ενέργεια Βραχέος Χρόνου (STE) - Παράθυρο {win_len_ms} ms')
ax2.set_ylabel('Ενέργεια')
ax2.grid(True, linestyle='--', alpha=0.7)

# ZCR
ax3.plot(frame_time_axis, zcr, color='purple', linewidth=1.5, drawstyle='steps-mid')
ax3.set_title(f'Ρυθμός Εναλλαγής Προσήμου (ZCR) - Παράθυρο {win_len_ms} ms')
ax3.set_ylabel('ZCR')
ax3.set_xlabel('Χρόνος (δευτερόλεπτα)')
ax3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()