import numpy as np
import matplotlib.pyplot as plt

N = 1000
n = np.arange(N)
silence_len = 100

frequencies = {
    '1': (0.5346, 0.9273), '2': (0.5346, 1.0247), '3': (0.5346, 1.1328),
    '4': (0.5906, 0.9273), '5': (0.5906, 1.0247), '6': (0.5906, 1.1328),
    '7': (0.6535, 0.9273), '8': (0.6535, 1.0247), '9': (0.6535, 1.1328),
    '0': (0.7217, 1.0247)
}

AM1 = 3122674
AM2 = 3123434
digits_str = str(AM1 + AM2).zfill(8)

sequence_parts = []
silence = np.zeros(silence_len)

for i, digit in enumerate(digits_str):
    omega_row, omega_col = frequencies[digit]
    tone = np.sin(omega_row * n) + np.sin(omega_col * n)
    sequence_parts.append(tone)
    if i < len(digits_str) - 1:
        sequence_parts.append(silence)

final_signal = np.concatenate(sequence_parts)
L = 1000
N_fft = 1024
num_tones = len(digits_str)
rect_window = np.ones(L)
hamm_window = np.hamming(L)
dft_rect_list = []
dft_hamm_list = []

print("Υπολογισμός DFT για κάθε ψηφίο...")

for i in range(num_tones):
    start_idx = i * (L + silence_len)
    end_idx = start_idx + L
    segment = final_signal[start_idx:end_idx]
    segment_rect = segment * rect_window
    segment_hamm = segment * hamm_window
    D_rect = np.fft.fft(segment_rect, n=N_fft)
    D_hamm = np.fft.fft(segment_hamm, n=N_fft)
    dft_rect_list.append(D_rect)
    dft_hamm_list.append(D_hamm)

k = np.arange(N_fft)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(k, np.abs(dft_rect_list[0]), color='blue')
plt.title(f'Μέτρο DFT 1ου ψηφίου ({digits_str[0]}) - Ορθογώνιο Παράθυρο')
plt.ylabel('Πλάτος')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(k, np.abs(dft_hamm_list[0]), color='orange')
plt.title(f'Μέτρο DFT 1ου ψηφίου ({digits_str[0]}) - Παράθυρο Hamming')
plt.ylabel('Πλάτος')
plt.xlabel('Δείκτης Συχνότητας (k)')
plt.grid(True)

plt.tight_layout()
plt.show()