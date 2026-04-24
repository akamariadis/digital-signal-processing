import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings

data = np.load('P_NM-26.npy')

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

def compute_power_spectrum(frames, fs, N=512):
    PN = 90.302
    window = np.hanning(N)
    windowed_frames = frames * window
    X = np.fft.fft(windowed_frames, n=N, axis=1)
    X_single_sided = X[:, : (N // 2) + 1]
    magnitude_squared = np.maximum(np.abs(X_single_sided) ** 2, 1e-12)
    P_k = PN + 10 * np.log10(magnitude_squared)
    k = np.arange((N // 2) + 1)
    f = k * (fs / N)
    return f, P_k

def compute_tonal_maskers(P_k):
    num_frames, num_bins = P_k.shape
    S_T = np.zeros_like(P_k, dtype=bool)
    P_TM = np.zeros_like(P_k, dtype=float)

    for k in range(2, 250):
        is_local_max = (P_k[:, k] > P_k[:, k - 1]) & (P_k[:, k] > P_k[:, k + 1])
        if 2 <= k < 63:
            deltas = [2]
        elif 63 <= k < 127:
            deltas = range(2, 4)
        elif 127 <= k < 250:
            deltas = range(2, 7)
        is_7db_greater = np.ones(num_frames, dtype=bool)
        for delta in deltas:
            is_7db_greater &= (P_k[:, k] > (P_k[:, k - delta] + 7))
            is_7db_greater &= (P_k[:, k] > (P_k[:, k + delta] + 7))
        S_T[:, k] = is_local_max & is_7db_greater
        power_sum_linear = (10 ** (0.1 * P_k[:, k - 1])) + \
                           (10 ** (0.1 * P_k[:, k])) + \
                           (10 ** (0.1 * P_k[:, k + 1]))
        P_TM[:, k] = np.where(S_T[:, k], 10 * np.log10(power_sum_linear), 0.0)
    return S_T, P_TM


fs, signal = wavfile.read('music_dsp2026.wav')
signal = signal.astype(np.float64)
signal_mono = np.mean(signal, axis=1) if signal.ndim == 2 else signal
max_abs = np.max(np.abs(signal_mono))
signal_normalized = signal_mono / max_abs if max_abs > 0 else signal_mono
N = 512
num_frames = len(signal_normalized) // N
frames = signal_normalized[:num_frames * N].reshape(num_frames, N)
f, P_k = compute_power_spectrum(frames, fs, N)
S_T, P_TM = compute_tonal_maskers(P_k)
frame_to_plot = 10
plt.figure(figsize=(12, 6))
plt.plot(f, P_k[frame_to_plot], label='Φάσμα Ισχύος $P(k)$', color='C0', linewidth=1.5)
masker_indices = np.where(S_T[frame_to_plot])[0]
masker_freqs = f[masker_indices]
masker_powers = P_TM[frame_to_plot, masker_indices]
plt.scatter(masker_freqs, masker_powers, color='red', s=50, zorder=5,
            label='Τονικές Μάσκες ($P_{TM}$)')
plt.title(f'Φασματική Ανάλυση & Τονικές Μάσκες (Πλαίσιο {frame_to_plot})', fontsize=14)
plt.xlabel('Συχνότητα (Hz)', fontsize=12)
plt.ylabel('Ισχύς (dB SPL)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xlim(0, 10000)
plt.tight_layout()
plt.show()