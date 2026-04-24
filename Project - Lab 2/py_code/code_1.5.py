import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

def compute_individual_thresholds(P_maskers, bark_scale, is_tonal=True):
    num_frames, num_bins = P_maskers.shape
    bark_matched = bark_scale[:num_bins]
    T = np.full((num_frames, num_bins, num_bins), -np.inf)
    for frame in range(num_frames):
        masker_indices = np.where(P_maskers[frame] > 0)[0]
        for j in masker_indices:
            P_j = P_maskers[frame, j]
            b_j = bark_matched[j]
            dz = bark_matched - b_j
            SF = np.full_like(dz, -np.inf)
            m1 = (dz >= -3) & (dz < -1)
            SF[m1] = 17 * dz[m1] - 0.4 * P_j + 11
            m2 = (dz >= -1) & (dz < 0)
            SF[m2] = (0.4 * P_j + 6) * dz[m2]
            m3 = (dz >= 0) & (dz < 1)
            SF[m3] = -17 * dz[m3]
            m4 = (dz >= 1) & (dz < 8)
            SF[m4] = (0.15 * P_j - 17) * dz[m4] - 0.15 * P_j
            if is_tonal:
                T[frame, j, :] = P_j - 0.275 * b_j + SF - 6.025
            else:
                T[frame, j, :] = P_j - 0.175 * b_j + SF - 2.025
    return T

def compute_global_masking_threshold(T_TM, T_NM, f_matched):
    f_safe = np.maximum(f_matched, 1e-6)
    f_kHz = f_safe / 1000.0
    T_q = 3.64 * (f_kHz ** -0.8) - 6.5 * np.exp(-0.6 * (f_kHz - 3.3) ** 2) + (10 ** -3) * (f_kHz ** 4)
    power_q = 10 ** (0.1 * T_q)
    power_TM = 10 ** (0.1 * T_TM)
    power_NM = 10 ** (0.1 * T_NM)
    sum_power_TM = np.sum(power_TM, axis=1)
    sum_power_NM = np.sum(power_NM, axis=1)
    total_power = power_q + sum_power_TM + sum_power_NM
    T_g = 10 * np.log10(total_power)
    return T_g, T_q

if __name__ == "__main__":
    fs, signal = wavfile.read('music_dsp2026.wav')
    signal = signal.astype(np.float64)
    signal_mono = np.mean(signal, axis=1) if signal.ndim == 2 else signal
    max_abs = np.max(np.abs(signal_mono))
    signal_normalized = signal_mono / max_abs if max_abs > 0 else signal_mono
    N = 512
    num_frames = len(signal_normalized) // N
    frames = signal_normalized[:num_frames * N].reshape(num_frames, N)
    PN = 90.302
    n = np.arange(N)
    window = 0.5 * (1.0 - np.cos(2 * np.pi * n / N))
    windowed_frames = frames * window
    X = np.fft.fft(windowed_frames, n=N, axis=1)
    X_single_sided = X[:, : (N // 2) + 1]
    magnitude_squared = np.maximum(np.abs(X_single_sided) ** 2, 1e-12)
    P_k = PN + 10 * np.log10(magnitude_squared)
    k_bins = np.arange((N // 2) + 1)
    f = k_bins * (fs / N)
    z = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
    P_TMc = np.load('P_TMc-26.npy').T
    P_NMc = np.load('P_NMc-26.npy').T
    f_matched = f[:P_TMc.shape[1]]
    P_k_matched = P_k[:, :P_TMc.shape[1]]
    T_TM = compute_individual_thresholds(P_TMc, z, is_tonal=True)
    T_NM = compute_individual_thresholds(P_NMc, z, is_tonal=False)
    T_g, T_q = compute_global_masking_threshold(T_TM, T_NM, f_matched)
    frame_to_plot = 10
    plt.figure(figsize=(15, 8))
    plt.plot(f_matched, P_k_matched[frame_to_plot], label='Φάσμα Ισχύος $P(k)$', color='#1f77b4', linewidth=1.2,
             alpha=0.8)
    plt.plot(f_matched, T_q, label='Κατώφλι Ακοής $T_q(f)$', color='green', linestyle=':', linewidth=2)
    plt.plot(f_matched, T_g[frame_to_plot], label='Ολικό Κατώφλι Συγκάλυψης $T_g(i)$', color='red', linewidth=2.5)
    plt.fill_between(f_matched, -20, T_g[frame_to_plot], color='red', alpha=0.15,
                     label='Περιοχή Συγκάλυψης (Μη ακουστή)')
    plt.title(f'Ψυχοακουστικό Μοντέλο: Ολικό Κατώφλι Συγκάλυψης - Πλαίσιο {frame_to_plot}', fontsize=16,
              fontweight='bold')
    plt.xlabel('Συχνότητα f (Hz)', fontsize=14)
    plt.ylabel('Ισχύς (dB SPL)', fontsize=14)
    plt.xlim(0, 10000)
    plt.ylim(-10, 130)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()