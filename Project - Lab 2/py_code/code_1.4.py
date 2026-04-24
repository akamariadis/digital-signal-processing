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
    T_TM = compute_individual_thresholds(P_TMc, z, is_tonal=True)
    T_NM = compute_individual_thresholds(P_NMc, z, is_tonal=False)
    frame_to_plot = 10
    plt.figure(figsize=(14, 7))
    plt.plot(f, P_k[frame_to_plot], label='Φάσμα Ισχύος $P(k)$', color='#1f77b4', linewidth=1.5, alpha=0.7)
    masker_indices_c = np.where(P_TMc[frame_to_plot] > 0)[0]
    if len(masker_indices_c) > 0:
        target_j = masker_indices_c[0]
        plt.scatter(f[target_j], P_TMc[frame_to_plot, target_j],
                    color='red', s=100, edgecolor='black', zorder=5, label='Συγκεκριμένη Τονική Μάσκα')
        T_curve = T_TM[frame_to_plot, target_j, :]
        f_matched = f[:len(T_curve)]
        valid_indices = T_curve > -100
        plt.plot(f_matched[valid_indices], T_curve[valid_indices],
                 color='darkorange', linewidth=3, linestyle='--', zorder=4,
                 label='Ατομικό Κατώφλι Συγκάλυψης $T_{TM}(i, j)$')
        plt.fill_between(f_matched[valid_indices], -20, T_curve[valid_indices],
                         color='orange', alpha=0.2)
    plt.title(f'Εφαρμογή Συνάρτησης Διασποράς (Spreading Function) - Πλαίσιο {frame_to_plot}', fontsize=16,
              fontweight='bold')
    plt.xlabel('Συχνότητα f (Hz)', fontsize=14)
    plt.ylabel('Ισχύς (dB SPL)', fontsize=14)
    plt.xlim(0, 8000)  # Κάνουμε ζουμ στις πρώτες συχνότητες
    plt.ylim(-10, 130)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()