import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

def compute_power_spectrum(frames, fs, N=512):
    PN = 90.302
    n = np.arange(N)
    window = 0.5 * (1.0 - np.cos(2 * np.pi * n / N))
    windowed_frames = frames * window
    X = np.fft.fft(windowed_frames, n=N, axis=1)
    X_single_sided = X[:, : (N // 2) + 1]
    magnitude_squared = np.maximum(np.abs(X_single_sided) ** 2, 1e-12)
    P_k = PN + 10 * np.log10(magnitude_squared)
    f = np.arange((N // 2) + 1) * (fs / N)
    return f, P_k

if __name__ == "__main__":
    fs, signal = wavfile.read('music_dsp2026.wav')
    signal = signal.astype(np.float64)
    signal_mono = np.mean(signal, axis=1) if signal.ndim == 2 else signal
    max_abs = np.max(np.abs(signal_mono))
    signal_normalized = signal_mono / max_abs if max_abs > 0 else signal_mono
    N = 512
    num_frames = len(signal_normalized) // N
    frames = signal_normalized[:num_frames * N].reshape(num_frames, N)
    f, P_k = compute_power_spectrum(frames, fs, N)
    P_TMc = np.load('P_TMc-26.npy').T
    P_NMc = np.load('P_NMc-26.npy').T
    print(f"Διαστάσεις P_k: {P_k.shape}")
    print(f"Διαστάσεις P_TMc μετά την αναστροφή: {P_TMc.shape}")
    frame_to_plot = 10
    plt.figure(figsize=(14, 7))
    plt.plot(f, P_k[frame_to_plot], label='Φάσμα Ισχύος $P(k)$', color='#1f77b4', linewidth=1.5, alpha=0.9)
    masker_indices_c = np.where(P_TMc[frame_to_plot] > 0)[0]
    if len(masker_indices_c) > 0:
        plt.scatter(f[masker_indices_c], P_TMc[frame_to_plot, masker_indices_c],
                    color='red', s=80, edgecolor='black', zorder=5, label='Τονικές Μάσκες ($P_{TMc}$)')
    noise_indices_c = np.where(P_NMc[frame_to_plot] > 0)[0]
    if len(noise_indices_c) > 0:
        plt.scatter(f[noise_indices_c], P_NMc[frame_to_plot, noise_indices_c],
                    color='limegreen', marker='X', s=80, edgecolor='black', zorder=5,
                    label='Μάσκες Θορύβου ($P_{NMc}$)')
    plt.title(f'Μειωμένες & Αναδιοργανωμένες Μάσκες (Πλαίσιο {frame_to_plot})', fontsize=16, fontweight='bold')
    plt.xlabel('Συχνότητα f (Hz)', fontsize=14)
    plt.ylabel('Ισχύς (dB SPL)', fontsize=14)
    plt.xlim(0, 10000)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()