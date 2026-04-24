import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

def create_filterbank(M=32):
    L = 2 * M
    h = np.zeros((M, L))
    g = np.zeros((M, L))
    n = np.arange(L)
    window = np.sin((n + 0.5) * (np.pi / (2 * M)))
    for k in range(M):
        cosine_term = np.cos((2 * n + M + 1) * (2 * k + 1) * np.pi / (4 * M))
        h[k, :] = window * np.sqrt(2 / M) * cosine_term
        g[k, :] = h[k, ::-1]
    return h, g

def subband_analysis(signal, h, M=32):
    y = []
    for k in range(M):
        v_k = np.convolve(signal, h[k, :], mode='full')
        y_k = v_k[::M]
        y.append(y_k)
    return np.array(y)

if __name__ == "__main__":
    fs, signal = wavfile.read('music_dsp2026.wav')
    signal = signal.astype(np.float64)
    signal_mono = np.mean(signal, axis=1) if signal.ndim == 2 else signal
    max_abs = np.max(np.abs(signal_mono))
    signal_normalized = signal_mono / max_abs if max_abs > 0 else signal_mono
    M = 32
    h, g = create_filterbank(M)
    print(f"Δημιουργήθηκαν {h.shape[0]} φίλτρα ανάλυσης, μήκους {h.shape[1]} δειγμάτων το καθένα.")
    y_subbands = subband_analysis(signal_normalized, h, M)
    print(f"Διαστάσεις εξόδου Y_k(n): {y_subbands.shape}")

    plt.figure(figsize=(12, 6))
    N_fft = 1024
    f_axis = fftfreq(N_fft, d=1 / fs)[:N_fft // 2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for k in range(4):
        H_k = fft(h[k, :], n=N_fft)
        H_k_mag = np.abs(H_k[:N_fft // 2])
        H_k_dB = 20 * np.log10(np.maximum(H_k_mag, 1e-10))
        plt.plot(f_axis, H_k_dB, label=f'Ζώνη $k={k}$', color=colors[k], linewidth=2)
        plt.fill_between(f_axis, -100, H_k_dB, color=colors[k], alpha=0.3)

    plt.title('Απόκριση Συχνότητας των 4 Πρώτων Ζωνοπερατών Φίλτρων (MDCT)', fontsize=16, fontweight='bold')
    plt.xlabel('Συχνότητα (Hz)', fontsize=14)
    plt.ylabel('Πλάτος (dB)', fontsize=14)
    plt.xlim(0, 4000)
    plt.ylim(-60, 5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()