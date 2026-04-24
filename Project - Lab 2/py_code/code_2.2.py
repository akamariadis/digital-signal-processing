import numpy as np
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
    num_frames, num_maskers, num_bins = T_TM.shape
    f_safe = np.maximum(f_matched, 20.0)
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


def calculate_bit_allocation(T_g_frame, R_bits=16):
    M = 32
    B_k = np.zeros(M, dtype=int)
    R = 2 ** R_bits  # 65536

    # Το μέγιστο θεωρητικό dB SPL στο μοντέλο μας (FFT χωρίς κανονικοποίηση 1/N)
    max_spl = 132.446

    for k in range(M):
        T_g_subband_dB = T_g_frame[k * 8: (k + 1) * 8]
        min_Tg_dB = np.min(T_g_subband_dB)
        min_Tg_norm = 10 ** ((min_Tg_dB - max_spl) / 20.0)
        min_Tg_level = min_Tg_norm * (R / 2)
        min_Tg_level = max(min_Tg_level, 1e-10)
        bits = int(np.log2(R / min_Tg_level) - 1)
        B_k[k] = np.clip(bits, 0, R_bits)

    return B_k

def adaptive_quantizer(y_frame, B_k):
    M = len(B_k)
    y_quantized = np.zeros_like(y_frame)

    for k in range(M):
        bits = B_k[k]
        y_k = y_frame[k, :]
        if bits == 0:
            y_quantized[k, :] = 0
            continue
        x_min, x_max = np.min(y_k), np.max(y_k)
        if x_max == x_min:
            y_quantized[k, :] = y_k
            continue
        levels = 2 ** bits
        delta = (x_max - x_min) / levels
        indices = np.round((y_k - x_min) / delta)
        indices = np.clip(indices, 0, levels - 1)
        y_quantized[k, :] = x_min + indices * delta + (delta / 2)
    return y_quantized


def fixed_quantizer(y_frame, B=8):
    y_quantized = np.zeros_like(y_frame)
    x_min, x_max = -1.0, 1.0
    levels = 2 ** B
    delta = (x_max - x_min) / levels
    for k in range(y_frame.shape[0]):
        y_k = y_frame[k, :]
        y_clipped = np.clip(y_k, x_min, x_max)
        indices = np.round((y_clipped - x_min) / delta)
        indices = np.clip(indices, 0, levels - 1)
        y_quantized[k, :] = x_min + indices * delta + (delta / 2)
    return y_quantized


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
    T_TM = compute_individual_thresholds(P_TMc, z, is_tonal=True)
    T_NM = compute_individual_thresholds(P_NMc, z, is_tonal=False)
    T_g, T_q = compute_global_masking_threshold(T_TM, T_NM, f_matched)
    M = 32
    h, g = create_filterbank(M)
    y_subbands = subband_analysis(signal_normalized, h, M)
    frame_idx = 10
    start_sample = frame_idx * (N // M)
    end_sample = start_sample + (N // M)
    y_frame = y_subbands[:, start_sample:end_sample]
    B_k = calculate_bit_allocation(T_g[frame_idx], R_bits=16)
    y_adaptive_quant = adaptive_quantizer(y_frame, B_k)
    y_fixed_quant = fixed_quantizer(y_frame, B=8)
    print(f"--- Πλαίσιο {frame_idx} ---")
    print(f"Κατανομή Bits (Προσαρμοζόμενος Κβαντιστής):\n{B_k}")
    print(f"\nΣυνολικά bits πλαισίου (Προσαρμοζόμενος): {np.sum(B_k)} από τα 512 αρχικά (32x16)")
    print(f"Συνολικά bits πλαισίου (Σταθερός): {32 * 8} από τα 512 αρχικά")