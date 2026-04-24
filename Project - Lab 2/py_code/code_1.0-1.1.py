import numpy as np
import warnings
from scipy.io import wavfile

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning) # Απόκρυψη Προειδοποιήσεων για .wav files
fs, signal = wavfile.read('music_dsp2026.wav')
signal = signal.astype(np.float64)

if signal.ndim == 2:
    signal_mono = np.mean(signal, axis=1)
else:
    signal_mono = signal

max_abs_value = np.max(np.abs(signal_mono))

if max_abs_value > 0:
    signal_normalized = signal_mono / max_abs_value
else:
    signal_normalized = signal_mono

N = 512
num_frames = len(signal_normalized) // N
signal_truncated = signal_normalized[:num_frames * N]
frames = signal_truncated.reshape(num_frames, N)

print(f"Μέγεθος STEREO: {signal.shape}")
print(f"Μέγεθος MONO: {signal_mono.shape}")
print(f"Μέγιστη και Ελάχιστη Τιμή - Κανονικοποίηση: {np.max(signal_normalized):.2f} / {np.min(signal_normalized):.2f}")
print(f"Μέγεθος Πίνακα Πλαισίων: {frames.shape} ({num_frames} πλαίσια των {N} δειγμάτων)")

def compute_pk(frames, N=512):
    PN = 90.302
    window = np.hanning(N)
    windowed_frames = frames * window
    X = np.fft.fft(windowed_frames, n=N, axis=1)
    X_single_sided = X[:, : (N // 2) + 1]
    magnitude_squared = np.maximum(np.abs(X_single_sided) ** 2, 1e-12)
    P_k = PN + 10 * np.log10(magnitude_squared)
    return P_k

P_k = compute_pk(frames)

print(f"Διαστάσεις του πίνακα: {P_k.shape}")
print(f"Μέγιστη τιμή (dB SPL): {np.max(P_k):.2f}")
print(f"Ελάχιστη τιμή (dB SPL): {np.min(P_k):.2f}")