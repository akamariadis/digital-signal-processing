import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display

file_path = 'music.wav'

signal, fs = sf.read(file_path)

win_len_ms = 30
win_length = int((win_len_ms / 1000) * fs)
hop_length = win_length // 2
n_fft = 2048

D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hamming')
S = np.abs(D)
S_db = librosa.amplitude_to_db(S, ref=np.max)

centroid = librosa.feature.spectral_centroid(S=S, sr=fs, n_fft=n_fft, hop_length=hop_length)[0]

diff = np.diff(S, axis=1)
flux = np.sum(diff ** 2, axis=0)
flux = np.insert(flux, 0, 0)

times = librosa.frames_to_time(range(S.shape[1]), sr=fs, hop_length=hop_length)

fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

img = librosa.display.specshow(S_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax[0],
                                   cmap='magma')
ax[0].set_title(f'Φασματογράφημα Μουσικής (Λογαριθμική Κλίμακα, Παράθυρο={win_len_ms}ms, N={n_fft})')
fig.colorbar(img, ax=ax[0], format="%+2.0f dB")

ax[1].plot(times, centroid, color='cyan', linewidth=2)
ax[1].set_title('Φασματικό Κέντρο (Spectral Centroid) - Μουσική')
ax[1].set_ylabel('Συχνότητα (Hz)')
ax[1].grid(True, linestyle='--', alpha=0.7)

ax[2].plot(times, flux, color='lime', linewidth=2)
ax[2].set_title('Φασματική Ροή (Spectral Flux) - Μουσική')
ax[2].set_xlabel('Χρόνος (δευτερόλεπτα)')
ax[2].set_ylabel('Τιμή Ροής')
ax[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()