import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot_mel_vs_pcen(filename, whale_type):
    y, sr = librosa.load(filename, sr=None)
    GF = 0.9
    SENSITIVITY = -165
    denominator = 10 ** (SENSITIVITY / 20)
    pressure = (1.5 * y * GF) / denominator
    M = librosa.feature.melspectrogram(y=pressure, sr=sr, power=1)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    M_pcen = librosa.pcen(M, sr=sr)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
    img1 = librosa.display.specshow(M_db, x_axis='time', y_axis='mel', sr=sr, ax=ax[0], cmap='magma')
    ax[0].set(title=f'1. Κλασικό Mel-Spectrogram (dB) - {whale_type}')
    fig.colorbar(img1, ax=[ax[0]], format="%+2.0f dB")
    img2 = librosa.display.specshow(M_pcen, x_axis='time', y_axis='mel', sr=sr, ax=ax[1], cmap='magma')
    ax[1].set(title=f'2. PCEN Mel-Spectrogram (Ενισχυμένη Αντίθεση) - {whale_type}')
    fig.colorbar(img2, ax=[ax[1]])
    plt.xlabel('Χρόνος (Δευτερόλεπτα)')
    plt.tight_layout()
    plt.show()
plot_mel_vs_pcen('63019002.wav', 'Humpback Whale')
plot_mel_vs_pcen('91003005.wav', 'Sperm Whale')