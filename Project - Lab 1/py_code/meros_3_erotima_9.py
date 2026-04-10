import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot_spectrogram(filename, whale_type):
    y, sr = librosa.load(filename, sr=None)
    GF = 0.9
    SENSITIVITY = -165
    denominator = 10 ** (SENSITIVITY / 20)
    pressure = (1.5 * y * GF) / denominator
    D = librosa.stft(pressure)
    amplitude = np.abs(D)
    S_db = librosa.amplitude_to_db(amplitude, ref=np.max)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB', label='Ένταση (dB)')
    plt.title(f'Ερώτημα 3.9: Φασματογράφημα (Spectrogram) - {whale_type}')
    plt.xlabel('Χρόνος (Δευτερόλεπτα)')
    plt.ylabel('Συχνότητα (Hz)')
    plt.tight_layout()
    plt.show()
plot_spectrogram('63019002.wav', 'Humpback Whale')
plot_spectrogram('91003005.wav', 'Sperm Whale')