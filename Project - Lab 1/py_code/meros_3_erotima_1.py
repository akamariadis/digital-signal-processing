import librosa
import numpy as np
import matplotlib.pyplot as plt

GF = 0.9
sensitivity = -165

def process_and_plot_whale_signal(filename, whale_type):
    y, sr = librosa.load(filename, sr=None)
    denominator = 10 ** (sensitivity / 20)
    pressure = (1.5 * y * GF) / denominator
    time = np.arange(len(pressure)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(time, pressure, color='teal', linewidth=0.5)
    plt.title(f'Διάγραμμα Πίεσης-Χρόνου: {whale_type}')
    plt.xlabel('Χρόνος (Δευτερόλεπτα)')
    plt.ylabel('Πίεση (μPa)')  # Μονάδα μέτρησης βάσει του 1V/μPa
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


process_and_plot_whale_signal('91003005.wav', 'Sperm Whale')
process_and_plot_whale_signal('63019002.wav', 'Humpback Whale')