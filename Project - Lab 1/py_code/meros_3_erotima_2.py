import librosa
import numpy as np
import matplotlib.pyplot as plt

GF = 0.9
sensitivity = -165

def process_and_plot_whale_signal(filename, whale_type):
    y, sr = librosa.load(filename, sr=None)
    denominator = 10 ** (sensitivity / 20)
    pressure = (1.5 * y * GF) / denominator
    def calculate_rms_and_spl(pressure, whale_type):
        # p rms
        p_rms = np.sqrt(np.mean(pressure ** 2))
        # SPL rms
        p_ref = 1.0
        spl_rms = 20 * np.log10(p_rms / p_ref)
        print(f"--- Αποτελέσματα για {whale_type} ---")
        print(f"P_rms: {p_rms:.6f} μPa")
        print(f"SPL_RMS: {spl_rms:.2f} dB relative to 1 μPa\n")
        return p_rms, spl_rms
    denominator = 10 ** (sensitivity / 20)
    pressure = (1.5 * y * GF) / denominator
    calculate_rms_and_spl(pressure, whale_type)
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

def calculate_rms_and_spl(pressure, whale_type):
    #p rms
    p_rms = np.sqrt(np.mean(pressure ** 2))
    # SPL rms
    p_ref = 1.0
    spl_rms = 20 * np.log10(p_rms / p_ref)
    print(f"--- Αποτελέσματα για {whale_type} ---")
    print(f"P_rms: {p_rms:.6f} μPa")
    print(f"SPL_RMS: {spl_rms:.2f} dB relative to 1 μPa\n")
    return p_rms, spl_rms