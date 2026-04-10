import librosa
import numpy as np

filename = 'Pile driving.wav'
y, sr = librosa.load(filename, sr=None)
GF = 0.16
SENSITIVITY = -125
P_REF = 1.0
denominator = 10 ** (SENSITIVITY / 20)
p = (1.5 * y * GF) / denominator
N = len(p)
E_time = np.sum(p**2)
P_k = np.fft.fft(p)
E_freq = (1 / N) * np.sum(np.abs(P_k)**2)
print(f"{'='*50}\nΕΠΙΒΕΒΑΙΩΣΗ ΘΕΩΡΗΜΑΤΟΣ PARSEVAL\n{'='*50}")
print(f"Ενέργεια στον Χρόνο: {E_time:.2f}")
print(f"Ενέργεια στη Συχνότητα: {E_freq:.2f}")
if np.isclose(E_time, E_freq):
    print("Το θεώρημα Parseval επιβεβαιώνεται\n")
p_rms_sperm = np.sqrt(E_freq / N)
spl_sperm = 20 * np.log10(p_rms_sperm / P_REF)
freqs = np.fft.fftfreq(N, d=1/sr)
band_indices = np.where((np.abs(freqs) >= 300) & (np.abs(freqs) <= 3000))[0]
E_freq_band = (1 / N) * np.sum(np.abs(P_k[band_indices])**2)
p_rms_humpback = np.sqrt(E_freq_band / N)
spl_humpback = 20 * np.log10(p_rms_humpback / P_REF)
print(f"{'='*50}\nΑΠΟΤΕΛΕΣΜΑΤΑ ΥΠΟΛΟΓΙΣΜΟΥ SPL ΜΕ PARSEVAL\n{'='*50}")
print(f"Sperm Whale: SPL = {spl_sperm:.2f} dB")
print(f"Humpback Whale: SPL = {spl_humpback:.2f} dB")