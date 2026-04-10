import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

N = 1000
N_fft = 1024
n = np.arange(N)

freqs = {
    '2': (0.5346, 1.0247),
    '3': (0.5346, 1.1328),
    '6': (0.5906, 1.1328)
}

# time domain signals
d2 = np.sin(freqs['2'][0] * n) + np.sin(freqs['2'][1] * n)
d3 = np.sin(freqs['3'][0] * n) + np.sin(freqs['3'][1] * n)
d6 = np.sin(freqs['6'][0] * n) + np.sin(freqs['6'][1] * n)

# frequency domain signals (DFT)
D2 = fft(d2, n=N_fft)
D3 = fft(d3, n=N_fft)
D6 = fft(d6, n=N_fft)

# magnitude of DFT
D2_mag = np.abs(D2)
D3_mag = np.abs(D3)
D6_mag = np.abs(D6)
k = np.arange(N_fft)
plt.figure(figsize=(10, 8))

# |D_2[k]| Plot
plt.subplot(3, 1, 1)
plt.plot(k, D2_mag, color='b')
plt.title('Μέτρο DFT του d_2[n]: $|D_2[k]|$')
plt.ylabel('Πλάτος')
plt.grid(True)

# |D_3[k]| Plot
plt.subplot(3, 1, 2)
plt.plot(k, D3_mag, color='g')
plt.title('Μέτρο DFT του d_3[n]: $|D_3[k]|$')
plt.ylabel('Πλάτος')
plt.grid(True)

# |D_6[k]| Plot
plt.subplot(3, 1, 3)
plt.plot(k, D6_mag, color='r')
plt.title('Μέτρο DFT του d_6[n]: $|D_6[k]|$')
plt.ylabel('Πλάτος')
plt.xlabel('Δείκτης Συχνότητας (k)')
plt.grid(True)
plt.tight_layout()
plt.show()