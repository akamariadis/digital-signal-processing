import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

file_path = 'speech_utterance.wav'
signal, fs = sf.read(file_path)
N = len(signal)
time_axis = np.arange(N) / fs
plt.figure(figsize=(14, 5))
plt.plot(time_axis, signal, color='teal', linewidth=0.8)
plt.title('Voice Signal in Time Domain')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()