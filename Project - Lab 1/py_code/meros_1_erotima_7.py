import numpy as np

N = 1000
n = np.arange(N)
silence_len = 100

frequencies = {
    '1': (0.5346, 0.9273), '2': (0.5346, 1.0247), '3': (0.5346, 1.1328),
    '4': (0.5906, 0.9273), '5': (0.5906, 1.0247), '6': (0.5906, 1.1328),
    '7': (0.6535, 0.9273), '8': (0.6535, 1.0247), '9': (0.6535, 1.1328),
    '0': (0.7217, 1.0247)
}

AM1 = 3122674
AM2 = 3123434
digits_str = str(AM1 + AM2).zfill(8)
print(f"Το αρχικό άθροισμα ψηφίων είναι: {digits_str}")

sequence_parts = []
silence = np.zeros(silence_len)

for i, digit in enumerate(digits_str):
    omega_row, omega_col = frequencies[digit]
    tone = np.sin(omega_row * n) + np.sin(omega_col * n)
    sequence_parts.append(tone)
    if i < len(digits_str) - 1:
        sequence_parts.append(silence)

final_signal = np.concatenate(sequence_parts)

def ttdecode(signIn):
    L = 1000
    silence_len = 100
    N_fft = 1024
    ideal_rows = [87, 96, 107, 118]
    ideal_cols = [151, 167, 185]
    keypad = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [None, 0, None]
    ]

    num_digits = int(np.ceil(len(signIn) / (L + silence_len)))
    decoded_vector = []
    hamm_window = np.hamming(L)

    for i in range(num_digits):
        start_idx = i * (L + silence_len)
        end_idx = start_idx + L
        segment = signIn[start_idx:end_idx]
        if len(segment) < L:
            break
        segment_hamm = segment * hamm_window
        D_mag = np.abs(np.fft.fft(segment_hamm, n=N_fft))
        k_row_obs = np.argmax(D_mag[50:130]) + 50
        k_col_obs = np.argmax(D_mag[140:200]) + 140
        row_idx = np.argmin(np.abs(np.array(ideal_rows) - k_row_obs))
        col_idx = np.argmin(np.abs(np.array(ideal_cols) - k_col_obs))
        digit = keypad[row_idx][col_idx]
        decoded_vector.append(digit)
    return decoded_vector

result_vector = ttdecode(final_signal)
print(f" Vector = {result_vector}")
if "".join(map(str, result_vector)) == digits_str:
    print("Αποκωδικοποίηση επιτυχής")
else:
    print("Αποκωδικοποίηση ανεπιτυχής")

print ("\n")

try:
    easySig = np.load('easy_sig.npy')
    mediumSig = np.load('medium_sig.npy')
    hardSig = np.load('hard_sig.npy')
    easy_digits = ttdecode(easySig)
    medium_digits = ttdecode(mediumSig)
    hard_digits = ttdecode(hardSig)
    print(f"Αποκωδικοποίηση easy_sig: {easy_digits}")
    print(f"Αποκωδικοποίηση medium_sig: {medium_digits}")
    print(f"Αποκωδικοποίηση hard_sig: {hard_digits}")

except FileNotFoundError as e:
    print(f"Σφάλμα: Το αρχείο: ({e.filename}) δεν βρέθηκε")