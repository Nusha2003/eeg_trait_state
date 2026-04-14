import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import antropy as ant

# --- Configuration ---
DATA_DIR = "/scratch1/amadapur/data/lemon/lemon_numpy"
SAVE_DIR = "/home1/amadapur/projects/eeg_trait_state_geometry/data/lemon"
sfreq = 250

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}

# Master lists for rows
psd_rows = []
entropy_rows = []
complexity_rows = []
label_rows = []

def get_entropy(epoch):
    variance = np.var(epoch, axis=1)
    # Adding a small epsilon to avoid log(0)
    return 0.5 * np.log(2 * np.pi * np.e * variance + 1e-9)

def get_complexity(epoch):

    # Note: This is computationally expensive
    return np.array([ant.higuchi_fd(ch) for ch in epoch])


files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])

for file in files:
    print(f"Processing: {file}")
    path = os.path.join(DATA_DIR, file)
    d = np.load(path)
    epochs = d["data"]  

    for epoch in epochs:

        if np.isnan(epoch).any() or np.std(epoch) == 0:
            continue

        freqs, pxx = welch(epoch, fs=sfreq, nperseg=sfreq*2, axis=1)
        
        current_psd_features = []
        for fmin, fmax in bands.values():
            idx = (freqs >= fmin) & (freqs <= fmax)
            if np.any(idx):
                band_power = pxx[:, idx].mean(axis=1)
            else:
                band_power = np.zeros(epoch.shape[0])
            current_psd_features.extend(band_power)


        current_entropy = get_entropy(epoch)
        current_complexity = get_complexity(epoch)
        psd_rows.append(current_psd_features)
        entropy_rows.append(current_entropy)
        complexity_rows.append(current_complexity)

        # Metadata
        subject = file.split("_")[0]
        condition = file.split("_")[1].replace(".npz", "")
        label_rows.append({"subject": subject, "condition": condition})

os.makedirs(SAVE_DIR, exist_ok=True)

df_psd = pd.DataFrame(psd_rows)
df_entropy = pd.DataFrame(entropy_rows)
df_complexity = pd.DataFrame(complexity_rows)
df_labels = pd.DataFrame(label_rows)

print(f"Final Counts -> PSD: {df_psd.shape}, Entropy: {df_entropy.shape}, Complexity: {df_complexity.shape}")

df_psd.to_csv(f"{SAVE_DIR}/lemon_psd_features.csv", index=False)
df_entropy.to_csv(f"{SAVE_DIR}/lemon_entropy_features.csv", index=False)
df_complexity.to_csv(f"{SAVE_DIR}/lemon_complexity_features.csv", index=False)
df_labels.to_csv(f"{SAVE_DIR}/lemon_labels.csv", index=False)

print("All features saved successfully and aligned.")