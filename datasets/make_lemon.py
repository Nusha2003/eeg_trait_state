import os
import mne
import numpy as np

INPUT_DIR = "/scratch1/amadapur/data/lemon/unpacked_data"
OUTPUT_DIR = "/scratch1/amadapur/data/lemon/lemon_numpy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- PASS 1: Find Common Channels ---
print("Finding common channels across all subjects...")
all_files = []
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".set"):
            all_files.append(os.path.join(root, file))

common_channels = None

for filepath in all_files:
    # Use preload=False to quickly read only the header/info
    info = mne.io.read_raw_eeglab(filepath, preload=False).info
    ch_names = set(info['ch_names'])
    
    if common_channels is None:
        common_channels = ch_names
    else:
        common_channels = common_channels.intersection(ch_names)

common_channels = sorted(list(common_channels))
print(f"Found {len(common_channels)} common channels.")

# --- PASS 2: Process and Save ---
for filepath in all_files:
    print("Processing:", filepath)
    
    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
        
        # Pick only the common channels and ensure they are in the same order
        raw.pick_channels(common_channels)
        
        # Standardize reference and filter
        raw.set_eeg_reference("average")
        raw.filter(1, 40, fir_design='firwin')

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=4,
            reject_by_annotation=True,
            preload=True
        )

        # X shape will now be consistent: (n_epochs, n_common_channels, n_times)
        X = epochs.get_data()

        subject_name = os.path.basename(filepath).replace(".set", "")
        save_path = os.path.join(OUTPUT_DIR, subject_name + ".npz")

        np.savez(
            save_path,
            data=X,
            sfreq=raw.info["sfreq"],
            channels=common_channels
        )
        print("Saved:", save_path, "| Shape:", X.shape)
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")