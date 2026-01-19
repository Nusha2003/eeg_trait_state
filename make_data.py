import pandas as pd
import os
import mne
import numpy as np
import argparse
from scipy.signal import welch
import antropy as ant 
from scipy.stats import skew, kurtosis

def psd_features(trial, sfreq):
    FREQ_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
    freqs, psd = welch(trial, fs=sfreq, nperseg=trial.shape[1], axis=1)
    feat = []
    for (fmin, fmax) in FREQ_BANDS.values():
        idx = (freqs >= fmin) & (freqs < fmax)
        feat.extend(psd[:, idx].mean(axis=1))
    return np.array(feat, dtype=np.float32)

def entropy_features(trial, sfreq=None):
    var = np.var(trial, axis=1)
    return 0.5 * np.log(2 * np.pi * np.exp(1) * var).astype(np.float32)

def complexity_features(trial, sfreq=None):
    return np.array([ant.higuchi_fd(ch) for ch in trial], dtype=np.float32)

def stats_features(trial, sfreq=None):
    means = np.mean(trial, axis=1)
    stds = np.std(trial, axis=1)
    skews = skew(trial, axis=1)
    kurts = kurtosis(trial, axis=1)
    return np.concatenate([means, stds, skews, kurts]).astype(np.float32)

FEATURE_MAP = {
    "psd": psd_features,
    "entropy": entropy_features,
    "complexity": complexity_features,
    "stats": stats_features
}

def make_all_motor(feature_type="psd", seed=0):
    EDF_ROOT = "/scratch1/amadapur/data/physionet.org/files/eegmmidb/1.0.0"
    OUT_DIR  = "/home1/amadapur/projects/eeg_trait_state_geometry/data"
    WINDOW_SEC = 10
    rng = np.random.default_rng(seed)

    RUN_META = {
        1: ("baseline_eyes_open",  None, None), 2: ("baseline_eyes_closed", None, None),
        3: ("task1", "real", "lr_fists"), 4: ("task2", "imag", "lr_fists"),
        5: ("task3", "real", "both_fists_feet"), 6: ("task4", "imag", "both_fists_feet"),
        7: ("task1", "real", "lr_fists"), 8: ("task2", "imag", "lr_fists"),
        9: ("task3", "real", "both_fists_feet"), 10:("task4", "imag", "both_fists_feet"),
        11:("task1", "real", "lr_fists"), 12:("task2", "imag", "lr_fists"),
        13:("task3", "real", "both_fists_feet"), 14:("task4", "imag", "both_fists_feet"),
    }

    def label_from(run, desc):
        name, modality, eff = RUN_META[run]
        if run in {1,2}: return name if desc == "T0" else None
        if desc not in {"T1","T2"}: return None
        if eff == "lr_fists": side = "left_fist" if desc == "T1" else "right_fist"
        else: side = "both_fists" if desc == "T1" else "both_feet"
        return f"{name}_{modality}_{side}"

    X, rows = [], []
    feature_func = FEATURE_MAP[feature_type]

    for subject in sorted(os.listdir(EDF_ROOT)):
        subj_dir = os.path.join(EDF_ROOT, subject)
        if not os.path.isdir(subj_dir): continue

        print(f"Processing {subject} with {feature_type} features...")
        subject_buffer = {}

        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith(".edf"): continue
            run = int(fname.split("R")[-1].replace(".edf", ""))
            if run not in RUN_META: continue

            raw = mne.io.read_raw_edf(os.path.join(subj_dir, fname), preload=True, verbose=False)
            data, sfreq = raw.get_data(), float(raw.info["sfreq"])
            win_len = int(WINDOW_SEC * sfreq)

            for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
                lab = label_from(run, desc)
                if lab is None: continue
                start, end = int(onset * sfreq), int((onset + duration) * sfreq)
                if end > data.shape[1] or end <= start: continue
                segment = data[:, start:end]

                if run in {1,2}:
                    if segment.shape[1] < win_len: continue
                    for i in range(0, segment.shape[1] - win_len + 1, win_len):
                        trial = segment[:, i:i+win_len]
                        subject_buffer.setdefault(lab, []).append(feature_func(trial, sfreq))
                else:
                    subject_buffer.setdefault(lab, []).append(feature_func(segment, sfreq))

        if not subject_buffer: continue
        min_trials = min(len(v) for v in subject_buffer.values())
        if min_trials == 0: continue

        for lab, feats in subject_buffer.items():
            idx = rng.choice(len(feats), size=min_trials, replace=False)
            for j in idx:
                X.append(feats[j])
                rows.append({"subject": subject, "condition": lab})

    X = np.vstack(X) if len(X) else np.empty((0, 0), dtype=np.float32)
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(X).to_csv(f"{OUT_DIR}/motor_{feature_type}_data.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/motor_{feature_type}_labels.csv", index=False)
    print(f"✅ Saved {feature_type} dataset: {X.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Feature Extraction for Trait-State Geometry")
    parser.add_argument("--feature", type=str, default="psd", choices=FEATURE_MAP.keys(),
                        help="Type of feature to extract (psd, entropy, complexity, stats)")
    args = parser.parse_args()

    make_all_motor(feature_type=args.feature)


