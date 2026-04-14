import pandas as pd
import os
import numpy as np
import argparse
from scipy.signal import welch
from scipy.io import loadmat
import antropy as ant
from scipy.stats import skew, kurtosis
import mne

import mne

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
    return (0.5 * np.log(2 * np.pi * np.exp(1) * var)).astype(np.float32)


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


LABEL_MAP = {
    "left_hand": 1,
    "right_hand": 2,
    "rest": 3,
    "feet": 4,
}


def make_gamma_all(feature_type="psd"):
    DATA_ROOT = "/scratch1/amadapur/data/gamma/high-gamma-dataset/data"
    OUT_DIR = "/home1/amadapur/projects/eeg_trait_state_geometry/data"

    feature_func = FEATURE_MAP[feature_type]

    X = []
    rows = []

    for part in ["train", "test"]:
        split_dir = os.path.join(DATA_ROOT, part)

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".edf"):
                continue

            subject = int(os.path.splitext(fname)[0])
            edf_path = os.path.join(split_dir, fname)

            print(f"Processing {edf_path} with {feature_type} features...")

            raw = mne.io.read_raw_edf(edf_path, preload=True, infer_types=True, verbose=False)
            sfreq = float(raw.info["sfreq"])

            # optional preprocessing
            raw.filter(1., 45., verbose=False)
            raw.notch_filter(60., verbose=False)

            # turn annotations into events
            events, event_id = mne.events_from_annotations(raw, event_id=LABEL_MAP)

            # make 4-second trials from each cue
            epochs = mne.Epochs(
                raw,
                events,
                event_id=LABEL_MAP,
                tmin=0.0,
                tmax=4.0,
                baseline=None,
                preload=True,
                verbose=False
            )

            data = epochs.get_data(copy=True)   # shape: trials x channels x samples
            labels = epochs.events[:, 2]

            for i in range(len(data)):
                trial = data[i]   # channels x samples
                feats = feature_func(trial, sfreq)

                X.append(feats)
                rows.append({
                    "subject": subject,
                    "original_part": part,
                    "file": fname,
                    "trial": i,
                    "condition": int(labels[i])
                })

    X = np.vstack(X) if len(X) else np.empty((0, 0), dtype=np.float32)

    os.makedirs(os.path.join(OUT_DIR, args.dataset), exist_ok=True)

    pd.DataFrame(X).to_csv(
        f"{OUT_DIR}/{args.dataset}/gamma_{feature_type}_data.csv",
        index=False
    )

    pd.DataFrame(rows).to_csv(
        f"{OUT_DIR}/{args.dataset}/gamma_{feature_type}_labels.csv",
        index=False
    )

    print(f"Saved {feature_type} dataset: {X.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Feature Extraction for High-Gamma Dataset"
    )

    parser.add_argument(
        "--feature",
        type=str,
        choices=FEATURE_MAP.keys(),
        required=True,
        help="Type of feature to extract"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    args = parser.parse_args()

    make_gamma_all(feature_type=args.feature)
