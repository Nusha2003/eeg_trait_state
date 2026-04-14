import pandas as pd
import os
import numpy as np
import argparse
from scipy.signal import welch
from scipy.io import loadmat
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


def make_all_motor(feature_type="psd", seed=0):

    DATA_ROOT = "/scratch1/amadapur/data/Lee2019"
    OUT_DIR = "/home1/amadapur/projects/eeg_trait_state_geometry/data"

    rng = np.random.default_rng(seed)
    feature_func = FEATURE_MAP[feature_type]

    X = []
    rows = []

    for fname in sorted(os.listdir(DATA_ROOT)):

        if not fname.endswith(".mat"):
            continue

        print(f"Processing {fname} with {feature_type} features...")

        path = os.path.join(DATA_ROOT, fname)
        mat = loadmat(path)

        # parse metadata from filename
        subject = fname.split("subj")[1][:2]
        session = int(fname.split("sess")[1][:2])

        for part in ["train", "test"]:

            key = f"EEG_MI_{part}"

            if key not in mat:
                continue

            data = mat[key][0,0]

            trials = data["smt"]            # samples × trials × channels
            labels = data["y_dec"].flatten()
            sfreq = float(data["fs"])

            n_trials = trials.shape[1]

            for i in range(n_trials):

                trial = trials[:, i, :].T   # channels × samples

                # optional crop (recommended for MI)
                trial = trial[:, 500:3500]

                feats = feature_func(trial, sfreq)

                X.append(feats)

                rows.append({
                    "subject": subject,
                    "session": session,
                    "part": part,
                    "condition": int(labels[i])
                })

    X = np.vstack(X) if len(X) else np.empty((0, 0), dtype=np.float32)

    os.makedirs(os.path.join(OUT_DIR, args.dataset), exist_ok=True)

    pd.DataFrame(X).to_csv(
        f"{OUT_DIR}/{args.dataset}/motor_{feature_type}_data.csv",
        index=False
    )

    pd.DataFrame(rows).to_csv(
        f"{OUT_DIR}/{args.dataset}/motor_{feature_type}_labels.csv",
        index=False
    )

    print(f"Saved {feature_type} dataset: {X.shape}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="EEG Feature Extraction for Lee2019 Motor Imagery"
    )

    parser.add_argument(
        "--feature",
        type=str,
        choices=FEATURE_MAP.keys(),
        help="Type of feature to extract"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    args = parser.parse_args()

    make_all_motor(feature_type=args.feature)