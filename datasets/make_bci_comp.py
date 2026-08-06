import os
import argparse
import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.stats import skew, kurtosis
import antropy as ant

from moabb.datasets import BNCI2014_002
from moabb.paradigms import MotorImagery


def psd_features(trial, sfreq):
    freq_bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    freqs, psd = welch(
        trial,
        fs=sfreq,
        nperseg=trial.shape[1],
        axis=1
    )

    feat = []
    for fmin, fmax in freq_bands.values():
        idx = (freqs >= fmin) & (freqs < fmax)
        feat.extend(psd[:, idx].mean(axis=1))

    return np.array(feat, dtype=np.float32)


def entropy_features(trial, sfreq=None):
    var = np.var(trial, axis=1)
    var = np.maximum(var, 1e-12)
    return (0.5 * np.log(2 * np.pi * np.e * var)).astype(np.float32)


def complexity_features(trial, sfreq=None):
    return np.array(
        [ant.higuchi_fd(ch) for ch in trial],
        dtype=np.float32
    )


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
    "stats": stats_features,
}


def make_bnci2014_002(feature_type, out_dir, seed=0, balance=True):
    rng = np.random.default_rng(seed)
    feature_func = FEATURE_MAP[feature_type]

    dataset = BNCI2014_002()

    paradigm = MotorImagery(
        events=["right_hand", "feet"],
        n_classes=2,
        tmin=3.0,
        tmax=8.0,
    )

    X_epochs, y, metadata = paradigm.get_data(dataset=dataset)

    sfreq = 512.0
    print(f"Loaded BNCI2014_002: X={X_epochs.shape}, sfreq={sfreq}")

    rows_by_group = {}

    for i in range(X_epochs.shape[0]):
        subject = str(metadata.iloc[i]["subject"])
        session = str(metadata.iloc[i].get("session", "session_0"))
        run = str(metadata.iloc[i].get("run", "run_0"))
        condition = str(y[i])

        trial = X_epochs[i]  # shape: channels x samples
        feats = feature_func(trial, sfreq)

        key = (subject, condition)
        rows_by_group.setdefault(key, []).append({
            "features": feats,
            "subject": subject,
            "condition": condition,
            "session": session,
            "run": run,
        })

    selected_rows = []

    if balance:
        subjects = sorted(set(k[0] for k in rows_by_group.keys()))

        for subject in subjects:
            subject_keys = [k for k in rows_by_group.keys() if k[0] == subject]

            if len(subject_keys) < 2:
                continue

            min_trials = min(len(rows_by_group[k]) for k in subject_keys)

            for key in subject_keys:
                group = rows_by_group[key]
                idx = rng.choice(len(group), size=min_trials, replace=False)
                selected_rows.extend([group[j] for j in idx])
    else:
        for group in rows_by_group.values():
            selected_rows.extend(group)

    X = np.vstack([r["features"] for r in selected_rows])

    labels = pd.DataFrame([
        {
            "subject": r["subject"],
            "condition": r["condition"],
            "session": r["session"],
            "run": r["run"],
        }
        for r in selected_rows
    ])

    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(out_dir, f"bnci2014_002_{feature_type}_data.csv")
    label_path = os.path.join(out_dir, f"bnci2014_002_{feature_type}_labels.csv")

    pd.DataFrame(X).to_csv(data_path, index=False)
    labels.to_csv(label_path, index=False)

    print(f"Saved features: {data_path}")
    print(f"Saved labels:   {label_path}")
    print(f"Final shape: X={X.shape}, labels={labels.shape}")
    print(labels.groupby(["subject", "condition"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract BNCI2014_002 features for EEG trait-state hierarchy."
    )

    parser.add_argument(
        "--feature",
        type=str,
        default="psd",
        choices=FEATURE_MAP.keys(),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home1/amadapur/projects/eeg_trait_state_geometry/data/bnci2014_002",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Do not balance trials per subject-condition.",
    )

    args = parser.parse_args()

    make_bnci2014_002(
        feature_type=args.feature,
        out_dir=args.out_dir,
        seed=args.seed,
        balance=not args.no_balance,
    )