import os
import argparse

import antropy as ant
import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.stats import skew, kurtosis

from moabb.datasets import ErpCore2021_ERN
from moabb.paradigms import P300


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
        axis=1,
    )

    feat = []

    for fmin, fmax in freq_bands.values():
        idx = (freqs >= fmin) & (freqs < fmax)
        feat.extend(psd[:, idx].mean(axis=1))

    return np.array(feat, dtype=np.float32)


def entropy_features(trial, sfreq=None):
    var = np.var(trial, axis=1)
    var = np.maximum(var, 1e-12)

    return (
        0.5 * np.log(2 * np.pi * np.e * var)
    ).astype(np.float32)


def complexity_features(trial, sfreq=None):
    return np.array(
        [ant.higuchi_fd(ch) for ch in trial],
        dtype=np.float32,
    )


def stats_features(trial, sfreq=None):
    means = np.mean(trial, axis=1)
    stds = np.std(trial, axis=1)
    skews = skew(trial, axis=1)
    kurts = kurtosis(trial, axis=1)

    return np.concatenate(
        [means, stds, skews, kurts]
    ).astype(np.float32)


FEATURE_MAP = {
    "psd": psd_features,
    "entropy": entropy_features,
    "complexity": complexity_features,
    "stats": stats_features,
}


def make_ern(
    feature_type,
    out_dir,
    seed=0,
    balance=True,
    min_trials_per_condition=10,
):
    rng = np.random.default_rng(seed)
    feature_func = FEATURE_MAP[feature_type]

    dataset = ErpCore2021_ERN()

    paradigm = P300(
        events=["Target", "NonTarget"],
        tmin=0.0,
        tmax=1.0,
    )

    X_epochs, y, metadata = paradigm.get_data(dataset=dataset)

    sfreq = 1024.0

    print(
        f"Loaded ErpCore2021_ERN: "
        f"X={X_epochs.shape}, sfreq={sfreq}"
    )
    print("Original labels:", np.unique(y, return_counts=True))

    rows_by_group = {}

    for i in range(X_epochs.shape[0]):
        subject = str(metadata.iloc[i]["subject"])
        session = str(
            metadata.iloc[i].get("session", "session_0")
        )
        run = str(
            metadata.iloc[i].get("run", "run_0")
        )
        condition = str(y[i])

        trial = X_epochs[i]
        feats = feature_func(trial, sfreq)

        key = (subject, condition)

        rows_by_group.setdefault(key, []).append(
            {
                "features": feats,
                "subject": subject,
                "condition": condition,
                "session": session,
                "run": run,
            }
        )

    all_subjects = sorted(
        {subject for subject, _ in rows_by_group}
    )

    usable_subjects = []
    excluded_subjects = []

    for subject in all_subjects:
        subject_keys = [
            key
            for key in rows_by_group
            if key[0] == subject
        ]

        condition_counts = {
            condition: len(rows_by_group[(subject, condition)])
            for _, condition in subject_keys
        }

        has_all_conditions = len(subject_keys) >= 2

        enough_trials = (
            has_all_conditions
            and min(condition_counts.values())
            >= min_trials_per_condition
        )

        if enough_trials:
            usable_subjects.append(subject)
        else:
            excluded_subjects.append(
                {
                    "subject": subject,
                    "counts": condition_counts,
                }
            )

    print(
        f"Usable subjects: {len(usable_subjects)} / "
        f"{len(all_subjects)}"
    )

    if excluded_subjects:
        print(
            "Excluded subjects with fewer than "
            f"{min_trials_per_condition} trials per condition:"
        )

        for item in excluded_subjects:
            print(
                f"  Subject {item['subject']}: "
                f"{item['counts']}"
            )

    if len(usable_subjects) < 4:
        raise ValueError(
            "Fewer than four subjects remain after "
            "minimum-trial filtering."
        )

    selected_rows = []

    for subject in usable_subjects:
        subject_keys = [
            key
            for key in rows_by_group
            if key[0] == subject
        ]

        if balance:
            min_trials = min(
                len(rows_by_group[key])
                for key in subject_keys
            )

            for key in subject_keys:
                group = rows_by_group[key]

                selected_indices = rng.choice(
                    len(group),
                    size=min_trials,
                    replace=False,
                )

                selected_rows.extend(
                    group[index]
                    for index in selected_indices
                )

        else:
            for key in subject_keys:
                selected_rows.extend(rows_by_group[key])

    if not selected_rows:
        raise ValueError(
            "No trials remained after filtering."
        )

    X = np.vstack(
        [row["features"] for row in selected_rows]
    )

    labels = pd.DataFrame(
        [
            {
                "subject": row["subject"],
                "condition": row["condition"],
                "session": row["session"],
                "run": row["run"],
            }
            for row in selected_rows
        ]
    )

    if len(X) != len(labels):
        raise RuntimeError(
            f"Feature-label mismatch: "
            f"X={len(X)}, labels={len(labels)}"
        )

    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(
        out_dir,
        f"ern_{feature_type}_data.csv",
    )

    label_path = os.path.join(
        out_dir,
        f"ern_{feature_type}_labels.csv",
    )

    pd.DataFrame(X).to_csv(
        data_path,
        index=False,
    )

    labels.to_csv(
        label_path,
        index=False,
    )

    counts = (
        labels.groupby(["subject", "condition"])
        .size()
        .unstack(fill_value=0)
    )

    print(f"Saved features: {data_path}")
    print(f"Saved labels:   {label_path}")
    print(f"Final shape: X={X.shape}, labels={labels.shape}")
    print("\nFinal trials per subject and condition:")
    print(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract ERN features for EEG "
            "trait-state hierarchy."
        )
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
        default=(
            "/home1/amadapur/projects/"
            "eeg_trait_state_geometry/data/ern"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--min_trials",
        type=int,
        default=10,
        help=(
            "Minimum required trials in every condition "
            "for each subject."
        ),
    )

    parser.add_argument(
        "--no_balance",
        action="store_true",
        help=(
            "Do not balance conditions within each subject."
        ),
    )

    args = parser.parse_args()

    make_ern(
        feature_type=args.feature,
        out_dir=args.out_dir,
        seed=args.seed,
        balance=not args.no_balance,
        min_trials_per_condition=args.min_trials,
    )