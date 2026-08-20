import os
import argparse

import antropy as ant
import mne
import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.stats import skew, kurtosis


# ============================================================
# Feature extraction
# ============================================================

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
        feat.extend(
            psd[:, idx].mean(axis=1)
        )

    return np.asarray(
        feat,
        dtype=np.float32,
    )


def entropy_features(trial, sfreq=None):
    var = np.var(
        trial,
        axis=1,
    )

    return (
        0.5
        * np.log(
            2
            * np.pi
            * np.e
            * var
        )
    ).astype(np.float32)


def complexity_features(trial, sfreq=None):
    return np.asarray(
        [
            ant.higuchi_fd(ch)
            for ch in trial
        ],
        dtype=np.float32,
    )


def stats_features(trial, sfreq=None):
    means = np.mean(
        trial,
        axis=1,
    )

    stds = np.std(
        trial,
        axis=1,
    )

    skews = skew(
        trial,
        axis=1,
    )

    kurts = kurtosis(
        trial,
        axis=1,
    )

    return np.concatenate(
        [
            means,
            stds,
            skews,
            kurts,
        ]
    ).astype(np.float32)


FEATURE_MAP = {
    "psd": psd_features,
    "entropy": entropy_features,
    "complexity": complexity_features,
    "stats": stats_features,
}


# ============================================================
# Motor dataset
# ============================================================

def make_all_motor(
    feature_type="psd",
    num_classes=10,
    seed=0,
):
    EDF_ROOT = (
        "/scratch1/amadapur/data/physionet/"
        "physionet.org/files/eegmmidb/1.0.0"
    )

    OUT_DIR = (
        "/home1/amadapur/projects/"
        "eeg_trait_state_geometry/data/motor_imagery"
    )

    WINDOW_SEC = 4

    if num_classes not in {6, 10}:
        raise ValueError(
            "num_classes must be either 6 or 10."
        )

    if feature_type not in FEATURE_MAP:
        raise ValueError(
            f"Unknown feature type: {feature_type}"
        )

    rng = np.random.default_rng(seed)

    feature_func = FEATURE_MAP[
        feature_type
    ]

    # --------------------------------------------------------
    # PhysioNet run definitions
    # --------------------------------------------------------

    RUN_META = {
        1: (
            "baseline_eyes_open",
            None,
            None,
        ),
        2: (
            "baseline_eyes_closed",
            None,
            None,
        ),

        3: (
            "task1",
            "real",
            "lr_fists",
        ),
        4: (
            "task2",
            "imag",
            "lr_fists",
        ),

        5: (
            "task3",
            "real",
            "both_fists_feet",
        ),
        6: (
            "task4",
            "imag",
            "both_fists_feet",
        ),

        7: (
            "task1",
            "real",
            "lr_fists",
        ),
        8: (
            "task2",
            "imag",
            "lr_fists",
        ),

        9: (
            "task3",
            "real",
            "both_fists_feet",
        ),
        10: (
            "task4",
            "imag",
            "both_fists_feet",
        ),

        11: (
            "task1",
            "real",
            "lr_fists",
        ),
        12: (
            "task2",
            "imag",
            "lr_fists",
        ),

        13: (
            "task3",
            "real",
            "both_fists_feet",
        ),
        14: (
            "task4",
            "imag",
            "both_fists_feet",
        ),
    }

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    def label_from(run, desc):
        name, modality, effector = (
            RUN_META[run]
        )

        # Baseline conditions
        if run == 1:
            if desc == "T0":
                return (
                    "EO"
                    if num_classes == 6
                    else "baseline_eyes_open"
                )
            return None

        if run == 2:
            if desc == "T0":
                return (
                    "EC"
                    if num_classes == 6
                    else "baseline_eyes_closed"
                )
            return None

        # Ignore non-task annotations
        if desc not in {"T1", "T2"}:
            return None

        # ------------------------------
        # 6-class setup
        # ------------------------------

        if num_classes == 6:
            if effector == "lr_fists":
                return (
                    "left_fist"
                    if desc == "T1"
                    else "right_fist"
                )

            if effector == "both_fists_feet":
                return (
                    "both_fists"
                    if desc == "T1"
                    else "both_feet"
                )

        # ------------------------------
        # 10-class setup
        # ------------------------------

        if effector == "lr_fists":
            side = (
                "left_fist"
                if desc == "T1"
                else "right_fist"
            )

        elif effector == "both_fists_feet":
            side = (
                "both_fists"
                if desc == "T1"
                else "both_feet"
            )

        else:
            raise ValueError(
                f"Unknown effector: {effector}"
            )

        return (
            f"{name}_{modality}_{side}"
        )

    # ========================================================
    # Extract all subjects
    # ========================================================

    X = []
    rows = []

    subjects = sorted(
        os.listdir(EDF_ROOT)
    )

    for subject in subjects:
        subj_dir = os.path.join(
            EDF_ROOT,
            subject,
        )

        if not os.path.isdir(
            subj_dir
        ):
            continue

        print(
            f"Processing {subject} | "
            f"feature={feature_type} | "
            f"classes={num_classes}"
        )

        # Key:
        # condition -> list of samples
        subject_buffer = {}

        edf_files = sorted(
            os.listdir(subj_dir)
        )

        for fname in edf_files:
            if not fname.endswith(
                ".edf"
            ):
                continue

            run = int(
                fname
                .split("R")[-1]
                .replace(".edf", "")
            )

            if run not in RUN_META:
                continue

            edf_path = os.path.join(
                subj_dir,
                fname,
            )

            raw = mne.io.read_raw_edf(
                edf_path,
                preload=True,
                verbose=False,
            )

            try:
                data = raw.get_data()

                sfreq = float(
                    raw.info["sfreq"]
                )

                win_len = int(
                    WINDOW_SEC
                    * sfreq
                )

                annotations = zip(
                    raw.annotations.onset,
                    raw.annotations.duration,
                    raw.annotations.description,
                )

                for (
                    onset,
                    duration,
                    desc,
                ) in annotations:

                    lab = label_from(
                        run,
                        desc,
                    )

                    if lab is None:
                        continue

                    start = int(
                        onset
                        * sfreq
                    )

                    end = int(
                        (onset + duration)
                        * sfreq
                    )

                    if (
                        end > data.shape[1]
                        or end <= start
                    ):
                        continue

                    segment = data[
                        :,
                        start:end,
                    ]

                    # --------------------------------
                    # Baseline runs
                    # Split into non-overlapping
                    # 4-second windows.
                    # --------------------------------

                    if run in {1, 2}:
                        if (
                            segment.shape[1]
                            < win_len
                        ):
                            continue

                        for i in range(
                            0,
                            segment.shape[1]
                            - win_len
                            + 1,
                            win_len,
                        ):
                            window_start = (
                                start + i
                            )

                            window_end = (
                                window_start
                                + win_len
                            )

                            trial = data[
                                :,
                                window_start:
                                window_end,
                            ]

                            sample = {
                                "features": (
                                    feature_func(
                                        trial,
                                        sfreq,
                                    )
                                ),
                                "run": run,
                                "file": fname,
                                "start_sample": (
                                    window_start
                                ),
                                "end_sample": (
                                    window_end
                                ),
                            }

                            subject_buffer.setdefault(
                                lab,
                                [],
                            ).append(
                                sample
                            )

                    # --------------------------------
                    # Task runs
                    # One annotation = one trial
                    # --------------------------------

                    else:
                        sample = {
                            "features": (
                                feature_func(
                                    segment,
                                    sfreq,
                                )
                            ),
                            "run": run,
                            "file": fname,
                            "start_sample": start,
                            "end_sample": end,
                        }

                        subject_buffer.setdefault(
                            lab,
                            [],
                        ).append(
                            sample
                        )

            finally:
                raw.close()
                del raw

        if not subject_buffer:
            continue

        # ----------------------------------------------------
        # Validate expected class count
        # ----------------------------------------------------

        expected_conditions = (
            {
                "EO",
                "EC",
                "left_fist",
                "right_fist",
                "both_fists",
                "both_feet",
            }
            if num_classes == 6
            else None
        )

        if expected_conditions is not None:
            found_conditions = set(
                subject_buffer.keys()
            )

            missing = (
                expected_conditions
                - found_conditions
            )

            if missing:
                print(
                    f"Skipping {subject}: "
                    f"missing conditions "
                    f"{sorted(missing)}"
                )
                continue

        # ----------------------------------------------------
        # Balance conditions within subject
        # ----------------------------------------------------

        min_trials = min(
            len(samples)
            for samples
            in subject_buffer.values()
        )

        if min_trials == 0:
            continue

        for (
            lab,
            samples,
        ) in subject_buffer.items():

            selected = rng.choice(
                len(samples),
                size=min_trials,
                replace=False,
            )

            for j in selected:
                sample = samples[j]

                X.append(
                    sample["features"]
                )

                rows.append(
                    {
                        "subject": subject,
                        "condition": lab,
                        "run": sample["run"],
                        "file": sample["file"],
                        "start_sample": (
                            sample[
                                "start_sample"
                            ]
                        ),
                        "end_sample": (
                            sample[
                                "end_sample"
                            ]
                        ),
                    }
                )

    # ========================================================
    # Save
    # ========================================================

    if X:
        X = np.vstack(
            X
        ).astype(
            np.float32
        )

    else:
        X = np.empty(
            (0, 0),
            dtype=np.float32,
        )

    labels_df = pd.DataFrame(
        rows
    )

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    if num_classes == 6:
        data_filename = (
            f"motor_{feature_type}"
            "_data_state6.csv"
        )

        labels_filename = (
            f"motor_{feature_type}"
            "_labels_state6.csv"
        )

    else:
        data_filename = (
            f"motor_{feature_type}"
            "_data.csv"
        )

        labels_filename = (
            f"motor_{feature_type}"
            "_labels.csv"
        )

    data_path = os.path.join(
        OUT_DIR,
        data_filename,
    )

    labels_path = os.path.join(
        OUT_DIR,
        labels_filename,
    )

    pd.DataFrame(
        X
    ).to_csv(
        data_path,
        index=False,
    )

    labels_df.to_csv(
        labels_path,
        index=False,
    )

    print()
    print("=" * 60)
    print(
        f"Saved {feature_type} "
        f"{num_classes}-class dataset"
    )
    print(
        f"Data shape: {X.shape}"
    )
    print(
        f"Data: {data_path}"
    )
    print(
        f"Labels: {labels_path}"
    )

    if len(labels_df):
        print(
            "Conditions:",
            sorted(
                labels_df[
                    "condition"
                ].unique()
            ),
        )

    print("=" * 60)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "EEG Feature Extraction "
            "for Trait-State Geometry"
        )
    )

    parser.add_argument(
        "--feature",
        type=str,
        default="psd",
        choices=FEATURE_MAP.keys(),
        help=(
            "Feature type: "
            "psd, entropy, complexity, "
            "or stats"
        ),
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        choices=[6, 10],
        help=(
            "Motor condition definition: "
            "6-class combined or "
            "10-class task-specific"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed used for "
            "within-subject balancing"
        ),
    )

    args = parser.parse_args()

    make_all_motor(
        feature_type=args.feature,
        num_classes=args.num_classes,
        seed=args.seed,
    )