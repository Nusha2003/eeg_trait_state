"""
within-subject (state decoding): use current splits (train, test)
between-subject (state decoding): all other subjects' train + held-out subject test
trait decoding: use existing train and test splits

Uses fixed subject-size settings:
- 4, 8, 10, "all"
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os


def make_splits(labels_path, n_subjects="all", seed=42):
    labels_df = pd.read_csv(labels_path)
    labels_df["original_index"] = np.arange(len(labels_df))
    rng = np.random.default_rng(seed)

    all_subs = labels_df["subject"].unique()
    n_available = len(all_subs)

    if n_available < 4:
        raise ValueError(f"Dataset must contain at least 4 subjects, found {n_available}.")

    if n_subjects == "all":
        selected_subs = all_subs
    else:
        if n_subjects < 4:
            raise ValueError(f"n_subjects must be at least 4, got {n_subjects}.")
        if n_subjects > n_available:
            raise ValueError(
                f"Requested n_subjects={n_subjects}, but only {n_available} subjects are available."
            )
        selected_subs = rng.choice(
            all_subs,
            size=n_subjects,
            replace=False
        )

    df = labels_df[labels_df["subject"].isin(selected_subs)].reset_index(drop=True)
    split_results = []

    # trait decoding
    trait_train = df[df["original_part"] == "train"].index.tolist()
    trait_test = df[df["original_part"] == "test"].index.tolist()

    for target_sub in selected_subs:
        subj_mask = df["subject"] == target_sub

        # within-subject state decoding
        within_train = df[
            subj_mask & (df["original_part"] == "train")
        ].index.tolist()

        within_test = df[
            subj_mask & (df["original_part"] == "test")
        ].index.tolist()

        # between-subject state decoding
        between_train = df[
            (~subj_mask) & (df["original_part"] == "train")
        ].index.tolist()

        between_test = df[
            subj_mask & (df["original_part"] == "test")
        ].index.tolist()

        split_results.append({
            "subject": target_sub,
            "within_state": {
                "train": within_train,
                "test": within_test
            },
            "between_state": {
                "train": between_train,
                "test": between_test
            }
        })

    return split_results, {"train": trait_train, "test": trait_test}, df


def get_subject_sizes(total_available):
    fixed_sizes = [4, 8, 10]
    sizes = [x for x in fixed_sizes if x <= total_available]
    sizes.append("all")
    return sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        default="/home1/amadapur/projects/eeg_trait_state_geometry/data/gamma/gamma_psd_labels.csv"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home1/amadapur/projects/eeg_trait_state_geometry/splits/gamma"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    total_available = labels_df["subject"].nunique()

    sizes = get_subject_sizes(total_available)
    print("Using subject sizes:", sizes)

    for size in sizes:
        iterations = 1 if size == "all" else 10

        for i in range(iterations):
            seed = 42 + i

            splits, trait_split, metadata = make_splits(
                args.labels,
                n_subjects=size,
                seed=seed
            )

            if size == "all":
                output_file = os.path.join(args.out_dir, "splits_nall.pkl")
            else:
                output_file = os.path.join(args.out_dir, f"splits_n{size}_rep{i}.pkl")

            joblib.dump({
                "splits": splits,
                "trait_split": trait_split,
                "metadata": metadata
            }, output_file)

            print(f"Saved {output_file}")

    print("Splits generated")