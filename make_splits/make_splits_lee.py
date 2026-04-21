import pandas as pd
import numpy as np
import joblib
import argparse
import os


def make_splits(labels_path, n_subjects=None, seed=42):
    labels_df = pd.read_csv(labels_path)

    rng = np.random.default_rng(seed)
    all_subs = labels_df["subject"].unique()
    n_available = len(all_subs)

    if n_available < 4:
        raise ValueError(f"Need at least 4 subjects total, found {n_available}.")

    if n_subjects is not None:
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
    else:
        df = labels_df.copy()
        selected_subs = df["subject"].unique()

    split_results = []

    selected_subs = np.array(selected_subs).copy()
    rng.shuffle(selected_subs)

    n_test = max(1, int(0.2 * len(selected_subs)))

    trait_test_subs = selected_subs[:n_test]
    trait_train_subs = selected_subs[n_test:]

    trait_train = df[
        (df["subject"].isin(trait_train_subs)) &
        (df["session"] == 1)
    ].index.tolist()

    trait_test = df[
        (df["subject"].isin(trait_test_subs)) &
        (df["session"] == 2)
    ].index.tolist()

    for target_sub in selected_subs:
        subj_mask = df["subject"] == target_sub

        within_train = df[
            subj_mask & (df["session"] == 1)
        ].index.tolist()

        within_test = df[
            subj_mask & (df["session"] == 2)
        ].index.tolist()

        between_train = df[
            (~subj_mask) & (df["session"] == 1)
        ].index.tolist()

        between_test = df[
            subj_mask & (df["session"] == 2)
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
    if total_available < 4:
        raise ValueError(f"Need at least 4 subjects total, found {total_available}.")

    fixed_sizes = [4, 10, 20, 40]
    sizes = [x for x in fixed_sizes if x <= total_available]
    sizes.append(None)   # None = all/full
    return sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        type=str,
        default="/home1/amadapur/projects/eeg_trait_state_geometry/data/lee/motor_complexity_labels.csv"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home1/amadapur/projects/eeg_trait_state_geometry/splits/lee"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    total_available = labels_df["subject"].nunique()

    sizes = get_subject_sizes(total_available)
    print("Using subject sizes:", ["all" if s is None else s for s in sizes])

    for size in sizes:
        iterations = 10 if size is not None else 1

        for i in range(iterations):
            seed = 42 + i

            splits, trait_split, metadata = make_splits(
                args.labels,
                n_subjects=size,
                seed=seed
            )

            if size is None:
                suffix = "_nall"
            else:
                suffix = f"_n{size}_rep{i}"

            output_file = os.path.join(args.out_dir, f"splits{suffix}.pkl")

            joblib.dump({
                "splits": splits,
                "trait_split": trait_split,
                "metadata": metadata
            }, output_file)

            print(f"Saved {output_file}")

    print("Splits generated")