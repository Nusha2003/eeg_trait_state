import pandas as pd
import numpy as np
import joblib
import argparse
import os


def make_splits(labels_path, n_subjects="all", seed=42, train_frac=0.8):
    labels_df = pd.read_csv(labels_path)
    labels_df["original_index"] = np.arange(len(labels_df))
    rng = np.random.default_rng(seed)

    all_subs = np.array(sorted(labels_df["subject"].unique(), key=lambda x: str(x)))
    n_available = len(all_subs)

    if n_available < 4:
        raise ValueError(f"Need at least 4 subjects, found {n_available}.")

    if n_subjects == "all":
        selected_subs = all_subs
    else:
        if n_subjects < 4:
            raise ValueError(f"n_subjects must be at least 4, got {n_subjects}.")
        if n_subjects > n_available:
            raise ValueError(
                f"Requested n_subjects={n_subjects}, but only {n_available} subjects available."
            )
        selected_subs = rng.choice(all_subs, size=n_subjects, replace=False)

    df = labels_df[labels_df["subject"].isin(selected_subs)].reset_index(drop=True)

    # Create train/test split within each subject-condition pair.
    df["original_part"] = "test"

    train_indices = []
    test_indices = []

    for (subj, cond), group in df.groupby(["subject", "condition"]):
        idx = group.index.to_numpy()
        rng.shuffle(idx)

        n_train = int(np.floor(train_frac * len(idx)))

        # Make sure both train/test are nonempty when possible.
        if len(idx) >= 2:
            n_train = min(max(n_train, 1), len(idx) - 1)

        train_indices.extend(idx[:n_train].tolist())
        test_indices.extend(idx[n_train:].tolist())

    df.loc[train_indices, "original_part"] = "train"

    trait_train = sorted(train_indices)
    trait_test = sorted(test_indices)

    split_results = []

    for target_sub in selected_subs:
        subj_mask = df["subject"] == target_sub

        # Within-subject state decoding:
        # train/test on same subject, split by original_part.
        within_train = df[
            subj_mask & (df["original_part"] == "train")
        ].index.tolist()

        within_test = df[
            subj_mask & (df["original_part"] == "test")
        ].index.tolist()

        # Between-subject state decoding:
        # train on all other subjects' train trials,
        # test on target subject's held-out trials.
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
        default="/home1/amadapur/projects/eeg_trait_state_geometry/data/bnci2014_002/bnci2014_002_psd_labels.csv"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home1/amadapur/projects/eeg_trait_state_geometry/splits/bnci2014_002"
    )

    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8
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
                seed=seed,
                train_frac=args.train_frac
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