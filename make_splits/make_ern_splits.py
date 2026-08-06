from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def split_subject_condition_trials(
    df: pd.DataFrame,
    rng: np.random.Generator,
    train_fraction: float,
) -> tuple[list[int], list[int]]:
    """
    Create a stratified train/test split within each subject-condition pair.

    This ensures every subject contributes both conditions to both the
    training and test sets.
    """
    train_indices: list[int] = []
    test_indices: list[int] = []

    for (_, _), group in df.groupby(["subject", "condition"], sort=False):
        indices = group.index.to_numpy()
        indices = rng.permutation(indices)

        n_train = int(np.floor(len(indices) * train_fraction))

        # Guarantee at least one trial in each split.
        n_train = max(1, min(n_train, len(indices) - 1))

        train_indices.extend(indices[:n_train].tolist())
        test_indices.extend(indices[n_train:].tolist())

    return train_indices, test_indices


def make_splits(
    labels_path: str | os.PathLike[str],
    n_subjects: int | str = "all",
    seed: int = 42,
    train_fraction: float = 0.7,
) -> tuple[list[dict], dict[str, list[int]], pd.DataFrame]:
    labels_df = pd.read_csv(labels_path)
    labels_df["original_index"] = np.arange(len(labels_df))
    required_columns = {"subject", "condition"}
    missing_columns = required_columns - set(labels_df.columns)

    if missing_columns:
        raise ValueError(
            f"Labels file is missing required columns: {sorted(missing_columns)}"
        )

    # Keep labels consistent across files and repetitions.
    labels_df["subject"] = labels_df["subject"].astype(str)
    labels_df["condition"] = labels_df["condition"].astype(str)

    all_subjects = labels_df["subject"].unique()
    n_available = len(all_subjects)

    if n_available < 4:
        raise ValueError(
            f"Need at least 4 usable subjects, found {n_available}."
        )

    rng = np.random.default_rng(seed)

    if n_subjects == "all":
        selected_subjects = all_subjects
    else:
        n_subjects = int(n_subjects)

        if n_subjects < 4:
            raise ValueError(
                f"n_subjects must be at least 4, got {n_subjects}."
            )

        if n_subjects > n_available:
            raise ValueError(
                f"Requested n_subjects={n_subjects}, but only "
                f"{n_available} usable subjects are available."
            )

        selected_subjects = rng.choice(
            all_subjects,
            size=n_subjects,
            replace=False,
        )

    # Reset indices so saved split indices align with the filtered feature matrix
    # used for this subject group.
    df = (
        labels_df[labels_df["subject"].isin(selected_subjects)]
        .reset_index(drop=True)
    )

    trait_train, trait_test = split_subject_condition_trials(
        df=df,
        rng=rng,
        train_fraction=train_fraction,
    )

    trait_split = {
        "train": trait_train,
        "test": trait_test,
    }

    split_results: list[dict] = []

    trait_train_array = np.asarray(trait_train, dtype=int)
    trait_test_array = np.asarray(trait_test, dtype=int)

    subjects = df["subject"].to_numpy()

    for target_subject in selected_subjects:
        target_subject = str(target_subject)

        target_train = trait_train_array[
            subjects[trait_train_array] == target_subject
        ].tolist()

        target_test = trait_test_array[
            subjects[trait_test_array] == target_subject
        ].tolist()

        other_subject_train = trait_train_array[
            subjects[trait_train_array] != target_subject
        ].tolist()

        split_results.append(
            {
                "subject": target_subject,

                # Train and test condition decoding within the same subject.
                "within_state": {
                    "train": target_train,
                    "test": target_test,
                },

                # Train on other subjects and test on unseen trials
                # from the held-out target subject.
                "between_state": {
                    "train": other_subject_train,
                    "test": target_test,
                },
            }
        )

    return split_results, trait_split, df


def get_subject_sizes(total_available: int) -> list[int | str]:
    fixed_sizes = [4, 10, 20, 40]

    sizes: list[int | str] = [
        size
        for size in fixed_sizes
        if size <= total_available
    ]

    sizes.append("all")
    return sizes


def validate_splits(
    splits: list[dict],
    trait_split: dict[str, list[int]],
    metadata: pd.DataFrame,
) -> None:
    """Run basic leakage and index checks before saving."""

    n_rows = len(metadata)

    trait_train = set(trait_split["train"])
    trait_test = set(trait_split["test"])

    if trait_train & trait_test:
        raise ValueError("Trait train and test indices overlap.")

    all_trait_indices = trait_train | trait_test

    if all_trait_indices != set(range(n_rows)):
        missing = set(range(n_rows)) - all_trait_indices
        raise ValueError(
            f"Trait split does not cover every row. Missing {len(missing)} rows."
        )

    for subject_split in splits:
        subject = str(subject_split["subject"])

        within_train = set(subject_split["within_state"]["train"])
        within_test = set(subject_split["within_state"]["test"])

        between_train = set(subject_split["between_state"]["train"])
        between_test = set(subject_split["between_state"]["test"])

        if within_train & within_test:
            raise ValueError(
                f"Within-state train/test overlap for subject {subject}."
            )

        if between_train & between_test:
            raise ValueError(
                f"Between-state train/test overlap for subject {subject}."
            )

        within_train_subjects = set(
            metadata.loc[list(within_train), "subject"].astype(str)
        )

        within_test_subjects = set(
            metadata.loc[list(within_test), "subject"].astype(str)
        )

        if within_train_subjects != {subject}:
            raise ValueError(
                f"Within-state training contains other subjects for {subject}."
            )

        if within_test_subjects != {subject}:
            raise ValueError(
                f"Within-state test contains other subjects for {subject}."
            )

        between_train_subjects = set(
            metadata.loc[list(between_train), "subject"].astype(str)
        )

        if subject in between_train_subjects:
            raise ValueError(
                f"Held-out subject {subject} appears in between-state training."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ERN trait-state train/test splits."
    )

    parser.add_argument(
        "--labels",
        type=str,
        default=(
            "/home1/amadapur/projects/eeg_trait_state_geometry/"
            "data/ern/ern_psd_labels.csv"
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=(
            "/home1/amadapur/projects/eeg_trait_state_geometry/"
            "splits/ern"
        ),
    )

    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.7,
        help="Fraction of each subject-condition group used for training.",
    )

    args = parser.parse_args()

    if not 0 < args.train_fraction < 1:
        raise ValueError("--train_fraction must be between 0 and 1.")

    labels_df = pd.read_csv(args.labels)
    total_available = labels_df["subject"].nunique()

    sizes = get_subject_sizes(total_available)

    print(f"Available subjects: {total_available}")
    print("Using subject sizes:", sizes)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        repetitions = 1 if size == "all" else 10

        for repetition in range(repetitions):
            seed = 42 + repetition

            splits, trait_split, metadata = make_splits(
                labels_path=args.labels,
                n_subjects=size,
                seed=seed,
                train_fraction=args.train_fraction,
            )

            validate_splits(
                splits=splits,
                trait_split=trait_split,
                metadata=metadata,
            )

            if size == "all":
                filename = "splits_nall.pkl"
            else:
                filename = f"splits_n{size}_rep{repetition}.pkl"

            output_path = output_dir / filename

            joblib.dump(
                {
                    "splits": splits,
                    "trait_split": trait_split,
                    "metadata": metadata,
                },
                output_path,
            )

            print(
                f"Saved {output_path} "
                f"| rows={len(metadata)} "
                f"| train={len(trait_split['train'])} "
                f"| test={len(trait_split['test'])}"
            )

    print("ERN splits generated.")


if __name__ == "__main__":
    main()