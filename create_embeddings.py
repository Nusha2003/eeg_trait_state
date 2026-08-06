from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from representation_utils import (
    fit_space,
    get_seed_id,
    get_space_labels,
    group_split_files,
    load_experiment_config,
    tune_joint_dims_for_group,
)


SPLIT_TYPES = [
    "trait",
    "within_state",
    "between_state",
]


def validate_indices(
    indices: np.ndarray,
    n_rows: int,
    name: str,
) -> None:
    if len(indices) == 0:
        raise ValueError(
            f"{name} is empty."
        )

    if indices.min() < 0 or indices.max() >= n_rows:
        raise IndexError(
            f"{name} indices must be between "
            f"0 and {n_rows - 1}."
        )


def get_outer_split_indices(
    split_data: dict[str, Any],
    split_type: str,
    target_subject: int | str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get train/test indices for one split protocol.
    """
    if split_type == "trait":
        if target_subject is not None:
            raise ValueError(
                "target_subject must be None "
                "for the trait split."
            )

        train_indices = np.asarray(
            split_data["trait_split"]["train"],
            dtype=int,
        )

        test_indices = np.asarray(
            split_data["trait_split"]["test"],
            dtype=int,
        )

        return train_indices, test_indices

    if split_type not in {
        "within_state",
        "between_state",
    }:
        raise ValueError(
            "split_type must be 'trait', "
            "'within_state', or 'between_state'."
        )

    if target_subject is None:
        raise ValueError(
            f"target_subject is required for "
            f"{split_type}."
        )

    subject_entry = next(
        (
            entry
            for entry in split_data["splits"]
            if str(entry["subject"])
            == str(target_subject)
        ),
        None,
    )

    if subject_entry is None:
        available_subjects = [
            entry["subject"]
            for entry in split_data["splits"]
        ]

        raise ValueError(
            f"Subject {target_subject} was not found. "
            f"Available subjects: {available_subjects}"
        )

    train_indices = np.asarray(
        subject_entry[split_type]["train"],
        dtype=int,
    )

    test_indices = np.asarray(
        subject_entry[split_type]["test"],
        dtype=int,
    )

    return train_indices, test_indices


def make_output_path(
    output_dir: Path,
    split_type: str,
    subject_group: str,
    seed_id: int | str,
    target_subject: int | str | None,
) -> Path:
    """
    Construct a separate output path for every split.
    """
    group_dir = (
        output_dir
        / split_type
        / f"n{subject_group}_seed{seed_id}"
    )

    if target_subject is None:
        group_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return (
            group_dir
            / "embeddings.pkl"
        )

    subject_dir = (
        group_dir
        / f"subject_{target_subject}"
    )

    subject_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        subject_dir
        / "embeddings.pkl"
    )


def process_one_split(
    *,
    split_data: dict[str, Any],
    split_file: str,
    metadata: pd.DataFrame,
    X: np.ndarray,
    subject_group: str,
    seed_id: int | str,
    split_type: str,
    target_subject: int | str | None,
    spaces: list[str],
    output_dir: Path,
    joint_pca_dim: int,
    joint_lda_dim: int,
) -> Path:
    """
    Fit all representation spaces using one split's training data
    and save that split's train/test embeddings.
    """
    train_indices, test_indices = (
        get_outer_split_indices(
            split_data=split_data,
            split_type=split_type,
            target_subject=target_subject,
        )
    )

    validate_indices(
        train_indices,
        len(metadata),
        f"{split_type} train",
    )

    validate_indices(
        test_indices,
        len(metadata),
        f"{split_type} test",
    )

    overlap = np.intersect1d(
        train_indices,
        test_indices,
    )

    if overlap.size:
        raise ValueError(
            f"{split_type} train and test indices overlap "
            f"for target_subject={target_subject}."
        )

    print(
        f"Creating {split_type} embeddings | "
        f"file={split_file} | "
        f"target_subject={target_subject}"
    )

    train_metadata = (
        metadata
        .iloc[train_indices]
        .copy()
        .reset_index(drop=True)
    )

    test_metadata = (
        metadata
        .iloc[test_indices]
        .copy()
        .reset_index(drop=True)
    )

    bundle = {
        "subject_group": subject_group,
        "seed": seed_id,
        "source_split_file": split_file,
        "split_type": split_type,
        "target_subject": target_subject,

        "train_indices": train_indices,
        "test_indices": test_indices,

        "train_metadata": train_metadata,
        "test_metadata": test_metadata,

        "spaces": {},
    }

    for space_name in spaces:
        labels = get_space_labels(
            metadata,
            space_name,
        )

        if space_name == "Joint_LDA":
            pca_dim = joint_pca_dim
            lda_dim = joint_lda_dim
        else:
            pca_dim = 30
            lda_dim = 10

        # Fit only on this split's training partition.
        transformer = fit_space(
            X=X,
            y=labels,
            train_idx=train_indices,
            space_type=space_name,
            pca_dim=pca_dim,
            lda_dim=lda_dim,
        )

        # Transform train and test separately.
        X_train = transformer.transform(
            X[train_indices]
        )

        X_test = transformer.transform(
            X[test_indices]
        )

        bundle["spaces"][space_name] = {
            "X_train": X_train,
            "X_test": X_test,
            "transformer": transformer,
            "pca_dim": pca_dim,
            "lda_dim": lda_dim,
        }

    output_path = make_output_path(
        output_dir=output_dir,
        split_type=split_type,
        subject_group=subject_group,
        seed_id=seed_id,
        target_subject=target_subject,
    )

    joblib.dump(
        bundle,
        output_path,
        compress=3,
    )

    return output_path


def process_split_file(
    split_file: str,
    split_dir: Path,
    X_full: np.ndarray,
    subject_group: str,
    spaces: list[str],
    output_dir: Path,
    joint_pca_dim: int,
    joint_lda_dim: int,
) -> list[Path]:
    """
    Process trait, within-state, and between-state splits
    stored in one split file.
    """
    print(
        f"\nLoading split file: {split_file}"
    )

    split_path = split_dir / split_file
    split_data = joblib.load(split_path)

    required_keys = {
        "metadata",
        "trait_split",
        "splits",
    }

    missing_keys = (
        required_keys - set(split_data)
    )

    if missing_keys:
        raise ValueError(
            f"{split_file} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    metadata = (
        split_data["metadata"]
        .copy()
    )

    metadata.columns = (
        metadata.columns
        .str.lower()
        .str.strip()
    )

    if "original_index" not in metadata.columns:
        raise ValueError(
            f"{split_file} is missing original_index."
        )

    original_indices = metadata[
        "original_index"
    ].to_numpy(dtype=int)

    if len(original_indices) == 0:
        raise ValueError(
            f"{split_file} contains no metadata rows."
        )

    if (
        original_indices.min() < 0
        or original_indices.max() >= len(X_full)
    ):
        raise IndexError(
            f"{split_file} contains original_index values "
            f"outside 0–{len(X_full) - 1}."
        )

    X = X_full[
        original_indices
    ]

    if len(metadata) != len(X):
        raise ValueError(
            f"Feature rows ({len(X)}) do not match "
            f"metadata rows ({len(metadata)}) "
            f"for {split_file}."
        )

    seed_id = get_seed_id(
        split_file
    )

    target_subjects = [
        entry["subject"]
        for entry in split_data["splits"]
    ]

    jobs: list[
        tuple[str, int | str | None]
    ] = [
        ("trait", None)
    ]

    jobs.extend(
        (
            "within_state",
            subject,
        )
        for subject in target_subjects
    )

    jobs.extend(
        (
            "between_state",
            subject,
        )
        for subject in target_subjects
    )

    output_paths: list[Path] = []

    for split_type, target_subject in jobs:
        output_path = process_one_split(
            split_data=split_data,
            split_file=split_file,
            metadata=metadata,
            X=X,
            subject_group=subject_group,
            seed_id=seed_id,
            split_type=split_type,
            target_subject=target_subject,
            spaces=spaces,
            output_dir=output_dir,
            joint_pca_dim=joint_pca_dim,
            joint_lda_dim=joint_lda_dim,
        )

        output_paths.append(
            output_path
        )

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and save separate EEG representation "
            "spaces for trait, within-state, and "
            "between-state splits."
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    experiment = load_experiment_config(
        args.config
    )

    split_dir = experiment["split_dir"]
    save_dir = experiment["save_dir"]
    spaces = experiment["spaces"]

    embedding_dir = (
        save_dir
        / "embeddings"
    )

    tune_dir = (
        save_dir
        / "dimension_tuning"
    )

    X_full = pd.read_csv(
        experiment["feature_path"]
    ).to_numpy()

    groups = group_split_files(
        split_dir
    )

    if not groups:
        raise FileNotFoundError(
            f"No split files found in {split_dir}"
        )

    print(
        "Detected subject groups:",
        list(groups),
    )

    for subject_group, files in groups.items():
        # This preserves your existing group-level tuning.
        best_pca, best_lda = (
            tune_joint_dims_for_group(
                files=files,
                split_dir=split_dir,
                X=X_full,
                subject_group=subject_group,
                tune_dir=tune_dir,
                dataset=experiment["dataset"],
                feature=experiment["feature"],
                num_classes=experiment["num_classes"],
            )
        )

        nested_outputs = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
        )(
            delayed(process_split_file)(
                split_file=split_file,
                split_dir=split_dir,
                X_full=X_full,
                subject_group=subject_group,
                spaces=spaces,
                output_dir=embedding_dir,
                joint_pca_dim=best_pca,
                joint_lda_dim=best_lda,
            )
            for split_file in files
        )

        for outputs in nested_outputs:
            for output in outputs:
                print(
                    f"Saved {output}"
                )


if __name__ == "__main__":
    main()