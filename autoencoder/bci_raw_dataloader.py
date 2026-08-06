# datasets/bnci2014_002_raw_ae_dataloader.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from moabb.datasets import BNCI2014_002
from moabb.paradigms import MotorImagery
from torch.utils.data import DataLoader, Dataset, Subset

#pull data from moabb
def recreate_balanced_bnci_data(
    seed: int = 0,
    balance: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Recreate the exact balanced raw-trial ordering produced by
    make_bnci2014_002().
    """
    rng = np.random.default_rng(seed)

    dataset = BNCI2014_002()

    paradigm = MotorImagery(
        events=["right_hand", "feet"],
        n_classes=2,
        tmin=3.0,
        tmax=8.0,
    )

    X_epochs, y, metadata = paradigm.get_data(
        dataset=dataset,
    )

    rows_by_group: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for index in range(len(X_epochs)):
        subject = str(
            metadata.iloc[index]["subject"]
        )

        condition = str(y[index])

        session = str(
            metadata.iloc[index].get(
                "session",
                "session_0",
            )
        )

        run = str(
            metadata.iloc[index].get(
                "run",
                "run_0",
            )
        )

        key = (
            subject,
            condition,
        )

        rows_by_group.setdefault(
            key,
            [],
        ).append(
            {
                "trial": X_epochs[index].astype(
                    np.float32
                ),
                "moabb_index": index,
                "subject": subject,
                "condition": condition,
                "session": session,
                "run": run,
            }
        )

    selected_rows: list[dict[str, Any]] = []

    if balance:
        subjects = sorted(
            {
                subject
                for subject, _ in rows_by_group
            }
        )

        for subject in subjects:
            subject_keys = [
                key
                for key in rows_by_group
                if key[0] == subject
            ]

            if len(subject_keys) < 2:
                continue

            min_trials = min(
                len(rows_by_group[key])
                for key in subject_keys
            )

            for key in subject_keys:
                group = rows_by_group[key]

                chosen = rng.choice(
                    len(group),
                    size=min_trials,
                    replace=False,
                )

                selected_rows.extend(
                    group[int(index)]
                    for index in chosen
                )

    else:
        for group in rows_by_group.values():
            selected_rows.extend(group)

    if not selected_rows:
        raise ValueError(
            "No BNCI2014_002 trials remained."
        )

    balanced_epochs = np.stack(
        [
            row["trial"]
            for row in selected_rows
        ]
    ).astype(np.float32)

    labels = pd.DataFrame(
        [
            {
                "subject": row["subject"],
                "condition": row["condition"],
                "session": row["session"],
                "run": row["run"],
                "moabb_index": row["moabb_index"],
            }
            for row in selected_rows
        ]
    )

    labels["original_index"] = np.arange(
        len(labels)
    )

    return balanced_epochs, labels


def validate_indices(
    indices: np.ndarray,
    n_rows: int,
    name: str,
) -> None:
    if len(indices) == 0:
        raise ValueError(
            f"{name} partition is empty."
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
    Return outer train/test indices for one evaluation protocol.

    trait:
        One global split.

    within_state:
        One split for each target subject.

    between_state:
        One split for each target subject.
    """
    if split_type == "trait":
        if target_subject is not None:
            raise ValueError(
                "target_subject must be None "
                "for the trait split."
            )

        outer_train = np.asarray(
            split_data["trait_split"]["train"],
            dtype=int,
        )

        outer_test = np.asarray(
            split_data["trait_split"]["test"],
            dtype=int,
        )

        return outer_train, outer_test

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
            f"split_type={split_type}."
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
            f"Subject {target_subject} was not found "
            f"in the split file. Available subjects: "
            f"{available_subjects}"
        )

    outer_train = np.asarray(
        subject_entry[split_type]["train"],
        dtype=int,
    )

    outer_test = np.asarray(
        subject_entry[split_type]["test"],
        dtype=int,
    )

    return outer_train, outer_test


def stratified_train_validation_split(
    metadata: pd.DataFrame,
    outer_train_indices: np.ndarray,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split only the current outer training partition into
    autoencoder training and validation sets.

    Splitting is performed within subject-condition groups.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(seed)

    outer_train_metadata = metadata.iloc[
        outer_train_indices
    ]

    train_indices: list[int] = []
    validation_indices: list[int] = []

    for _, group in outer_train_metadata.groupby(
        ["subject", "condition"],
        sort=False,
    ):
        indices = group.index.to_numpy(
            dtype=int
        )

        shuffled = rng.permutation(
            indices
        )

        if len(shuffled) < 2:
            train_indices.extend(
                shuffled.tolist()
            )
            continue

        n_validation = int(
            round(
                len(shuffled)
                * validation_fraction
            )
        )

        n_validation = max(
            1,
            min(
                n_validation,
                len(shuffled) - 1,
            ),
        )

        validation_indices.extend(
            shuffled[:n_validation].tolist()
        )

        train_indices.extend(
            shuffled[n_validation:].tolist()
        )

    if not validation_indices:
        raise ValueError(
            "No validation samples could be created."
        )

    return (
        np.asarray(
            train_indices,
            dtype=int,
        ),
        np.asarray(
            validation_indices,
            dtype=int,
        ),
    )


class BNCIRawDataset(Dataset):
    """
    Dataset over an already reconstructed and aligned
    BNCI2014_002 raw-trial matrix.
    """

    def __init__(
        self,
        X_epochs: np.ndarray,
        metadata: pd.DataFrame,
    ):
        if len(X_epochs) != len(metadata):
            raise ValueError(
                f"Epoch/metadata mismatch: "
                f"{len(X_epochs)} versus {len(metadata)}."
            )

        self.X_epochs = X_epochs

        self.metadata = metadata.reset_index(
            drop=True
        )

        condition_values = sorted(
            self.metadata["condition"]
            .astype(str)
            .unique()
        )

        self.condition_to_id = {
            condition: index
            for index, condition in enumerate(
                condition_values
            )
        }

    def __len__(self) -> int:
        return len(self.X_epochs)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.metadata.iloc[index]

        x = torch.from_numpy(
            self.X_epochs[index]
        ).float()

        condition_text = str(
            row["condition"]
        )

        return {
            "x": x,

            # Autoencoder reconstruction target.
            "labels": x,

            "subject": torch.tensor(
                int(row["subject"]),
                dtype=torch.long,
            ),

            "condition": torch.tensor(
                self.condition_to_id[
                    condition_text
                ],
                dtype=torch.long,
            ),

            "condition_text": condition_text,
            "session": str(row["session"]),
            "run": str(row["run"]),

            "original_index": torch.tensor(
                int(row["original_index"]),
                dtype=torch.long,
            ),

            "moabb_index": torch.tensor(
                int(row["moabb_index"]),
                dtype=torch.long,
            ),
        }


def make_bnci_ae_dataloaders(
    split_path: str | os.PathLike[str],
    split_type: str = "trait",
    target_subject: int | str | None = None,
    batch_size: int = 128,
    generation_seed: int = 0,
    balance: bool = True,
    validation_fraction: float = 0.1,
    validation_seed: int = 42,
    num_workers: int = 4,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Build BNCI train, validation, and test loaders for one
    evaluation split.

    For trait:
        target_subject must be None.

    For within_state and between_state:
        target_subject selects the corresponding per-subject split.
    """
    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    split_data = joblib.load(
        split_path
    )

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
            f"{split_path.name} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    split_metadata = split_data[
        "metadata"
    ].copy()

    split_metadata.columns = (
        split_metadata.columns
        .str.lower()
        .str.strip()
    )

    required_columns = {
        "subject",
        "condition",
        "session",
        "run",
        "original_index",
    }

    missing_columns = (
        required_columns
        - set(split_metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{split_path.name} metadata is missing: "
            f"{sorted(missing_columns)}"
        )

    X_full, recreated_metadata = (
        recreate_balanced_bnci_data(
            seed=generation_seed,
            balance=balance,
        )
    )

    original_indices = split_metadata[
        "original_index"
    ].to_numpy(dtype=int)

    if len(original_indices) == 0:
        raise ValueError(
            f"{split_path.name} contains no rows."
        )

    if (
        original_indices.min() < 0
        or original_indices.max() >= len(X_full)
    ):
        raise IndexError(
            f"{split_path.name} contains original_index "
            f"outside 0–{len(X_full) - 1}."
        )

    # Select the reconstructed raw trials belonging to
    # this subject group and repetition.
    X_group = X_full[
        original_indices
    ]

    recreated_group_metadata = (
        recreated_metadata
        .iloc[original_indices]
        .reset_index(drop=True)
    )

    split_metadata = (
        split_metadata
        .reset_index(drop=True)
    )

    # Confirm that reconstruction produced exactly the same
    # ordering used when the split file was generated.
    comparison_columns = [
        "subject",
        "condition",
        "session",
        "run",
    ]

    expected = (
        split_metadata[
            comparison_columns
        ]
        .astype(str)
        .reset_index(drop=True)
    )

    recreated = (
        recreated_group_metadata[
            comparison_columns
        ]
        .astype(str)
        .reset_index(drop=True)
    )

    if not expected.equals(recreated):
        mismatch_mask = (
            expected != recreated
        ).any(axis=1)

        examples = pd.concat(
            {
                "split": expected[
                    mismatch_mask
                ].head(),

                "recreated": recreated[
                    mismatch_mask
                ].head(),
            },
            axis=1,
        )

        raise ValueError(
            "Recreated BNCI raw-trial ordering does not "
            "match the saved split metadata. Check the "
            "generation seed, balancing settings, or "
            f"preprocessing.\n{examples}"
        )

    # Keep the split-file metadata because the split indices
    # refer to its local row ordering.
    dataset_metadata = (
        split_metadata.copy()
    )

    dataset_metadata["moabb_index"] = (
        recreated_group_metadata[
            "moabb_index"
        ].to_numpy()
    )

    full_group_dataset = BNCIRawDataset(
        X_epochs=X_group,
        metadata=dataset_metadata,
    )

    outer_train, outer_test = (
        get_outer_split_indices(
            split_data=split_data,
            split_type=split_type,
            target_subject=target_subject,
        )
    )

    validate_indices(
        outer_train,
        len(full_group_dataset),
        f"{split_type} outer train",
    )

    validate_indices(
        outer_test,
        len(full_group_dataset),
        f"{split_type} outer test",
    )

    overlap = np.intersect1d(
        outer_train,
        outer_test,
    )

    if overlap.size:
        raise ValueError(
            f"{split_type} outer train and "
            "test indices overlap."
        )

    ae_train, ae_validation = (
        stratified_train_validation_split(
            metadata=dataset_metadata,
            outer_train_indices=outer_train,
            validation_fraction=(
                validation_fraction
            ),
            seed=validation_seed,
        )
    )

    train_dataset = Subset(
        full_group_dataset,
        ae_train.tolist(),
    )

    validation_dataset = Subset(
        full_group_dataset,
        ae_validation.tolist(),
    )

    test_dataset = Subset(
        full_group_dataset,
        outer_test.tolist(),
    )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (
            torch.cuda.is_available()
        ),
        "persistent_workers": (
            num_workers > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_args,
    )

    print(
        f"[BNCI] split={split_path.name}"
    )

    print(
        f"[BNCI] type={split_type}, "
        f"target_subject={target_subject}"
    )

    print(
        f"[BNCI] total subjects="
        f"{dataset_metadata['subject'].nunique()}"
    )

    print(
        f"[BNCI] AE train trials="
        f"{len(train_dataset)}"
    )

    print(
        f"[BNCI] AE validation trials="
        f"{len(validation_dataset)}"
    )

    print(
        f"[BNCI] AE test trials="
        f"{len(test_dataset)}"
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )