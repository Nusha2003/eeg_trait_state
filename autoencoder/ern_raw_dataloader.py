# datasets/ern_raw_ae_dataloader.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MNE_DATA_PATH = "/scratch1/amadapur/mne_data"

Path(MNE_DATA_PATH).mkdir(parents=True, exist_ok=True)
os.environ["MNE_DATA"] = MNE_DATA_PATH

import mne

mne.set_config(
    "MNE_DATA",
    MNE_DATA_PATH,
    set_env=True,
)

import joblib
import numpy as np
import pandas as pd
import torch

from moabb.datasets import ErpCore2021_ERN
from moabb.paradigms import P300
from torch.utils.data import DataLoader, Dataset, Subset


def recreate_balanced_ern_data(
    seed: int = 0,
    min_trials_per_condition: int = 10,
    balance: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Reproduce the exact trial filtering, balancing, and ordering
    used when the ERN feature dataset was created.
    """
    rng = np.random.default_rng(seed)

    dataset = ErpCore2021_ERN()

    paradigm = P300(
        events=["Target", "NonTarget"],
        tmin=0.0,
        tmax=1.0,
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

    all_subjects = sorted(
        {
            subject
            for subject, _ in rows_by_group
        }
    )

    usable_subjects: list[str] = []

    for subject in all_subjects:
        subject_keys = [
            key
            for key in rows_by_group
            if key[0] == subject
        ]

        if len(subject_keys) < 2:
            continue

        minimum_count = min(
            len(rows_by_group[key])
            for key in subject_keys
        )

        if minimum_count >= min_trials_per_condition:
            usable_subjects.append(subject)

    if len(usable_subjects) < 4:
        raise ValueError(
            "Fewer than four usable ERN subjects remain."
        )

    selected_rows: list[dict[str, Any]] = []

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
                    group[int(index)]
                    for index in selected_indices
                )

        else:
            for key in subject_keys:
                selected_rows.extend(
                    rows_by_group[key]
                )

    if not selected_rows:
        raise ValueError(
            "No ERN trials remained after filtering."
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


def validate_split_indices(
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
            f"{name} indices must lie between "
            f"0 and {n_rows - 1}."
        )


def get_outer_split_indices(
    split_data: dict[str, Any],
    split_type: str,
    target_subject: int | str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the outer train/test indices for one evaluation protocol.
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
    Reserve validation trials from the current outer training partition.

    Splitting is stratified within subject-condition groups.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(seed)

    outer_train_metadata = metadata.iloc[
        outer_train_indices
    ]

    final_train: list[int] = []
    validation: list[int] = []

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
            final_train.extend(
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

        validation.extend(
            shuffled[:n_validation].tolist()
        )

        final_train.extend(
            shuffled[n_validation:].tolist()
        )

    if not validation:
        raise ValueError(
            "No validation trials could be created."
        )

    return (
        np.asarray(
            final_train,
            dtype=int,
        ),
        np.asarray(
            validation,
            dtype=int,
        ),
    )


class ERNRawDataset(Dataset):
    """
    Dataset over reconstructed and aligned ERN raw trials.
    """

    def __init__(
        self,
        X_epochs: np.ndarray,
        metadata: pd.DataFrame,
    ):
        if len(X_epochs) != len(metadata):
            raise ValueError(
                f"Trial/metadata mismatch: "
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


def make_ern_ae_dataloaders(
    split_path: str | os.PathLike[str],
    split_type: str = "trait",
    target_subject: int | str | None = None,
    batch_size: int = 128,
    generation_seed: int = 0,
    min_trials_per_condition: int = 10,
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
    Build ERN train, validation, and test loaders for one split protocol.

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
        recreate_balanced_ern_data(
            seed=generation_seed,
            min_trials_per_condition=(
                min_trials_per_condition
            ),
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
            f"{split_path.name} has original_index "
            f"values outside 0–{len(X_full) - 1}."
        )

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
            "Recreated ERN raw-trial ordering does not "
            "match the saved split metadata. This may "
            "mean the generation seed, balancing, or "
            f"preprocessing changed.\n{examples}"
        )

    dataset_metadata = (
        split_metadata.copy()
    )

    dataset_metadata["moabb_index"] = (
        recreated_group_metadata[
            "moabb_index"
        ].to_numpy()
    )

    full_group_dataset = ERNRawDataset(
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

    validate_split_indices(
        outer_train,
        len(full_group_dataset),
        f"{split_type} outer training",
    )

    validate_split_indices(
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
            f"{split_type} outer training and "
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[ERN] split={split_path.name}"
    )

    print(
        f"[ERN] type={split_type}, "
        f"target_subject={target_subject}"
    )

    print(
        f"[ERN] total subjects="
        f"{dataset_metadata['subject'].nunique()}"
    )

    print(
        f"[ERN] AE train trials="
        f"{len(train_dataset)}"
    )

    print(
        f"[ERN] AE validation trials="
        f"{len(validation_dataset)}"
    )

    print(
        f"[ERN] AE test trials="
        f"{len(test_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )