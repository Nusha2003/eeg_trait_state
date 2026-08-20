# datasets/physionet_raw_ae_dataloader.py

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any

import joblib
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from collections import OrderedDict

IGNORED_SUBJECTS = {88, 90, 92, 100}


def run_to_condition(run: int) -> int:
    mapping = {
        1: 0,
        2: 1,

        3: 2,
        7: 2,
        11: 2,

        4: 3,
        8: 3,
        12: 3,

        5: 4,
        9: 4,
        13: 4,

        6: 5,
        10: 5,
        14: 5,
    }

    if run not in mapping:
        raise ValueError(
            f"Unknown PhysioNet run: {run}"
        )

    return mapping[run]


def parse_subject_run(
    path: str | os.PathLike[str],
) -> tuple[int, int]:
    name = os.path.basename(path)

    match = re.match(
        r"S(\d{3})R(\d{2})\.edf",
        name,
    )

    if match is None:
        raise ValueError(
            f"Bad EDF filename: {name}"
        )

    subject = int(match.group(1))
    run = int(match.group(2))

    return subject, run


def get_physionet_files(
    data_root: str | os.PathLike[str],
) -> list[str]:
    edf_files = sorted(
        glob.glob(
            os.path.join(
                str(data_root),
                "S*",
                "S*R*.edf",
            )
        )
    )

    usable = []

    for path in edf_files:
        subject, _ = parse_subject_run(path)

        if subject in IGNORED_SUBJECTS:
            continue

        usable.append(path)

    if not usable:
        raise ValueError(
            f"No usable PhysioNet EDF files found "
            f"under {data_root}."
        )

    return usable


def validate_indices(
    indices: np.ndarray,
    n_rows: int,
    name: str,
) -> None:
    if len(indices) == 0:
        raise ValueError(
            f"{name} partition is empty."
        )

    if (
        indices.min() < 0
        or indices.max() >= n_rows
    ):
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
    Return outer train/test indices for one evaluation
    protocol.

    trait:
        One global train/test split.

    within_state:
        One train/test split per target subject.

    between_state:
        One train/test split per target subject.
    """

    if split_type == "trait":
        if target_subject is not None:
            raise ValueError(
                "target_subject must be None "
                "for the trait split."
            )

        outer_train = np.asarray(
            split_data[
                "trait_split"
            ]["train"],
            dtype=int,
        )

        outer_test = np.asarray(
            split_data[
                "trait_split"
            ]["test"],
            dtype=int,
        )

        return (
            outer_train,
            outer_test,
        )

    if split_type not in {
        "within_state",
        "between_state",
    }:
        raise ValueError(
            "split_type must be 'trait', "
            "'within_state', or "
            "'between_state'."
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
        subject_entry[
            split_type
        ]["train"],
        dtype=int,
    )

    outer_test = np.asarray(
        subject_entry[
            split_type
        ]["test"],
        dtype=int,
    )

    return (
        outer_train,
        outer_test,
    )


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
            "validation_fraction must be "
            "between 0 and 1."
        )

    rng = np.random.default_rng(
        seed
    )

    outer_train_metadata = metadata.iloc[
        outer_train_indices
    ]

    train_indices: list[int] = []
    validation_indices: list[int] = []

    for _, group in (
        outer_train_metadata.groupby(
            ["subject", "condition"],
            sort=False,
        )
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
            shuffled[
                :n_validation
            ].tolist()
        )

        train_indices.extend(
            shuffled[
                n_validation:
            ].tolist()
        )

    if not validation_indices:
        raise ValueError(
            "No validation samples could "
            "be created."
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

class PhysioNetRawDataset(Dataset):
    def __init__(
        self,
        data_root,
        metadata,
        target_sfreq=160.0,
        epoch_len_sec=4.0,
    ):
        self.data_root = Path(data_root)
        self.metadata = metadata.reset_index(drop=True)
        self.target_sfreq = target_sfreq
        self._raw_cache = OrderedDict()
        self._max_cache_files = 2
        self.epoch_len_sec = epoch_len_sec
        self.condition_to_id = {
            condition: i
            for i, condition in enumerate(
                sorted(self.metadata["condition"].astype(str).unique())
            )
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        subject = str(row["subject"])
        filename = str(row["file"])

        edf_path = self.data_root / subject / filename

        if not edf_path.exists():
            raise FileNotFoundError(
                f"EDF not found: {edf_path}"
            )

        # 1. Load/cached EDF
        raw = self._load_raw(edf_path)

        original_sfreq = float(raw.info["sfreq"])

        # 2. Extract exact segment
        start = int(row["start_sample"])
        end = int(row["end_sample"])

        x = raw.get_data(
            start=start,
            stop=end,
        ).astype(np.float32)

        # 3. Resample if necessary
        final_sfreq = original_sfreq

        if (
            self.target_sfreq is not None
            and not np.isclose(
                original_sfreq,
                self.target_sfreq,
            )
        ):
            x = mne.filter.resample(
                x,
                up=float(self.target_sfreq),
                down=original_sfreq,
                axis=-1,
                verbose=False,
            ).astype(np.float32)

            final_sfreq = float(self.target_sfreq)

        # ==========================================
        # 4. ADD THE 4-SECOND LOGIC HERE
        # ==========================================

        target_samples = int(
            round(
                self.epoch_len_sec
                * final_sfreq
            )
        )

        if x.shape[-1] > target_samples:
            # Too long -> crop to 4 seconds
            x = x[:, :target_samples]

        elif x.shape[-1] < target_samples:
            # Too short -> zero-pad to 4 seconds
            pad = target_samples - x.shape[-1]

            x = np.pad(
                x,
                ((0, 0), (0, pad)),
                mode="constant",
            )

        # 5. Now convert to tensor
        x = torch.from_numpy(x).float()

        condition = str(row["condition"])

        return {
            "x": x,
            "labels": x,

            "subject": torch.tensor(
                int(subject.replace("S", "")),
                dtype=torch.long,
            ),
            "condition": torch.tensor(
                self.condition_to_id[condition],
                dtype=torch.long,
            ),

            "run": torch.tensor(
                int(row["run"]),
                dtype=torch.long,
            ),

            "original_index": torch.tensor(
                int(row["original_index"]),
                dtype=torch.long,
            ),
        }
    def _load_raw(self, edf_path):
        cache_key = str(edf_path)

        if cache_key in self._raw_cache:
            raw = self._raw_cache.pop(cache_key)
            self._raw_cache[cache_key] = raw
            return raw

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose=False,
        )

        self._raw_cache[cache_key] = raw

        if len(self._raw_cache) > self._max_cache_files:
            _, old_raw = self._raw_cache.popitem(last=False)
            old_raw.close()

        return raw
def make_physionet_ae_dataloaders(
    data_root: str | os.PathLike[str],
    split_path: str | os.PathLike[str],
    split_type: str = "trait",
    target_subject: int | str | None = None,
    batch_size: int = 128,
    validation_fraction: float = 0.1,
    validation_seed: int = 42,
    sfreq: float = 160.0,
    epoch_len_sec: float = 4.0,
    num_workers: int = 4,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Build PhysioNet train, validation, and test loaders
    for one saved evaluation split.

    For trait:
        target_subject must be None.

    For within_state and between_state:
        target_subject selects the corresponding
        per-subject split.
    """

    split_path = Path(
        split_path
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: "
            f"{split_path}"
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
        required_keys
        - set(split_data)
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

    # IMPORTANT:
    # epoch is intentionally NOT required here.
    #
    # The saved split only needs enough information to
    # locate the corresponding epochs in the deterministically
    # reconstructed full dataset.
    required_columns = {
        "subject",
        "condition",
        "run",
        "file",
        "start_sample",
        "end_sample",
        "original_index",
    }

    missing_columns = (
        required_columns
        - set(split_metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{split_path.name} metadata "
            f"is missing columns: "
            f"{sorted(missing_columns)}"
        )
    dataset_metadata = (
        split_metadata
        .reset_index(drop=True)
    )
    # Keep split metadata because the split indices
    # refer to its local row ordering.
    #
    # Add epoch from the reconstructed data instead
    # of requiring it to already exist in the .pkl.

    full_group_dataset = PhysioNetRawDataset(
        data_root=data_root,
        metadata=dataset_metadata,
        target_sfreq=sfreq,
        epoch_len_sec=epoch_len_sec,
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
        f"[PhysioNet] split="
        f"{split_path.name}"
    )

    print(
        f"[PhysioNet] type="
        f"{split_type}, "
        f"target_subject="
        f"{target_subject}"
    )

    print(
        f"[PhysioNet] total subjects="
        f"{dataset_metadata['subject'].nunique()}"
    )

    print(
        f"[PhysioNet] AE train epochs="
        f"{len(train_dataset)}"
    )

    print(
        f"[PhysioNet] AE validation epochs="
        f"{len(validation_dataset)}"
    )

    print(
        f"[PhysioNet] AE test epochs="
        f"{len(test_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )