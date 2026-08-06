# datasets/gamma_raw_ae_dataloader.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


LABEL_MAP = {
    "left_hand": 1,
    "right_hand": 2,
    "rest": 3,
    "feet": 4,
}


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


def stratified_train_val_split(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the outer training partition into AE training and validation.

    Splitting is performed within each subject-condition group so that
    validation represents every subject and condition whenever possible.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(seed)

    train_rows = metadata.iloc[
        train_indices
    ]

    final_train_indices: list[int] = []
    validation_indices: list[int] = []

    for _, group in train_rows.groupby(
        ["subject", "condition"],
        sort=False,
    ):
        group_indices = group.index.to_numpy(
            dtype=int
        )

        shuffled = rng.permutation(
            group_indices
        )

        if len(shuffled) < 2:
            # Keep singleton groups in training.
            final_train_indices.extend(
                shuffled.tolist()
            )
            continue

        n_validation = int(
            round(
                len(shuffled)
                * val_fraction
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

        final_train_indices.extend(
            shuffled[n_validation:].tolist()
        )

    if not validation_indices:
        raise ValueError(
            "No validation samples could be created "
            "from the outer training partition."
        )

    return (
        np.asarray(
            final_train_indices,
            dtype=int,
        ),
        np.asarray(
            validation_indices,
            dtype=int,
        ),
    )


def get_outer_split_indices(
    split_data: dict[str, Any],
    split_type: str,
    target_subject: int | str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return outer train/test indices for one evaluation protocol.

    trait:
        One global split. target_subject must be None.

    within_state:
        One split per target subject.

    between_state:
        One split per target subject.
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


def resize_trial(
    trial: np.ndarray,
    target_samples: int,
) -> np.ndarray:
    """
    Crop or zero-pad a trial to a fixed number of samples.
    """
    current_samples = trial.shape[-1]

    if current_samples == target_samples:
        return trial

    if current_samples > target_samples:
        return trial[..., :target_samples]

    padding = (
        target_samples - current_samples
    )

    return np.pad(
        trial,
        pad_width=(
            (0, 0),
            (0, padding),
        ),
        mode="constant",
    )


class GammaSplitAEDataset(Dataset):
    """
    Load raw High-Gamma epochs selected by an existing split file.

    Required split metadata columns:
        subject
        original_part
        file
        trial
        condition

    The saved trial value refers to the position in the MNE Epochs
    object recreated from that EDF file.
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        split_path: str | os.PathLike[str],
        partition: str,
        split_type: str = "trait",
        target_subject: int | str | None = None,
        val_fraction: float = 0.1,
        validation_seed: int = 42,
        target_sfreq: float | None = None,
        tmin: float = 0.0,
        tmax: float = 4.0,
    ):
        if partition not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                "partition must be 'train', "
                "'val', or 'test'."
            )

        self.data_root = Path(data_root)
        self.split_path = Path(split_path)
        self.partition = partition
        self.split_type = split_type
        self.target_subject = target_subject
        self.target_sfreq = target_sfreq
        self.tmin = float(tmin)
        self.tmax = float(tmax)

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Gamma data root does not exist: "
                f"{self.data_root}"
            )

        if not self.split_path.exists():
            raise FileNotFoundError(
                f"Split file does not exist: "
                f"{self.split_path}"
            )

        split_data = joblib.load(
            self.split_path
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
                f"{self.split_path.name} is missing keys: "
                f"{sorted(missing_keys)}"
            )

        metadata = split_data[
            "metadata"
        ].copy()

        metadata.columns = (
            metadata.columns
            .str.lower()
            .str.strip()
        )

        required_columns = {
            "subject",
            "original_part",
            "file",
            "trial",
            "condition",
        }

        missing_columns = (
            required_columns
            - set(metadata.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{self.split_path.name} metadata is "
                f"missing columns: "
                f"{sorted(missing_columns)}"
            )

        metadata["subject"] = (
            metadata["subject"].astype(int)
        )

        metadata["original_part"] = (
            metadata["original_part"].astype(str)
        )

        metadata["file"] = (
            metadata["file"].astype(str)
        )

        metadata["trial"] = (
            metadata["trial"].astype(int)
        )

        metadata["condition"] = (
            metadata["condition"].astype(int)
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
            len(metadata),
            f"{split_type} outer train",
        )

        validate_indices(
            outer_test,
            len(metadata),
            f"{split_type} outer test",
        )

        overlap = np.intersect1d(
            outer_train,
            outer_test,
        )

        if overlap.size:
            raise ValueError(
                f"{split_type} outer train and "
                "test partitions overlap."
            )

        ae_train, ae_validation = (
            stratified_train_val_split(
                metadata=metadata,
                train_indices=outer_train,
                val_fraction=val_fraction,
                seed=validation_seed,
            )
        )

        if partition == "train":
            selected_indices = ae_train

        elif partition == "val":
            selected_indices = ae_validation

        else:
            selected_indices = outer_test

        self.metadata = (
            metadata
            .iloc[selected_indices]
            .copy()
            .reset_index(drop=True)
        )

        # Each DataLoader worker maintains its own cache.
        # Key: (original_part, filename)
        self._epoch_cache: dict[
            tuple[str, str],
            tuple[
                np.ndarray,
                np.ndarray,
                float,
            ],
        ] = {}

        condition_counts = (
            self.metadata["condition"]
            .value_counts()
            .sort_index()
        )

        print(
            f"[Gamma {partition}] "
            f"split={self.split_path.name}"
        )

        print(
            f"[Gamma {partition}] "
            f"type={split_type}, "
            f"target_subject={target_subject}"
        )

        print(
            f"[Gamma {partition}] "
            f"subjects="
            f"{self.metadata['subject'].nunique()}, "
            f"trials={len(self.metadata)}"
        )

        print(
            f"[Gamma {partition}] conditions:\n"
            f"{condition_counts}"
        )

    def _load_file_epochs(
        self,
        original_part: str,
        filename: str,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """
        Reproduce the preprocessing and epoching from make_gamma_all().
        """
        cache_key = (
            original_part,
            filename,
        )

        if cache_key in self._epoch_cache:
            return self._epoch_cache[
                cache_key
            ]

        edf_path = (
            self.data_root
            / original_part
            / filename
        )

        if not edf_path.exists():
            raise FileNotFoundError(
                f"Gamma EDF file not found: "
                f"{edf_path}"
            )

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            infer_types=True,
            verbose=False,
        )

        original_sfreq = float(
            raw.info["sfreq"]
        )

        # Must match the preprocessing used when the
        # Gamma split metadata was created.
        raw.filter(
            1.0,
            45.0,
            verbose=False,
        )

        raw.notch_filter(
            60.0,
            verbose=False,
        )

        events, _ = (
            mne.events_from_annotations(
                raw,
                event_id=LABEL_MAP,
                verbose=False,
            )
        )

        epochs = mne.Epochs(
            raw,
            events,
            event_id=LABEL_MAP,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )

        data = epochs.get_data(
            copy=True,
        ).astype(np.float32)

        labels = (
            epochs.events[:, 2]
            .astype(int)
        )

        raw.close()

        final_sfreq = original_sfreq

        if (
            self.target_sfreq is not None
            and not np.isclose(
                original_sfreq,
                self.target_sfreq,
            )
        ):
            data = mne.filter.resample(
                data,
                up=float(
                    self.target_sfreq
                ),
                down=original_sfreq,
                axis=-1,
                verbose=False,
            ).astype(np.float32)

            final_sfreq = float(
                self.target_sfreq
            )

        # MNE includes the endpoint, so a 0–4 second epoch
        # often contains 4 * sfreq + 1 samples.
        target_samples = (
            int(
                round(
                    (self.tmax - self.tmin)
                    * final_sfreq
                )
            )
            + 1
        )

        if data.shape[-1] != target_samples:
            data = np.stack(
                [
                    resize_trial(
                        trial=trial,
                        target_samples=(
                            target_samples
                        ),
                    )
                    for trial in data
                ]
            ).astype(np.float32)

        result = (
            data,
            labels,
            final_sfreq,
        )

        self._epoch_cache[
            cache_key
        ] = result

        return result

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.metadata.iloc[index]

        original_part = str(
            row["original_part"]
        )

        filename = str(
            row["file"]
        )

        trial_index = int(
            row["trial"]
        )

        expected_condition = int(
            row["condition"]
        )

        data, labels, sfreq = (
            self._load_file_epochs(
                original_part=original_part,
                filename=filename,
            )
        )

        if (
            trial_index < 0
            or trial_index >= len(data)
        ):
            raise IndexError(
                f"Trial {trial_index} is invalid for "
                f"{filename}; the recreated Epochs "
                f"object contains {len(data)} trials."
            )

        actual_condition = int(
            labels[trial_index]
        )

        if actual_condition != expected_condition:
            raise ValueError(
                f"Condition mismatch for {filename}, "
                f"trial {trial_index}: split metadata "
                f"says {expected_condition}, but the "
                f"recreated raw epoch says "
                f"{actual_condition}."
            )

        x = torch.from_numpy(
            data[trial_index]
        ).float()

        item: dict[str, Any] = {
            "x": x,

            # Autoencoder reconstruction target.
            "labels": x,

            "subject": torch.tensor(
                int(row["subject"]),
                dtype=torch.long,
            ),

            "condition": torch.tensor(
                expected_condition,
                dtype=torch.long,
            ),

            "trial": torch.tensor(
                trial_index,
                dtype=torch.long,
            ),

            "sfreq": torch.tensor(
                sfreq,
                dtype=torch.float32,
            ),

            "original_part": (
                original_part
            ),

            "file": filename,
        }

        if (
            "original_index"
            in self.metadata.columns
        ):
            item["original_index"] = (
                torch.tensor(
                    int(
                        row["original_index"]
                    ),
                    dtype=torch.long,
                )
            )

        return item


def make_gamma_ae_dataloaders(
    data_root: str | os.PathLike[str],
    split_path: str | os.PathLike[str],
    split_type: str = "trait",
    target_subject: int | str | None = None,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    validation_seed: int = 42,
    target_sfreq: float | None = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
    num_workers: int = 4,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Construct train, validation, and test loaders for one
    Gamma split protocol.

    For trait:
        target_subject must be None.

    For within_state and between_state:
        target_subject identifies the per-subject split.
    """
    common_args = {
        "data_root": data_root,
        "split_path": split_path,
        "split_type": split_type,
        "target_subject": target_subject,
        "val_fraction": val_fraction,
        "validation_seed": validation_seed,
        "target_sfreq": target_sfreq,
        "tmin": tmin,
        "tmax": tmax,
    }

    train_dataset = GammaSplitAEDataset(
        partition="train",
        **common_args,
    )

    val_dataset = GammaSplitAEDataset(
        partition="val",
        **common_args,
    )

    test_dataset = GammaSplitAEDataset(
        partition="test",
        **common_args,
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

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_args,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )