# datasets/gamma_raw_ae_dataloader.py

from __future__ import annotations

import hashlib
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


# ============================================================
# Split utilities
# ============================================================

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


def stratified_train_val_split(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:

    if not 0 < val_fraction < 1:
        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(seed)

    train_rows = metadata.iloc[
        train_indices
    ]

    final_train_indices = []
    validation_indices = []

    for _, group in train_rows.groupby(
        ["subject", "condition"],
        sort=False,
    ):
        group_indices = (
            group.index.to_numpy(dtype=int)
        )

        shuffled = rng.permutation(
            group_indices
        )

        if len(shuffled) < 2:
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
            "No validation samples could be created."
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

    if split_type == "trait":

        if target_subject is not None:
            raise ValueError(
                "target_subject must be None "
                "for trait split."
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
            "split_type must be "
            "'trait', 'within_state', "
            "or 'between_state'."
        )

    if target_subject is None:
        raise ValueError(
            f"target_subject required for "
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
        raise ValueError(
            f"Subject {target_subject} "
            "not found in split."
        )

    train_indices = np.asarray(
        subject_entry[
            split_type
        ]["train"],
        dtype=int,
    )

    test_indices = np.asarray(
        subject_entry[
            split_type
        ]["test"],
        dtype=int,
    )

    return train_indices, test_indices


# ============================================================
# Fixed-size trial helper
# ============================================================

def resize_trial(
    trial: np.ndarray,
    target_samples: int,
) -> np.ndarray:

    current_samples = (
        trial.shape[-1]
    )

    if current_samples == target_samples:
        return trial

    if current_samples > target_samples:
        return trial[
            ...,
            :target_samples,
        ]

    padding = (
        target_samples
        - current_samples
    )

    return np.pad(
        trial,
        pad_width=(
            (0, 0),
            (0, padding),
        ),
        mode="constant",
    )


# ============================================================
# Disk cache helpers
# ============================================================

def get_cache_paths(
    cache_dir: Path,
    original_part: str,
    filename: str,
    target_sfreq: float | None,
    tmin: float,
    tmax: float,
) -> tuple[Path, Path, Path]:

    # Include preprocessing parameters so changing them
    # automatically creates a different cache.
    cache_string = (
        f"{original_part}|"
        f"{filename}|"
        f"sfreq={target_sfreq}|"
        f"tmin={tmin}|"
        f"tmax={tmax}|"
        "bandpass=1-45|"
        "notch=60|"
        "version=1"
    )

    cache_hash = hashlib.sha1(
        cache_string.encode()
    ).hexdigest()[:16]

    stem = Path(filename).stem

    base = (
        cache_dir
        / f"{stem}_{cache_hash}"
    )

    return (
        Path(str(base) + "_data.npy"),
        Path(str(base) + "_labels.npy"),
        Path(str(base) + "_sfreq.npy"),
    )


def preprocess_gamma_file(
    data_root: Path,
    cache_dir: Path,
    original_part: str,
    filename: str,
    target_sfreq: float | None,
    tmin: float,
    tmax: float,
) -> None:
    """
    Process one EDF once and save its epochs to disk.
    """

    data_path, labels_path, sfreq_path = (
        get_cache_paths(
            cache_dir=cache_dir,
            original_part=original_part,
            filename=filename,
            target_sfreq=target_sfreq,
            tmin=tmin,
            tmax=tmax,
        )
    )

    # Already processed.
    if (
        data_path.exists()
        and labels_path.exists()
        and sfreq_path.exists()
    ):
        return

    edf_path = (
        data_root
        / original_part
        / filename
    )

    if not edf_path.exists():
        raise FileNotFoundError(
            f"Gamma EDF not found: "
            f"{edf_path}"
        )

    print(
        f"[Gamma cache] Processing "
        f"{edf_path}"
    )

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        infer_types=True,
        verbose=False,
    )

    try:
        original_sfreq = float(
            raw.info["sfreq"]
        )

        # Must match the preprocessing
        # used for Gamma features.
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
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )

        data = (
            epochs
            .get_data(copy=True)
            .astype(np.float32)
        )

        labels = (
            epochs.events[:, 2]
            .astype(np.int64)
        )

        del epochs

    finally:
        raw.close()
        del raw

    final_sfreq = (
        original_sfreq
    )

    if (
        target_sfreq is not None
        and not np.isclose(
            original_sfreq,
            target_sfreq,
        )
    ):
        data = mne.filter.resample(
            data,
            up=float(target_sfreq),
            down=original_sfreq,
            axis=-1,
            verbose=False,
        ).astype(np.float32)

        final_sfreq = float(
            target_sfreq
        )

    # MNE epochs include the endpoint.
    target_samples = (
        int(
            round(
                (tmax - tmin)
                * final_sfreq
            )
        )
        + 1
    )

    if (
        data.shape[-1]
        != target_samples
    ):
        data = np.stack(
            [
                resize_trial(
                    trial,
                    target_samples,
                )
                for trial in data
            ]
        ).astype(np.float32)

    # Save uncompressed .npy files so they can
    # later be memory-mapped.
    np.save(
        data_path,
        data,
    )

    np.save(
        labels_path,
        labels,
    )

    np.save(
        sfreq_path,
        np.asarray(
            [final_sfreq],
            dtype=np.float32,
        ),
    )

    print(
        f"[Gamma cache] Saved "
        f"{len(data)} trials | "
        f"shape={data.shape}"
    )

    del data
    del labels


def prepare_gamma_cache(
    metadata: pd.DataFrame,
    data_root: Path,
    cache_dir: Path,
    target_sfreq: float | None,
    tmin: float,
    tmax: float,
) -> None:
    """
    Ensure every EDF needed by this split is
    processed exactly once before training.
    """

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = (
        metadata[
            [
                "original_part",
                "file",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"[Gamma cache] "
        f"{len(files)} unique EDF files required"
    )

    for _, row in files.iterrows():

        preprocess_gamma_file(
            data_root=data_root,
            cache_dir=cache_dir,
            original_part=str(
                row["original_part"]
            ),
            filename=str(
                row["file"]
            ),
            target_sfreq=target_sfreq,
            tmin=tmin,
            tmax=tmax,
        )


# ============================================================
# Dataset
# ============================================================

class GammaCachedDataset(Dataset):

    def __init__(
        self,
        metadata: pd.DataFrame,
        cache_dir: str | os.PathLike,
        target_sfreq: float | None,
        tmin: float,
        tmax: float,
    ):

        self.metadata = (
            metadata
            .reset_index(drop=True)
        )

        self.cache_dir = Path(
            cache_dir
        )

        self.target_sfreq = (
            target_sfreq
        )

        self.tmin = float(tmin)
        self.tmax = float(tmax)

        # Small cache of mmap handles.
        # These do NOT load entire arrays into RAM.
        self._memmap_cache = {}
        self._max_open_files = 4

    def __len__(self):
        return len(
            self.metadata
        )

    def _get_cached_arrays(
        self,
        original_part: str,
        filename: str,
    ):

        key = (
            original_part,
            filename,
        )

        if key in self._memmap_cache:
            return self._memmap_cache[
                key
            ]

        (
            data_path,
            labels_path,
            sfreq_path,
        ) = get_cache_paths(
            cache_dir=self.cache_dir,
            original_part=original_part,
            filename=filename,
            target_sfreq=self.target_sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
        )

        data = np.load(
            data_path,
            mmap_mode="r",
        )

        labels = np.load(
            labels_path,
            mmap_mode="r",
        )

        sfreq = float(
            np.load(
                sfreq_path
            )[0]
        )

        result = (
            data,
            labels,
            sfreq,
        )

        # Keep only a few mmap handles.
        if (
            len(self._memmap_cache)
            >= self._max_open_files
        ):
            first_key = next(
                iter(
                    self._memmap_cache
                )
            )

            del self._memmap_cache[
                first_key
            ]

        self._memmap_cache[
            key
        ] = result

        return result

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        row = self.metadata.iloc[
            index
        ]

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
            self._get_cached_arrays(
                original_part,
                filename,
            )
        )

        if (
            trial_index < 0
            or trial_index >= len(data)
        ):
            raise IndexError(
                f"Invalid trial "
                f"{trial_index} for "
                f"{filename}. "
                f"File has {len(data)} trials."
            )

        actual_condition = int(
            labels[trial_index]
        )

        if (
            actual_condition
            != expected_condition
        ):
            raise ValueError(
                f"Condition mismatch: "
                f"{filename}, "
                f"trial={trial_index}, "
                f"expected="
                f"{expected_condition}, "
                f"actual="
                f"{actual_condition}"
            )

        # Copy only ONE trial from the mmap.
        x_np = np.array(
            data[trial_index],
            dtype=np.float32,
            copy=True,
        )

        x = torch.from_numpy(
            x_np
        ).float()

        item = {
            "x": x,
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
            item[
                "original_index"
            ] = torch.tensor(
                int(
                    row[
                        "original_index"
                    ]
                ),
                dtype=torch.long,
            )

        return item


# ============================================================
# DataLoader construction
# ============================================================

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
    num_workers: int = 2,
    cache_dir: (
        str
        | os.PathLike[str]
        | None
    ) = None,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:

    data_root = Path(
        data_root
    )

    split_path = Path(
        split_path
    )

    if cache_dir is None:
        cache_dir = (
            data_root
            / "_gamma_ae_cache"
        )

    cache_dir = Path(
        cache_dir
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split not found: "
            f"{split_path}"
        )

    split_data = joblib.load(
        split_path
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

    required_columns = {
        "subject",
        "original_part",
        "file",
        "trial",
        "condition",
    }

    missing = (
        required_columns
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            f"Missing metadata columns: "
            f"{sorted(missing)}"
        )

    metadata[
        "subject"
    ] = (
        metadata[
            "subject"
        ].astype(int)
    )

    metadata[
        "original_part"
    ] = (
        metadata[
            "original_part"
        ].astype(str)
    )

    metadata[
        "file"
    ] = (
        metadata[
            "file"
        ].astype(str)
    )

    metadata[
        "trial"
    ] = (
        metadata[
            "trial"
        ].astype(int)
    )

    metadata[
        "condition"
    ] = (
        metadata[
            "condition"
        ].astype(int)
    )

    # --------------------------------------------------------
    # Outer split
    # --------------------------------------------------------

    outer_train, outer_test = (
        get_outer_split_indices(
            split_data,
            split_type,
            target_subject,
        )
    )

    validate_indices(
        outer_train,
        len(metadata),
        "outer train",
    )

    validate_indices(
        outer_test,
        len(metadata),
        "outer test",
    )

    if np.intersect1d(
        outer_train,
        outer_test,
    ).size:
        raise ValueError(
            "Train/test split overlap."
        )

    ae_train, ae_val = (
        stratified_train_val_split(
            metadata,
            outer_train,
            val_fraction,
            validation_seed,
        )
    )

    # --------------------------------------------------------
    # Process each required EDF ONCE.
    # --------------------------------------------------------

    needed_indices = np.unique(
        np.concatenate(
            [
                ae_train,
                ae_val,
                outer_test,
            ]
        )
    )

    needed_metadata = (
        metadata
        .iloc[needed_indices]
    )

    prepare_gamma_cache(
        metadata=needed_metadata,
        data_root=data_root,
        cache_dir=cache_dir,
        target_sfreq=target_sfreq,
        tmin=tmin,
        tmax=tmax,
    )

    # --------------------------------------------------------
    # Create partition metadata
    # --------------------------------------------------------

    train_metadata = (
        metadata
        .iloc[ae_train]
        .copy()
        .reset_index(drop=True)
    )

    val_metadata = (
        metadata
        .iloc[ae_val]
        .copy()
        .reset_index(drop=True)
    )

    test_metadata = (
        metadata
        .iloc[outer_test]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    common_dataset_args = {
        "cache_dir": cache_dir,
        "target_sfreq": target_sfreq,
        "tmin": tmin,
        "tmax": tmax,
    }

    train_dataset = (
        GammaCachedDataset(
            metadata=train_metadata,
            **common_dataset_args,
        )
    )

    val_dataset = (
        GammaCachedDataset(
            metadata=val_metadata,
            **common_dataset_args,
        )
    )

    test_dataset = (
        GammaCachedDataset(
            metadata=test_metadata,
            **common_dataset_args,
        )
    )

    print(
        f"[Gamma train] "
        f"subjects="
        f"{train_metadata['subject'].nunique()}, "
        f"trials={len(train_metadata)}"
    )

    print(
        f"[Gamma val] "
        f"subjects="
        f"{val_metadata['subject'].nunique()}, "
        f"trials={len(val_metadata)}"
    )

    print(
        f"[Gamma test] "
        f"subjects="
        f"{test_metadata['subject'].nunique()}, "
        f"trials={len(test_metadata)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
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

    return (
        train_loader,
        val_loader,
        test_loader,
    )