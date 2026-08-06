# datasets/physionet_raw_ae_dataloader.py

import os
import re
import glob

import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


IGNORED_SUBJECTS = {88, 90, 92, 100}


def run_to_condition(run):
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
        raise ValueError(f"Unknown PhysioNet run: {run}")

    return mapping[run]


def parse_subject_run(path):
    name = os.path.basename(path)
    match = re.match(r"S(\d{3})R(\d{2})\.edf", name)

    if match is None:
        raise ValueError(f"Bad EDF filename: {name}")

    subject = int(match.group(1))
    run = int(match.group(2))

    return subject, run


def get_physionet_files(data_root):
    edf_files = sorted(
        glob.glob(
            os.path.join(data_root, "S*", "S*R*.edf")
        )
    )

    usable = []

    for path in edf_files:
        subject, _ = parse_subject_run(path)

        if subject in IGNORED_SUBJECTS:
            continue

        usable.append(path)

    return usable


class PhysioNetRawAEDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        sfreq=160,
        epoch_len_sec=4.0,
    ):
        self.data_root = data_root
        self.split = split
        self.sfreq = sfreq
        self.epoch_len_samples = int(
            sfreq * epoch_len_sec
        )

        self.samples = []

        split_runs = {
            "train": set(range(1, 7)),
            "val": set(range(7, 11)),
            "test": set(range(11, 15)),
        }

        if split not in split_runs:
            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        allowed_runs = split_runs[split]

        edf_files = get_physionet_files(data_root)

        edf_files = [
            path
            for path in edf_files
            if parse_subject_run(path)[1] in allowed_runs
        ]

        print(
            f"[{split}] Runs: {sorted(allowed_runs)}"
        )
        print(
            f"[{split}] Found {len(edf_files)} EDF files"
        )

        for edf_path in edf_files:
            subject, run = parse_subject_run(edf_path)

            raw = mne.io.read_raw_edf(
                edf_path,
                preload=True,
                verbose=False,
            )

            raw.resample(
                sfreq,
                verbose=False,
            )

            data = raw.get_data().astype(np.float32)

            n_times = data.shape[1]
            n_epochs = (
                n_times // self.epoch_len_samples
            )

            for epoch_idx in range(n_epochs):
                start = (
                    epoch_idx * self.epoch_len_samples
                )
                end = (
                    start + self.epoch_len_samples
                )

                x = data[:, start:end]

                self.samples.append(
                    {
                        "x": x,
                        "subject": subject,
                        "run": run,
                        "condition": run_to_condition(run),
                        "epoch": epoch_idx,
                    }
                )

        print(
            f"[{split}] Created "
            f"{len(self.samples)} epochs"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        x = torch.tensor(
            item["x"],
            dtype=torch.float32,
        )

        return {
            "x": x,
            "labels": x,
            "subject": torch.tensor(
                item["subject"],
                dtype=torch.long,
            ),
            "run": torch.tensor(
                item["run"],
                dtype=torch.long,
            ),
            "condition": torch.tensor(
                item["condition"],
                dtype=torch.long,
            ),
            "epoch": torch.tensor(
                item["epoch"],
                dtype=torch.long,
            ),
        }


def make_physionet_ae_dataloaders(
    data_root,
    batch_size=128,
    sfreq=160,
    epoch_len_sec=4.0,
    num_workers=4,
):
    common_args = {
        "data_root": data_root,
        "sfreq": sfreq,
        "epoch_len_sec": epoch_len_sec,
    }

    train_dataset = PhysioNetRawAEDataset(
        split="train",
        **common_args,
    )

    val_dataset = PhysioNetRawAEDataset(
        split="val",
        **common_args,
    )

    test_dataset = PhysioNetRawAEDataset(
        split="test",
        **common_args,
    )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
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

    return train_loader, val_loader, test_loader