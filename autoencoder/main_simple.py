import os

MNE_DATA_PATH = "/scratch1/amadapur/mne_data"

os.makedirs(MNE_DATA_PATH, exist_ok=True)
os.environ["MNE_DATA"] = MNE_DATA_PATH

import mne

mne.set_config(
    "MNE_DATA",
    MNE_DATA_PATH,
    set_env=True,
)
print("ENV:", os.environ.get("MNE_DATA"))
print("CONFIG:", mne.get_config("MNE_DATA"))

from bci_raw_dataloader import make_bnci_ae_dataloaders as bci_raw_dataloader

from ern_raw_dataloader import make_ern_ae_dataloaders as ern_raw_dataloader

from gamma_raw_dataloader import make_gamma_ae_dataloaders as gamma_raw_dataloader

from motor_raw_dataloader import make_physionet_ae_dataloaders as motor_raw_dataloader

from trainer import PreTrainerEEG
from eegnet_ae import EEGNetAutoEncoder

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml


def reproducible(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    default=(
        "/home1/amadapur/projects/eeg_trait_state_geometry/"
        "autoencoder/config.json"
    ),
)

parser.add_argument(
    "--logdir",
    type=str,
    default=(
        "/home1/amadapur/projects/eeg_trait_state_geometry/"
        "autoencoder/logs"
    ),
)

parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--dataset", required=True)

parser.add_argument(
    "--num_subjects",
    type=str,
    required=True,
)

parser.add_argument(
    "--rep",
    type=int,
    required=True,
)


args = parser.parse_args()

reproducible(args.seed)


# ---------------------------------------------------------
# Load configurations
# ---------------------------------------------------------

if args.num_subjects == "all":
    num_subjects = "all"
else:
    num_subjects = int(args.num_subjects)

rep = args.rep



with open(args.config, "r") as f:
    model_config = json.load(f)

with open(
    "/home1/amadapur/projects/eeg_trait_state_geometry/config.yaml",
    "r",
) as f:
    experiment_config = yaml.safe_load(f)


dataset = args.dataset
data_cfg = experiment_config["data"][dataset]


# ---------------------------------------------------------
# Resolve split and output directories
# ---------------------------------------------------------

if dataset == "motor":
    num_classes = data_cfg["classes"]

    if num_classes == 6:
        split_dir = Path(data_cfg["6split_dir"])
        save_dir = Path(data_cfg["save_dir_6"])

    elif num_classes == 10:
        split_dir = Path(data_cfg["10split_dir"])
        save_dir = Path(data_cfg["save_dir_10"])

    else:
        raise ValueError(
            f"Motor classes must be 6 or 10, got {num_classes}."
        )

else:
    split_dir = Path(data_cfg["split_dir"])
    save_dir = Path(data_cfg["save_dir"])


subjects_list = data_cfg["subjects"]

ae_save_dir = save_dir / "autoencoder"
ae_save_dir.mkdir(parents=True, exist_ok=True)

base_log_dir = Path(args.logdir)
base_log_dir.mkdir(parents=True, exist_ok=True)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



split_types = [
    "trait",
    "within_state",
    "between_state",
]


# ---------------------------------------------------------
# Loop through split files
# ---------------------------------------------------------

if num_subjects == "all":
    split_file = [split_dir / "splits_nall.pkl"]

else:
    split_files = sorted(
        split_dir.glob(
            f"splits_n{num_subjects}_rep*.pkl"
        )
    )
    split_file =(split_dir / f"splits_n{num_subjects}_rep{rep}.pkl")

if not split_file.exists():
    raise FileNotFoundError(
        f"Split file does not exist: {split_file}"
    )
print(
    f"Selected dataset={dataset}, "
    f"n={num_subjects}, rep={rep}"
)


split_data = joblib.load(split_file)

target_subjects = [
    entry["subject"]
    for entry in split_data["splits"]
]

for split_type in split_types:

    # Trait has one global split.
    if split_type == "trait":
        subjects_for_split = [None]

    # Within and between have one split per target subject.
    else:
        subjects_for_split = target_subjects

    for target_subject in subjects_for_split:

        # -----------------------------------------
        # Construct run name
        # -----------------------------------------

        if dataset == "motor":
            dataset_tag = (
                f"{dataset}_{data_cfg['classes']}class"
            )
        else:
            dataset_tag = dataset

        if split_type == "trait":
            run_name = (
                f"{dataset_tag}_trait_"
                f"n{num_subjects}_rep{rep}"
            )
        else:
            run_name = (
                f"{dataset_tag}_{split_type}_"
                f"n{num_subjects}_rep{rep}_"
                f"subject{target_subject}"
            )

        # -----------------------------------------
        # Separate directories for each AE
        # -----------------------------------------

        run_log_dir = base_log_dir / run_name

        run_save_dir = (
            ae_save_dir
            / split_type
            / f"n{num_subjects}_rep{rep}"
        )

        if target_subject is not None:
            run_save_dir = (
                run_save_dir
                / f"subject_{target_subject}"
            )

        run_log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\n" + "=" * 60)
        print(f"Run: {run_name}")
        print(f"Split file: {split_file}")
        print(f"Split type: {split_type}")
        print(f"Target subject: {target_subject}")
        print("=" * 60)

        # -----------------------------------------
        # Construct dataloaders
        # -----------------------------------------

        common_split_args = {
            "split_path": split_file,
            "split_type": split_type,
            "target_subject": target_subject,
            "validation_seed": args.seed,
        }

        if dataset == "motor":
            (
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = motor_raw_dataloader(
                data_root=(
                    "/scratch1/amadapur/data/"
                    "physionet/physionet.org/files/"
                    "eegmmidb/1.0.0"
                ),
                validation_fraction=0.1,
                **common_split_args,
            )

        elif dataset == "bci":
            (
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = bci_raw_dataloader(
                validation_fraction=0.1,
                **common_split_args,
            )

        elif dataset == "ern":
            (
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = ern_raw_dataloader(
                validation_fraction=0.1,
                **common_split_args,
            )

        elif dataset == "gamma":
            (
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = gamma_raw_dataloader(
                data_root=(
                    "/scratch1/amadapur/data/"
                    "gamma/high-gamma-dataset/data/"
                ),
                cache_dir=(
                    "/scratch1/amadapur/data/"
                    "gamma/ae_epoch_cache/"
                ),
                val_fraction=0.1,
                tmin=0.0,
                tmax=4.0,
                target_sfreq=250,
                num_workers=2,
                **common_split_args,
            )

        else:
            raise ValueError(
                f"Unknown dataset: {dataset}"
            )
        split_seed_offsets = {
            "trait": 0,
            "within_state": 10_000,
            "between_state": 20_000,
        }
        run_seed = (
            args.seed
            + rep
            + split_seed_offsets[split_type]
        )

        if target_subject is not None:
            if isinstance(target_subject, str) and target_subject.startswith("S"):
                subject_num = int(target_subject[1:])
            else:
                subject_num = int(target_subject)

            run_seed += subject_num

        reproducible(run_seed)
        #infer number of channels from actual dataset

        SAMPLING_RATES = {
            "motor": 160,
            "gamma": 250,
            "bci": 512,
            "ern": 1024,
        }
        sample = train_dataloader.dataset[0]

        model_params = model_config["model_params"].copy()

        model_params["n_channels"] = sample["x"].shape[0]
        model_params["sampling_rate"] = SAMPLING_RATES[dataset]

        n_timepoints = sample["x"].shape[1]
        model_params["n_seconds_input"] = (
            n_timepoints / model_params["sampling_rate"]
        )

        print(
            f"{dataset}: "
            f"channels={model_params['n_channels']}, "
            f"sampling_rate={model_params['sampling_rate']}, "
            f"n_timepoints={n_timepoints}, "
            f"duration={model_params['n_seconds_input']:.3f}s"
        )

        model = EEGNetAutoEncoder(
            **model_params
        ).to(device)


        pretrainer = PreTrainerEEG(
            model=model,
            logdir=str(run_log_dir),
            **model_config["trainer_args"],
        )

        pretrainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            **model_config["train_fn_args"],
        )

        # Only save embeddings for samples held out
        # from this exact autoencoder.
        pretrainer.save_embeddings(
            train_dataloader,
            save_path=str(run_save_dir),
            split="train",
        )

        pretrainer.save_embeddings(
            val_dataloader,
            save_path=str(run_save_dir),
            split="val",
        )

        pretrainer.save_embeddings(
            test_dataloader,
            save_path=str(run_save_dir),
            split="test",
        )