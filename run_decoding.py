from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

from sklearn.svm import LinearSVC

from representation_utils import (
    load_experiment_config,
)


warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
)


# ============================================================
# Classifier
# ============================================================

def run_decoding_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict[str, float]:

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if len(np.unique(y_train)) < 2:
        return {
            "Bal_Acc": np.nan,
            "F1": np.nan,
            "Kappa": np.nan,
        }

    classifier = LinearSVC(
        dual=False,
        max_iter=5000,
        random_state=42,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = classifier.predict(
        X_test
    )

    return {
        "Bal_Acc": balanced_accuracy_score(
            y_test,
            predictions,
        ),

        "F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "Kappa": cohen_kappa_score(
            y_test,
            predictions,
        ),
    }


# ============================================================
# Standard representation decoding
# ============================================================

def evaluate_standard_embedding_file(
    embedding_file: Path,
) -> list[dict]:
    """
    Decode one standard embeddings.pkl bundle.

    trait:
        predict subject

    within_state / between_state:
        predict condition
    """

    bundle = joblib.load(
        embedding_file
    )

    split_type = bundle[
        "split_type"
    ]

    train_metadata = (
        bundle["train_metadata"]
        .copy()
        .reset_index(drop=True)
    )

    test_metadata = (
        bundle["test_metadata"]
        .copy()
        .reset_index(drop=True)
    )

    # ----------------------------------------------
    # Determine decoding target.
    # ----------------------------------------------

    if split_type == "trait":

        y_train = train_metadata[
            "subject"
        ]

        y_test = test_metadata[
            "subject"
        ]

    elif split_type in {
        "within_state",
        "between_state",
    }:

        y_train = train_metadata[
            "condition"
        ]

        y_test = test_metadata[
            "condition"
        ]

    else:
        raise ValueError(
            f"Unknown split type: "
            f"{split_type}"
        )

    rows = []

    for (
        space_name,
        space_data,
    ) in bundle["spaces"].items():

        X_train = space_data[
            "X_train"
        ]

        X_test = space_data[
            "X_test"
        ]

        if len(X_train) != len(y_train):
            raise ValueError(
                f"Train embedding/label "
                f"mismatch in {embedding_file}: "
                f"{len(X_train)} vs "
                f"{len(y_train)}"
            )

        if len(X_test) != len(y_test):
            raise ValueError(
                f"Test embedding/label "
                f"mismatch in {embedding_file}: "
                f"{len(X_test)} vs "
                f"{len(y_test)}"
            )

        metrics = run_decoding_experiment(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        rows.append({
            "num_subjects": (
                bundle["subject_group"]
            ),
            "seed": bundle["seed"],
            "split_type": split_type,
            "subject": bundle[
                "target_subject"
            ],
            "space": space_name,
            "Bal_Acc": metrics[
                "Bal_Acc"
            ],
            "F1": metrics["F1"],
            "Kappa": metrics[
                "Kappa"
            ],
        })

    return rows


# ============================================================
# Autoencoder decoding
# ============================================================

def evaluate_autoencoder_folder(
    run_folder: Path,
    split_type: str,
    num_subjects: str,
    rep: int,
    target_subject: str | None = None,
) -> list[dict]:
    """
    Decode one autoencoder train/test embedding folder.

    trait:
        labels = subjects

    within_state / between_state:
        labels = conditions
    """

    train_embeddings_path = (
        run_folder
        / "train_embeddings.npy"
    )

    test_embeddings_path = (
        run_folder
        / "test_embeddings.npy"
    )

    # ----------------------------------------------
    # Select target labels
    # ----------------------------------------------

    if split_type == "trait":

        train_labels_path = (
            run_folder
            / "train_subjects.npy"
        )

        test_labels_path = (
            run_folder
            / "test_subjects.npy"
        )

    elif split_type in {
        "within_state",
        "between_state",
    }:

        train_labels_path = (
            run_folder
            / "train_conditions.npy"
        )

        test_labels_path = (
            run_folder
            / "test_conditions.npy"
        )

    else:

        raise ValueError(
            f"Unknown split type: "
            f"{split_type}"
        )

    required_files = [
        train_embeddings_path,
        test_embeddings_path,
        train_labels_path,
        test_labels_path,
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing autoencoder files:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    # ----------------------------------------------
    # Load data
    # ----------------------------------------------

    X_train = np.load(
        train_embeddings_path
    )

    X_test = np.load(
        test_embeddings_path
    )

    y_train = np.load(
        train_labels_path
    )

    y_test = np.load(
        test_labels_path
    )

    # ----------------------------------------------
    # Sanity checks
    # ----------------------------------------------

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Train embedding/label mismatch "
            f"in {run_folder}: "
            f"{len(X_train)} embeddings vs "
            f"{len(y_train)} labels."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"Test embedding/label mismatch "
            f"in {run_folder}: "
            f"{len(X_test)} embeddings vs "
            f"{len(y_test)} labels."
        )

    print(
        f"AE | "
        f"split={split_type}, "
        f"n={num_subjects}, "
        f"rep={rep}, "
        f"subject={target_subject}, "
        f"train={len(X_train)}, "
        f"test={len(X_test)}, "
        f"classes="
        f"{len(np.unique(y_train))}"
    )

    metrics = run_decoding_experiment(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    return [
        {
            "num_subjects": num_subjects,

            # Keep column name "seed" so output format
            # remains compatible with standard results.
            "seed": rep,

            "split_type": split_type,
            "subject": target_subject,
            "space": "autoencoder",

            "Bal_Acc": metrics[
                "Bal_Acc"
            ],

            "F1": metrics[
                "F1"
            ],

            "Kappa": metrics[
                "Kappa"
            ],
        }
    ]


# ============================================================
# Find standard embedding files
# ============================================================

def find_standard_embedding_files(
    experiment: dict,
) -> list[Path]:

    embedding_dir = (
        experiment["save_dir"]
        / "embeddings"
    )

    embedding_files = []

    for split_type in [
        "trait",
        "within_state",
        "between_state",
    ]:

        split_dir = (
            embedding_dir
            / split_type
        )

        if not split_dir.exists():

            print(
                f"Missing split directory: "
                f"{split_dir}"
            )

            continue

        if split_type == "trait":

            files = sorted(
                split_dir.glob(
                    "n*_seed*/"
                    "embeddings.pkl"
                )
            )

        else:

            files = sorted(
                split_dir.glob(
                    "n*_seed*/"
                    "subject_*/"
                    "embeddings.pkl"
                )
            )

        print(
            f"{split_type}: "
            f"found {len(files)} "
            f"standard embedding files"
        )

        embedding_files.extend(
            files
        )

    return embedding_files


# ============================================================
# Find autoencoder folders
# ============================================================

def find_autoencoder_runs(
    experiment: dict,
) -> list[dict]:

    autoencoder_dir = (
        experiment["save_dir"]
        / "autoencoder"
    )

    if not autoencoder_dir.exists():
        raise FileNotFoundError(
            f"Autoencoder directory "
            f"not found: "
            f"{autoencoder_dir}"
        )

    runs = []

    for split_type in [
        "trait",
        "within_state",
        "between_state",
    ]:

        split_dir = (
            autoencoder_dir
            / split_type
        )

        if not split_dir.exists():

            print(
                f"Missing AE split directory: "
                f"{split_dir}"
            )

            continue

        # ==============================================
        # Trait
        #
        # trait/
        #     n4_rep0/
        #     n4_rep1/
        # ==============================================

        if split_type == "trait":

            for run_folder in sorted(
                split_dir.iterdir()
            ):

                if not run_folder.is_dir():
                    continue

                match = re.fullmatch(
                    r"n(\d+|all)_rep(\d+)",
                    run_folder.name,
                )

                if match is None:
                    continue

                num_subjects = (
                    match.group(1)
                )

                rep = int(
                    match.group(2)
                )

                runs.append({
                    "folder": (
                        run_folder
                    ),
                    "split_type": (
                        split_type
                    ),
                    "num_subjects": (
                        num_subjects
                    ),
                    "rep": rep,
                    "target_subject": None,
                })

        # ==============================================
        # Within / between
        #
        # within_state/
        #     n4_rep0/
        #         subject_1/
        #         subject_2/
        #
        # between_state/
        #     n4_rep0/
        #         subject_1/
        # ==============================================

        else:

            for seed_folder in sorted(
                split_dir.iterdir()
            ):

                if not seed_folder.is_dir():
                    continue

                match = re.fullmatch(
                    r"n(\d+|all)_rep(\d+)",
                    seed_folder.name,
                )

                if match is None:
                    continue

                num_subjects = (
                    match.group(1)
                )

                rep = int(
                    match.group(2)
                )

                subject_folders = sorted(
                    seed_folder.glob(
                        "subject_*"
                    )
                )

                if num_subjects != "all":
                    expected_subjects = int(num_subjects)

                    valid_subject_folders = [
                        folder
                        for folder in subject_folders
                        if folder.is_dir()
                    ]

                    if len(valid_subject_folders) != expected_subjects:
                        print(
                            f"Skipping incomplete {split_type} run: "
                            f"{seed_folder.name} "
                            f"(found {len(valid_subject_folders)}/"
                            f"{expected_subjects} subjects)"
                        )
                        continue

                    subject_folders = valid_subject_folders

                for subject_folder in (
                    subject_folders
                ):

                    if not (
                        subject_folder
                        .is_dir()
                    ):
                        continue

                    target_subject = (
                        subject_folder.name
                        .replace(
                            "subject_",
                            "",
                        )
                    )

                    runs.append({
                        "folder": (
                            subject_folder
                        ),
                        "split_type": (
                            split_type
                        ),
                        "num_subjects": (
                            num_subjects
                        ),
                        "rep": rep,
                        "target_subject": (
                            target_subject
                        ),
                    })

        print(
            f"{split_type}: "
            f"found "
            f"{sum(r['split_type'] == split_type for r in runs)} "
            f"AE runs"
        )

    return runs


# ============================================================
# Summaries
# ============================================================

def save_results(
    results: pd.DataFrame,
    experiment: dict,
) -> None:

    output_dir = (
        experiment["save_dir"]
        / "geometry"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # All individual results
    # ----------------------------------------------

    all_results_file = (
        output_dir
        / "decoding_all_seeds.csv"
    )

    results.to_csv(
        all_results_file,
        index=False,
    )

    print(
        f"\nSaved "
        f"{all_results_file}"
    )

    # ----------------------------------------------
    # Summary per subject count + split
    # ----------------------------------------------

    for (
        subject_group,
        split_type,
    ), group in results.groupby(
        [
            "num_subjects",
            "split_type",
        ]
    ):

        # Within/between have multiple target
        # subjects per repetition.
        #
        # First average those target subjects
        # within each repetition.
        seed_level = (
            group
            .groupby(
                [
                    "space",
                    "seed",
                ]
            )
            .agg({
                "Bal_Acc": "mean",
                "F1": "mean",
                "Kappa": "mean",
            })
            .reset_index()
        )

        # Then calculate mean/std across reps.
        summary = (
            seed_level
            .groupby(
                "space"
            )
            .agg({
                "Bal_Acc": [
                    "mean",
                    "std",
                ],

                "F1": [
                    "mean",
                    "std",
                ],

                "Kappa": [
                    "mean",
                    "std",
                ],
            })
        )

        filename = (
            f"geometry_results_"
            f"{split_type}_"
            f"n{subject_group}_"
            f"{experiment['num_classes']}"
            f"classes_"
            f"{experiment['feature']}"
            f".csv"
        )

        output_file = (
            output_dir
            / filename
        )

        summary.to_csv(
            output_file
        )

        print(
            f"Saved "
            f"{output_file}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run decoding experiments "
            "from saved train/test "
            "representations."
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

    experiment = (
        load_experiment_config(
            args.config
        )
    )

    feature = str(
        experiment["feature"]
    ).lower()

    print(
        f"Feature: {feature}"
    )

    # ========================================================
    # Autoencoder
    # ========================================================

    if feature == "autoencoder":

        runs = find_autoencoder_runs(
            experiment
        )

        if not runs:
            raise FileNotFoundError(
                "No autoencoder runs found."
            )

        nested_rows = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
        )(
            delayed(
                evaluate_autoencoder_folder
            )(
                run_folder=run[
                    "folder"
                ],
                split_type=run[
                    "split_type"
                ],
                num_subjects=run[
                    "num_subjects"
                ],
                rep=run[
                    "rep"
                ],
                target_subject=run[
                    "target_subject"
                ],
            )
            for run in runs
        )

    # ========================================================
    # Standard representations
    # ========================================================

    else:

        embedding_files = (
            find_standard_embedding_files(
                experiment
            )
        )

        if not embedding_files:
            raise FileNotFoundError(
                "No standard embedding "
                "files found."
            )

        nested_rows = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
        )(
            delayed(
                evaluate_standard_embedding_file
            )(
                embedding_file
            )
            for embedding_file
            in embedding_files
        )

    # ========================================================
    # Flatten
    # ========================================================

    rows = [
        row
        for file_rows in nested_rows
        for row in file_rows
    ]

    if not rows:
        raise ValueError(
            "No decoding results produced."
        )

    results = pd.DataFrame(
        rows
    )

    save_results(
        results=results,
        experiment=experiment,
    )


if __name__ == "__main__":
    main()