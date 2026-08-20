from __future__ import annotations

import argparse
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

from representation_utils import load_experiment_config


warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
)


def run_decoding_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict[str, float]:

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Cannot train a classifier with fewer than 2 classes.
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


def evaluate_embedding_file(
    embedding_file: Path,
) -> list[dict]:
    """
    Run the decoding task corresponding to one saved
    embedding bundle.

    trait:
        subject classification

    within_state:
        condition classification within target subject

    between_state:
        condition classification across subjects
    """

    bundle = joblib.load(
        embedding_file
    )

    split_type = bundle["split_type"]

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
    # Determine decoding labels.
    # ----------------------------------------------

    if split_type == "trait":

        y_train = train_metadata["subject"]
        y_test = test_metadata["subject"]

    elif split_type in {
        "within_state",
        "between_state",
    }:

        y_train = train_metadata["condition"]
        y_test = test_metadata["condition"]

    else:

        raise ValueError(
            f"Unknown split type: {split_type}"
        )

    rows = []

    # ----------------------------------------------
    # Evaluate every representation space.
    # ----------------------------------------------

    for space_name, space_data in (
        bundle["spaces"].items()
    ):

        X_train = space_data["X_train"]
        X_test = space_data["X_test"]

        # Sanity checks
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Train embedding/label mismatch in "
                f"{embedding_file}: "
                f"{len(X_train)} embeddings vs "
                f"{len(y_train)} labels."
            )

        if len(X_test) != len(y_test):
            raise ValueError(
                f"Test embedding/label mismatch in "
                f"{embedding_file}: "
                f"{len(X_test)} embeddings vs "
                f"{len(y_test)} labels."
            )

        metrics = run_decoding_experiment(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        rows.append({
            "num_subjects": bundle["subject_group"],
            "seed": bundle["seed"],
            "split_type": split_type,
            "subject": bundle["target_subject"],
            "space": space_name,
            "Bal_Acc": metrics["Bal_Acc"],
            "F1": metrics["F1"],
            "Kappa": metrics["Kappa"],
        })

    return rows


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run decoding experiments from saved "
            "train/test embeddings."
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

    embedding_dir = (
        experiment["save_dir"]
        / "embeddings"
    )

    output_dir = (
        experiment["save_dir"]
        / "geometry"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Find all valid embedding files.
    #
    # trait/
    #   n4_seed0/
    #       embeddings.pkl
    #
    # within_state/
    #   n4_seed0/
    #       subject_S010/
    #           embeddings.pkl
    #
    # between_state/
    #   n4_seed0/
    #       subject_S010/
    #           embeddings.pkl
    # --------------------------------------------------

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
                    "n*_seed*/embeddings.pkl"
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
            f"found {len(files)} files"
        )

        embedding_files.extend(
            files
        )

    if not embedding_files:

        raise FileNotFoundError(
            f"No embeddings found in "
            f"{embedding_dir}. "
            f"Run create_embeddings.py first."
        )

    # --------------------------------------------------
    # Decode each saved split independently.
    # --------------------------------------------------

    nested_rows = Parallel(
        n_jobs=args.n_jobs,
        backend="loky",
    )(
        delayed(
            evaluate_embedding_file
        )(
            embedding_file
        )
        for embedding_file
        in embedding_files
    )

    rows = [
        row
        for file_rows in nested_rows
        for row in file_rows
    ]

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------
    # Save every individual decoding result.
    # --------------------------------------------------

    all_results_file = (
        output_dir
        / "decoding_all_seeds.csv"
    )

    results.to_csv(
        all_results_file,
        index=False,
    )

    print(
        f"Saved {all_results_file}"
    )

    # --------------------------------------------------
    # Summaries
    #
    # Keep trait / within / between separate.
    # --------------------------------------------------

    for (
        subject_group,
        split_type,
    ), group in results.groupby(
        [
            "num_subjects",
            "split_type",
        ]
    ):

        # For within/between there may be many subjects
        # per seed. First average subjects within each seed.
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

        # Then mean/std across seeds.
        summary = (
            seed_level
            .groupby("space")
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
            f"Saved {output_file}"
        )


if __name__ == "__main__":
    main()