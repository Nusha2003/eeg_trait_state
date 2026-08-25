from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hierarchy_metric import HierarchyMetric
from representation_utils import load_experiment_config


# ============================================================
# Autoencoder hierarchy
# ============================================================

def evaluate_autoencoder_folder(
    run_folder: Path,
    num_subjects: str,
    rep: int,
    n_perm: int,
) -> dict:
    """
    Calculate hierarchy using TEST embeddings only from
    one trait autoencoder run.

    Expected files:
        test_embeddings.npy
        test_subjects.npy
        test_conditions.npy
    """

    embeddings_path = (
        run_folder
        / "test_embeddings.npy"
    )

    subjects_path = (
        run_folder
        / "test_subjects.npy"
    )

    conditions_path = (
        run_folder
        / "test_conditions.npy"
    )

    required_files = [
        embeddings_path,
        subjects_path,
        conditions_path,
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

    X_test = np.load(
        embeddings_path
    )

    subjects = np.load(
        subjects_path
    )

    conditions = np.load(
        conditions_path
    )

    if not (
        len(X_test)
        == len(subjects)
        == len(conditions)
    ):
        raise ValueError(
            f"Embedding/metadata mismatch in "
            f"{run_folder}:\n"
            f"embeddings={len(X_test)}\n"
            f"subjects={len(subjects)}\n"
            f"conditions={len(conditions)}"
        )

    labels = pd.DataFrame({
        "subject": subjects,
        "condition": conditions,
    })

    print(
        f"AE | "
        f"n={num_subjects}, "
        f"rep={rep}, "
        f"samples={len(X_test)}, "
        f"subjects={labels['subject'].nunique()}, "
        f"conditions={labels['condition'].nunique()}"
    )

    metric = HierarchyMetric(
        X_test,
        labels,
    )

    result = metric.evaluate(
        n_perm=n_perm,
    )

    return {
        "num_subjects": num_subjects,
        "seed": rep,
        "split_type": "trait",
        "space": "autoencoder",
        "hier_ratio": result["ratio"],
        "inter": result["inter"],
        "intra": result["intra"],
        "hier_pval": result["p_value"],
    }


def evaluate_autoencoder(
    experiment: dict,
    n_perm: int,
) -> pd.DataFrame:
    """
    Traverse:

    save_dir/
        autoencoder/
            trait/
                n4_rep0/
                n4_rep1/
                ...
                nall_rep0/
    """

    trait_dir = (
        experiment["save_dir"]
        / "autoencoder"
        / "trait"
    )

    if not trait_dir.exists():
        raise FileNotFoundError(
            f"Autoencoder trait directory "
            f"not found: {trait_dir}"
        )

    rows = []

    for run_folder in sorted(
        trait_dir.iterdir()
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

        try:
            row = evaluate_autoencoder_folder(
                run_folder=run_folder,
                num_subjects=num_subjects,
                rep=rep,
                n_perm=n_perm,
            )

            rows.append(
                row
            )

        except FileNotFoundError as exc:
            print(
                f"Skipping {run_folder}: "
                f"{exc}"
            )

    if not rows:
        raise ValueError(
            "No valid autoencoder trait "
            "embedding folders were found."
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Existing non-autoencoder hierarchy
# ============================================================

def evaluate_embedding_file(
    embedding_file: Path,
    n_perm: int,
) -> list[dict]:
    """
    Calculate hierarchy on TEST embeddings stored
    in one traditional embeddings.pkl bundle.
    """

    bundle = joblib.load(
        embedding_file
    )

    test_labels = (
        bundle["test_metadata"][
            ["subject", "condition"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    rows = []

    for (
        space_name,
        space_data,
    ) in bundle["spaces"].items():

        X_test = (
            space_data["X_test"]
        )

        if len(X_test) != len(
            test_labels
        ):
            raise ValueError(
                f"Embedding/metadata mismatch "
                f"in {embedding_file}: "
                f"{len(X_test)} vs "
                f"{len(test_labels)}"
            )

        metric = HierarchyMetric(
            X_test,
            test_labels,
        )

        result = metric.evaluate(
            n_perm=n_perm,
        )

        rows.append({
            "num_subjects": (
                bundle["subject_group"]
            ),
            "seed": bundle["seed"],
            "split_type": "trait",
            "space": space_name,
            "hier_ratio": result["ratio"],
            "inter": result["inter"],
            "intra": result["intra"],
            "hier_pval": (
                result["p_value"]
            ),
        })

    return rows


def evaluate_standard_embeddings(
    experiment: dict,
    n_perm: int,
) -> pd.DataFrame:
    """
    Existing representation pipeline:

    save_dir/
        embeddings/
            trait/
                n4_seed0/
                    embeddings.pkl
    """

    trait_dir = (
        experiment["save_dir"]
        / "embeddings"
        / "trait"
    )

    if not trait_dir.exists():
        raise FileNotFoundError(
            f"Trait embedding directory "
            f"not found: {trait_dir}"
        )

    rows = []

    for seed_folder in sorted(
        trait_dir.iterdir()
    ):
        if not seed_folder.is_dir():
            continue

        match = re.fullmatch(
            r"n(\d+|all)_seed(\d+)",
            seed_folder.name,
        )

        if match is None:
            continue

        embedding_file = (
            seed_folder
            / "embeddings.pkl"
        )

        if not embedding_file.exists():
            print(
                f"Missing: {embedding_file}"
            )
            continue

        rows.extend(
            evaluate_embedding_file(
                embedding_file,
                n_perm,
            )
        )

    if not rows:
        raise ValueError(
            "No valid trait embeddings found."
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Save hierarchy results
# ============================================================

def save_results(
    results: pd.DataFrame,
    experiment: dict,
) -> None:

    if experiment["feature"] == "autoencoder":
        output_dir = (
            experiment["save_dir"]
            / "autoencoder"
            / "hierarchy"
            / "trait"
        )
    else:
        output_dir = (
            experiment["save_dir"]
            / "hierarchy"
            / "trait"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results_file = (
        output_dir
        / "hierarchy_all_seeds.csv"
    )

    results.to_csv(
        all_results_file,
        index=False,
    )

    print(
        f"\nSaved {all_results_file}"
    )

    for (
        subject_group,
        group,
    ) in results.groupby(
        "num_subjects"
    ):

        summary = (
            group
            .groupby("space")
            .agg({
                "hier_ratio": [
                    "mean",
                    "std",
                ],
                "inter": [
                    "mean",
                    "std",
                ],
                "intra": [
                    "mean",
                    "std",
                ],
                "hier_pval": [
                    "mean",
                    "std",
                ],
            })
        )

        filename = (
            f"hierarchy_results_"
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


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Calculate trait hierarchy metrics "
            "from held-out test embeddings."
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--n_perm",
        type=int,
        default=100,
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

    # --------------------------------------------------------
    # Autoencoder representation
    # --------------------------------------------------------

    if feature == "autoencoder":

        results = evaluate_autoencoder(
            experiment=experiment,
            n_perm=args.n_perm,
        )

    # --------------------------------------------------------
    # PSD / entropy / complexity / etc.
    # --------------------------------------------------------

    else:

        results = (
            evaluate_standard_embeddings(
                experiment=experiment,
                n_perm=args.n_perm,
            )
        )

    save_results(
        results=results,
        experiment=experiment,
    )


if __name__ == "__main__":
    main()