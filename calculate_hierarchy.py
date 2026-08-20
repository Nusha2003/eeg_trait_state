from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from joblib import Parallel, delayed
import numpy as np

from hierarchy_metric import HierarchyMetric
from representation_utils import load_experiment_config

"""
def evaluate_embedding_file(
    embedding_file: Path,
    n_perm: int,
) -> list[dict]:

    Calculate hierarchy metrics on the TEST embeddings
    stored in one embeddings.pkl file.


    bundle = joblib.load(embedding_file)

    # These are already the metadata rows corresponding
    # to X_test.
    test_labels = (
        bundle["test_metadata"][
            ["subject", "condition"]
        ]
        .reset_index(drop=True)
    )

    rows = []

    for space_name, space_data in bundle["spaces"].items():

        # These are already the TEST embeddings.
        X_test = space_data["X_test"]

        if len(X_test) != len(test_labels):
            raise ValueError(
                f"Embedding/metadata mismatch in {embedding_file}: "
                f"{len(X_test)} embeddings vs "
                f"{len(test_labels)} metadata rows."
            )

        metric = HierarchyMetric(
            X_test,
            test_labels,
        )

        result = metric.evaluate(
            n_perm=n_perm,
        )

        rows.append({
            "num_subjects": bundle["subject_group"],
            "seed": bundle["seed"],
            "split_type": bundle["split_type"],
            "target_subject": bundle["target_subject"],
            "space": space_name,
            "hier_ratio": result["ratio"],
            "inter": result["inter"],
            "intra": result["intra"],
            "hier_pval": result["p_value"],
        })

    return rows

"""

def evaluate_seed_folder(
    embedding_files: list[Path],
    n_perm: int,
) -> list[dict]:

    bundles = [
        joblib.load(file)
        for file in embedding_files
    ]

    first_bundle = bundles[0]

    rows = []

    space_names = first_bundle["spaces"].keys()

    for space_name in space_names:

        X_parts = []
        metadata_parts = []

        for bundle in bundles:

            X_test = bundle["spaces"][space_name]["X_test"]
            test_metadata = (
                bundle["test_metadata"][
                    ["subject", "condition"]
                ]
                .copy()
                .reset_index(drop=True)
            )

            if len(X_test) != len(test_metadata):
                raise ValueError(
                    "Embedding/metadata mismatch: "
                    f"{len(X_test)} embeddings vs "
                    f"{len(test_metadata)} metadata rows."
                )

            X_parts.append(X_test)
            metadata_parts.append(test_metadata)

        X = np.concatenate(
            X_parts,
            axis=0,
        )

        labels = pd.concat(
            metadata_parts,
            ignore_index=True,
        )

        print(
            f"n={first_bundle['subject_group']}, "
            f"seed={first_bundle['seed']}, "
            f"space={space_name}, "
            f"samples={len(X)}, "
            f"subjects={labels['subject'].nunique()}, "
            f"conditions={labels['condition'].nunique()}"
        )

        metric = HierarchyMetric(
            X,
            labels,
        )

        result = metric.evaluate(
            n_perm=n_perm,
        )

        rows.append({
            "num_subjects": first_bundle["subject_group"],
            "seed": first_bundle["seed"],
            "split_type": first_bundle["split_type"],
            "space": space_name,
            "hier_ratio": result["ratio"],
            "inter": result["inter"],
            "intra": result["intra"],
            "hier_pval": result["p_value"],
        })

    return rows


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Calculate hierarchy metrics "
            "from saved test embeddings."
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

    parser.add_argument(
        "--n_perm",
        type=int,
        default=100,
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
        / "hierarchy"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # embeddings/
    #
    #   trait/
    #       n4_seed0/
    #           embeddings.pkl
    #
    #   within_state/
    #       n4_seed0/
    #           subject_S001/
    #               embeddings.pkl
    #
    #   between_state/
    #       n4_seed0/
    #           subject_S001/
    #               embeddings.pkl
    # --------------------------------------------------

    for split_dir in sorted(
        embedding_dir.iterdir()
    ):

        if not split_dir.is_dir():
            continue

        split_type = split_dir.name

        if split_type not in {
            "trait",
            "within_state",
            "between_state",
        }:
            continue

        print(
            f"\nProcessing split: {split_type}"
        )

        split_output_dir = (
            output_dir
            / split_type
        )

        split_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_rows = []

        # ----------------------------------------------
        # Example:
        #
        # n4_seed0
        # n4_seed1
        # n10_seed0
        # n10_seed1
        # ----------------------------------------------

        for seed_folder in sorted(
            split_dir.iterdir()
        ):

            if not seed_folder.is_dir():
                continue

            match = re.fullmatch(
                r"n(\d+|all)_seed(\d+)",
                seed_folder.name,
            )

            if match is None:
                continue

            num_subjects = match.group(1)
            seed = int(match.group(2))

            print(
                f"  n={num_subjects}, "
                f"seed={seed}"
            )

            # ------------------------------------------
            # Trait has:
            #
            # n4_seed0/
            #     embeddings.pkl
            #
            # because target_subject=None
            # ------------------------------------------

            if split_type == "trait":

                embedding_file = (
                    seed_folder
                    / "embeddings.pkl"
                )

                if not embedding_file.exists():
                    print(
                        f"    Missing: "
                        f"{embedding_file}"
                    )
                    continue

                embedding_files = [
                    embedding_file
                ]

            # ------------------------------------------
            # within_state / between_state have:
            #
            # n4_seed0/
            #     subject_S001/
            #         embeddings.pkl
            #     subject_S002/
            #         embeddings.pkl
            # ------------------------------------------

            else:

                embedding_files = sorted(
                    seed_folder.glob(
                        "subject_*/embeddings.pkl"
                    )
                )

                if not embedding_files:
                    print(
                        f"    No subject embeddings "
                        f"found in {seed_folder}"
                    )
                    continue

            print(
                f"    Found "
                f"{len(embedding_files)} "
                f"embedding file(s)"
            )

            # Evaluate files in this seed in parallel.
            rows = evaluate_seed_folder(
                embedding_files,
                args.n_perm,
            )

            all_rows.extend(rows)

        if not all_rows:

            print(
                f"No valid embeddings found "
                f"for {split_type}."
            )

            continue

        results = pd.DataFrame(
            all_rows
        )

        # ----------------------------------------------
        # Save every individual result.
        # ----------------------------------------------

        all_results_file = (
            split_output_dir
            / "hierarchy_all_seeds.csv"
        )

        results.to_csv(
            all_results_file,
            index=False,
        )

        print(
            f"Saved {all_results_file}"
        )

        # ----------------------------------------------
        # Aggregate by subject-group size.
        #
        # n4 results together
        # n10 results together
        # n20 results together
        # etc.
        # ----------------------------------------------

        for subject_group, group in (
            results.groupby(
                "num_subjects"
            )
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
                split_output_dir
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