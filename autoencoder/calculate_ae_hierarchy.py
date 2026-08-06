from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# hierarchy_metric.py is in the parent directory
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hierarchy_metric import HierarchyMetric


def load_vector(path: Path, expected_length: int, name: str) -> np.ndarray:
    values = np.load(path, allow_pickle=True)
    values = np.asarray(values).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"{name} must contain one value per sample. Got shape {values.shape}."
        )

    if len(values) != expected_length:
        raise ValueError(
            f"{name} has {len(values)} rows, but embeddings have "
            f"{expected_length} rows."
        )

    return values


def evaluate_autoencoder_embeddings(
    embedding_dir: Path,
    split: str,
    condition_source: str,
    n_perm: int,
    seed: int,
    space_name: str,
) -> list[dict]:
    embeddings_path = embedding_dir / f"{split}_embeddings.npy"
    subjects_path = embedding_dir / f"{split}_subjects.npy"
    conditions_path = embedding_dir / f"{split}_{condition_source}.npy"

    for path in [embeddings_path, subjects_path, conditions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    X = np.load(embeddings_path, allow_pickle=True)

    if X.ndim < 2:
        raise ValueError(
            f"Embeddings must be at least 2D. Got shape {X.shape}."
        )

    # Ensure shape is (samples, embedding dimensions)
    X = X.reshape(X.shape[0], -1)

    subjects = load_vector(
        subjects_path,
        expected_length=len(X),
        name="subjects",
    )

    conditions = load_vector(
        conditions_path,
        expected_length=len(X),
        name=condition_source,
    )

    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        n_removed = int((~finite_mask).sum())
        print(f"Removing {n_removed} rows containing NaN or infinite values.")

        X = X[finite_mask]
        subjects = subjects[finite_mask]
        conditions = conditions[finite_mask]

    test_labels = pd.DataFrame({
        "subject": subjects.astype(str),
        "condition": conditions.astype(str),
    })

    num_subjects = test_labels["subject"].nunique()
    num_conditions = test_labels["condition"].nunique()

    if num_subjects < 2:
        raise ValueError("At least two subjects are required.")

    if num_conditions < 2:
        raise ValueError(
            f"At least two conditions are required. "
            f"'{condition_source}' contains {num_conditions} unique value(s)."
        )

    metric = HierarchyMetric(X, test_labels)
    result = metric.evaluate(n_perm=n_perm)

    return [{
        "num_subjects": num_subjects,
        "space": space_name,
        "seed": seed,
        "hier_ratio": result["ratio"],
        "inter": result["inter"],
        "intra": result["intra"],
        "hier_pval": result["p_value"],
    }]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate hierarchy metrics from saved autoencoder embeddings "
            "using the same output format as the representation-space script."
        )
    )

    parser.add_argument(
        "--embedding_dir",
        required=True,
        help="Directory containing test_embeddings.npy and metadata arrays.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for hierarchy results. Defaults to "
            "<embedding_dir>/hierarchy."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Saved split prefix. Default: test",
    )
    parser.add_argument(
        "--condition_source",
        choices=["conditions", "labels", "runs", "epochs"],
        default="labels",
        help="Array used as the condition/state label. Default: labels",
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed identifier written to the output table.",
    )
    parser.add_argument(
        "--space_name",
        default="Autoencoder",
        help="Representation-space name written to the output table.",
    )
    parser.add_argument(
        "--feature",
        default="autoencoder",
        help="Feature name used in the summary filename.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help=(
            "Number of state classes used in the summary filename. "
            "If omitted, inferred from the selected condition array."
        ),
    )

    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else embedding_dir / "hierarchy"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate_autoencoder_embeddings(
        embedding_dir=embedding_dir,
        split=args.split,
        condition_source=args.condition_source,
        n_perm=args.n_perm,
        seed=args.seed,
        space_name=args.space_name,
    )

    results = pd.DataFrame(rows)

    # Same raw-results filename and column structure as the other script.
    all_seeds_path = output_dir / "hierarchy_all_seeds.csv"
    results.to_csv(all_seeds_path, index=False)
    print(f"Saved {all_seeds_path}")

    if args.num_classes is None:
        condition_path = (
            embedding_dir
            / f"{args.split}_{args.condition_source}.npy"
        )
        conditions = np.asarray(
            np.load(condition_path, allow_pickle=True)
        ).squeeze()
        num_classes = len(np.unique(conditions))
    else:
        num_classes = args.num_classes

    # Same grouped multi-index summary format.
    for subject_group, group in results.groupby("num_subjects"):
        summary = (
            group.groupby("space")
            .agg({
                "hier_ratio": ["mean", "std"],
                "inter": ["mean", "std"],
                "intra": ["mean", "std"],
                "hier_pval": ["mean", "std"],
            })
        )

        filename = (
            f"hierarchy_results_n{subject_group}_"
            f"{num_classes}classes_{args.feature}.csv"
        )

        summary_path = output_dir / filename
        summary.to_csv(summary_path)
        print(f"Saved {summary_path}")

    print("\nResults")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()