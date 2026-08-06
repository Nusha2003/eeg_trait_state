from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from joblib import Parallel, delayed

from hierarchy_metric import HierarchyMetric
from representation_utils import load_experiment_config


def evaluate_embedding_file(embedding_file: Path, n_perm: int) -> list[dict]:
    bundle = joblib.load(embedding_file)
    metadata = bundle["metadata"]
    test_indices = bundle["trait_split"]["test"]
    test_labels = metadata.iloc[test_indices][["subject", "condition"]].reset_index(drop=True)

    rows = []
    for space_name, space_data in bundle["spaces"].items():
        X = space_data["X"]
        metric = HierarchyMetric(X[test_indices], test_labels)
        result = metric.evaluate(n_perm=n_perm)

        rows.append({
            "num_subjects": bundle["subject_group"],
            "space": space_name,
            "seed": bundle["seed"],
            "hier_ratio": result["ratio"],
            "inter": result["inter"],
            "intra": result["intra"],
            "hier_pval": result["p_value"],
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate hierarchy metrics from saved embeddings.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--n_perm", type=int, default=100)
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    embedding_dir = experiment["save_dir"] / "embeddings"
    output_dir = experiment["save_dir"] / "hierarchy"
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_files = sorted(embedding_dir.glob("embeddings_n*_seed*.pkl"))
    if not embedding_files:
        raise FileNotFoundError(
            f"No embeddings found in {embedding_dir}. Run create_embeddings.py first."
        )

    nested_rows = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(evaluate_embedding_file)(file, args.n_perm)
        for file in embedding_files
    )
    rows = [row for group in nested_rows for row in group]
    results = pd.DataFrame(rows)

    results.to_csv(output_dir / "hierarchy_all_seeds.csv", index=False)

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
            f"{experiment['num_classes']}classes_{experiment['feature']}.csv"
        )
        summary.to_csv(output_dir / filename)
        print(f"Saved {output_dir / filename}")


if __name__ == "__main__":
    main()