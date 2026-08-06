from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.svm import LinearSVC

from representation_utils import load_experiment_config

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def run_decoding_experiment(X_train, X_test, y_train, y_test) -> dict[str, float]:
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if len(np.unique(y_train)) < 2:
        return {"Bal_Acc": np.nan, "F1": np.nan, "Kappa": np.nan}

    classifier = LinearSVC(dual=False, max_iter=5000, random_state=42)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    return {
        "Bal_Acc": balanced_accuracy_score(y_test, predictions),
        "F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "Kappa": cohen_kappa_score(y_test, predictions),
    }


def evaluate_embedding_file(embedding_file: Path) -> list[dict]:
    bundle = joblib.load(embedding_file)
    metadata = bundle["metadata"]
    trait_split = bundle["trait_split"]
    state_splits = bundle["state_splits"]

    y_subject = metadata["subject"]
    y_state = metadata["condition"]
    rows = []

    for space_name, space_data in bundle["spaces"].items():
        X = space_data["X"]

        trait_metrics = run_decoding_experiment(
            X[trait_split["train"]],
            X[trait_split["test"]],
            y_subject.iloc[trait_split["train"]],
            y_subject.iloc[trait_split["test"]],
        )

        for split in state_splits:
            within = split["within_state"]
            between = split["between_state"]

            within_metrics = run_decoding_experiment(
                X[within["train"]],
                X[within["test"]],
                y_state.iloc[within["train"]],
                y_state.iloc[within["test"]],
            )
            between_metrics = run_decoding_experiment(
                X[between["train"]],
                X[between["test"]],
                y_state.iloc[between["train"]],
                y_state.iloc[between["test"]],
            )

            rows.append({
                "num_subjects": bundle["subject_group"],
                "space": space_name,
                "seed": bundle["seed"],
                "subject": split["subject"],
                "trait_ident_acc": trait_metrics["Bal_Acc"],
                "within_state_acc": within_metrics["Bal_Acc"],
                "between_acc": between_metrics["Bal_Acc"],
                "trait_ident_f1": trait_metrics["F1"],
                "within_state_f1": within_metrics["F1"],
                "between_f1": between_metrics["F1"],
                "kappa_trait": trait_metrics["Kappa"],
                "kappa_within": within_metrics["Kappa"],
                "kappa_between": between_metrics["Kappa"],
            })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decoding from saved embeddings.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    experiment = load_experiment_config(args.config)
    embedding_dir = experiment["save_dir"] / "embeddings"
    output_dir = experiment["save_dir"] / "geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_files = sorted(embedding_dir.glob("embeddings_n*_seed*.pkl"))
    if not embedding_files:
        raise FileNotFoundError(
            f"No embeddings found in {embedding_dir}. Run create_embeddings.py first."
        )

    nested_rows = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(evaluate_embedding_file)(file)
        for file in embedding_files
    )
    rows = [row for group in nested_rows for row in group]
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "decoding_all_seeds.csv", index=False)

    for subject_group, group in results.groupby("num_subjects"):
        seed_level = (
            group.groupby(["space", "seed"])
            .mean(numeric_only=True)
            .reset_index()
        )
        summary = seed_level.groupby("space").agg(["mean", "std"])
        filename = (
            f"geometry_results_n{subject_group}_"
            f"{experiment['num_classes']}classes_{experiment['feature']}.csv"
        )
        summary.to_csv(output_dir / filename)
        print(f"Saved {output_dir / filename}")


if __name__ == "__main__":
    main()