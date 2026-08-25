


from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DEFAULT_SPACES = ["Raw", "Trait_LDA", "State_LDA", "Joint_LDA"]

def load_experiment_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the global dataset/feature selection and resolve dataset paths."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    feature = config["feature"]
    data_cfg = config["data"][dataset]
    num_classes = int(data_cfg["classes"])

    if dataset == "motor":
        if num_classes not in {6, 10}:
            raise ValueError("Motor classes must be 6 or 10.")

        split_dir = Path(data_cfg[f"{num_classes}split_dir"])
        base_save_dir = Path(data_cfg[f"save_dir_{num_classes}"])

    else:
        split_dir = Path(data_cfg["split_dir"])
        base_save_dir = Path(data_cfg["save_dir"])

    # Autoencoder embeddings are generated outputs,
    # not an input feature file in config.yaml.
    if feature == "autoencoder":
        feature_path = None
        save_dir = base_save_dir
    else:
        feature_path = Path(data_cfg[feature])
        save_dir = base_save_dir / feature

    spaces = config.get(
        "experiment",
        {}
    ).get(
        "spaces",
        DEFAULT_SPACES,
    )

    return {
        "dataset": dataset,
        "feature": feature,
        "num_classes": num_classes,
        "split_dir": split_dir,
        "save_dir": save_dir,
        "feature_path": feature_path,
        "spaces": spaces,
        "raw_config": config,
        "data_config": data_cfg,
    }

def parse_subject_group(filename: str) -> str | None:
    """Extract n-subject group from names such as splits_n4_rep0.pkl or splits_all.pkl."""
    if "all" in filename.lower():
        return "all"
    match = re.search(r"n(\d+)", filename)
    return match.group(1) if match else None


def get_seed_id(filename: str) -> int:
    match = re.search(r"_rep(\d+)", filename)
    return int(match.group(1)) if match else 0


def group_split_files(split_dir: Path) -> dict[str, list[str]]:
    files = [
        name
        for name in os.listdir(split_dir)
        if name.startswith("splits") and name.endswith(".pkl")
    ]

    groups: dict[str, list[str]] = {}
    for filename in files:
        subject_group = parse_subject_group(filename)
        if subject_group is not None:
            groups.setdefault(subject_group, []).append(filename)

    return {key: sorted(value) for key, value in groups.items()}


def get_space_labels(metadata: pd.DataFrame, space_name: str) -> pd.Series | None:
    if space_name == "Raw":
        return None
    if space_name == "Trait_LDA":
        return metadata["subject"]
    if space_name == "State_LDA":
        return metadata["condition"]
    if space_name == "Joint_LDA":
        return metadata["subject"].astype(str) + "_" + metadata["condition"].astype(str)
    raise ValueError(f"Unknown space: {space_name}")


def fit_space(
    X: np.ndarray,
    y: pd.Series | np.ndarray | None,
    train_idx: np.ndarray | list[int],
    space_type: str,
    pca_dim: int = 30,
    lda_dim: int = 10,
) -> Pipeline:
    """Fit scaler + PCA + optional LDA using only the supplied training indices."""
    train_idx = np.asarray(train_idx, dtype=int)
    X_train = X[train_idx]

    n_samples, n_features = X_train.shape
    n_pca = min(pca_dim, n_samples - 1, n_features)
    if n_pca < 1:
        raise ValueError("At least two training samples and one feature are required.")

    steps: list[tuple[str, Any]] = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=42)),
    ]

    y_train = None
    if space_type != "Raw":
        if y is None:
            raise ValueError(f"Labels are required for {space_type}.")
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else np.asarray(y)[train_idx]
        n_classes = len(np.unique(y_train))
        n_lda = min(lda_dim, n_classes - 1, n_pca)
        if n_lda >= 1:
            steps.append(("lda", LinearDiscriminantAnalysis(n_components=n_lda)))

    pipeline = Pipeline(steps)
    if space_type == "Raw" or "lda" not in pipeline.named_steps:
        pipeline.fit(X_train)
    else:
        pipeline.fit(X_train, y_train)

    return pipeline


def tune_joint_dims_for_split(
    X: np.ndarray,
    y_joint: pd.Series,
    train_idx: np.ndarray | list[int],
    pca_range: range,
    lda_range: range,
    nfold: int = 3,
    n_repeat: int = 1,
) -> pd.DataFrame:
    """Tune Joint_LDA dimensions using CV inside the outer training split only."""
    train_idx = np.asarray(train_idx, dtype=int)
    X_outer = X[train_idx]
    y_outer = y_joint.iloc[train_idx].to_numpy()
    rows: list[dict[str, float | int]] = []

    class_counts = pd.Series(y_outer).value_counts()
    effective_folds = min(nfold, int(class_counts.min()))
    if effective_folds < 2:
        raise ValueError("Not enough samples per joint class for cross-validation.")

    for pca_dim in pca_range:
        for lda_dim in lda_range:
            scores: list[float] = []

            for rep in range(n_repeat):
                cv = StratifiedKFold(
                    n_splits=effective_folds,
                    shuffle=True,
                    random_state=42 + rep,
                )

                for inner_train_idx, val_idx in cv.split(X_outer, y_outer):
                    X_inner = X_outer[inner_train_idx]
                    X_val = X_outer[val_idx]
                    y_inner = y_outer[inner_train_idx]
                    y_val = y_outer[val_idx]

                    n_samples, n_features = X_inner.shape
                    n_classes = len(np.unique(y_inner))
                    actual_pca = min(pca_dim, n_samples - 1, n_features)
                    actual_lda = min(lda_dim, actual_pca, n_classes - 1)

                    if actual_pca < 1 or actual_lda < 1:
                        continue

                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=actual_pca, random_state=42)),
                        ("lda", LinearDiscriminantAnalysis(n_components=actual_lda)),
                    ])
                    pipeline.fit(X_inner, y_inner)

                    X_inner_t = pipeline.transform(X_inner)
                    X_val_t = pipeline.transform(X_val)

                    classifier = LinearSVC(dual=False, max_iter=5000, random_state=42)
                    classifier.fit(X_inner_t, y_inner)
                    predictions = classifier.predict(X_val_t)
                    scores.append(balanced_accuracy_score(y_val, predictions))

            if scores:
                rows.append({
                    "pca_dim": pca_dim,
                    "lda_dim": lda_dim,
                    "val_bal_acc": float(np.mean(scores)),
                })

    return pd.DataFrame(rows)


def select_joint_dims(results_df: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    if results_df.empty:
        raise ValueError("Joint dimension tuning returned no valid results.")

    auc_rows = []
    for pca_dim, group in results_df.groupby("pca_dim"):
        group = group.sort_values("lda_dim")
        if len(group) == 1:
            auc = float(group["val_bal_acc"].iloc[0])
        else:
            auc = float(np.trapezoid(group["val_bal_acc"], group["lda_dim"]))
        auc_rows.append({"pca_dim": int(pca_dim), "accuracy_auc": auc})

    auc_df = pd.DataFrame(auc_rows).sort_values("pca_dim")
    pca_knee = KneeLocator(
        auc_df["pca_dim"].to_numpy(),
        auc_df["accuracy_auc"].to_numpy(),
        curve="concave",
        direction="increasing",
    ).knee

    if pca_knee is None:
        pca_knee = auc_df.loc[auc_df["accuracy_auc"].idxmax(), "pca_dim"]
    pca_knee = int(pca_knee)

    lda_curve = (
        results_df[results_df["pca_dim"] == pca_knee]
        .groupby("lda_dim", as_index=False)["val_bal_acc"]
        .mean()
        .sort_values("lda_dim")
    )

    lda_knee = KneeLocator(
        lda_curve["lda_dim"].to_numpy(),
        lda_curve["val_bal_acc"].to_numpy(),
        curve="concave",
        direction="increasing",
    ).knee

    if lda_knee is None:
        lda_knee = lda_curve.loc[lda_curve["val_bal_acc"].idxmax(), "lda_dim"]

    return pca_knee, int(lda_knee), auc_df


def tune_joint_dims_for_group(
    files: list[str],
    split_dir: Path,
    X: np.ndarray,
    subject_group: str,
    tune_dir: Path,
    dataset: str,
    feature: str,
    num_classes: int,
) -> tuple[int, int]:
    print(f"Tuning Joint_LDA dimensions for n_subjects={subject_group}")

    pca_range = range(5, 31, 10)
    lda_range = range(1, 21)
    all_results = []

    for split_file in files:
        data = joblib.load(split_dir / split_file)

        metadata = data["metadata"].copy()
        metadata.columns = metadata.columns.str.lower().str.strip()

        if "original_index" not in metadata.columns:
            raise ValueError(f"{split_file} is missing original_index.")

        original_indices = metadata["original_index"].to_numpy(dtype=int)

        if len(original_indices) == 0:
            raise ValueError(f"{split_file} contains no metadata rows.")

        if original_indices.min() < 0 or original_indices.max() >= len(X):
            raise IndexError(
                f"{split_file} has original_index values outside "
                f"0 to {len(X) - 1}."
            )

        # Subset the full feature matrix to the rows in this split.
        X_group = X[original_indices]

        if len(X_group) != len(metadata):
            raise ValueError(
                f"Feature rows ({len(X_group)}) do not match metadata rows "
                f"({len(metadata)}) for {split_file}."
            )

        trait_split = data["trait_split"]

        y_joint = (
            metadata["subject"].astype(str)
            + "_"
            + metadata["condition"].astype(str)
        )

        results = tune_joint_dims_for_split(
            X=X_group,
            y_joint=y_joint,
            train_idx=trait_split["train"],
            pca_range=pca_range,
            lda_range=lda_range,
        )

        results["seed"] = get_seed_id(split_file)
        results["num_subjects"] = subject_group
        all_results.append(results)

    results_df = pd.concat(all_results, ignore_index=True)

    avg_results = (
        results_df
        .groupby(["pca_dim", "lda_dim"], as_index=False)["val_bal_acc"]
        .mean()
    )

    best_pca, best_lda, auc_df = select_joint_dims(avg_results)

    tune_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{dataset}_n{subject_group}_{num_classes}classes_{feature}"

    results_df.to_csv(
        tune_dir / f"joint_tuning_allseeds_{stem}.csv",
        index=False,
    )
    avg_results.to_csv(
        tune_dir / f"joint_tuning_avg_{stem}.csv",
        index=False,
    )
    auc_df.to_csv(
        tune_dir / f"joint_tuning_auc_{stem}.csv",
        index=False,
    )

    print(f"Selected Joint_LDA dimensions: PCA={best_pca}, LDA={best_lda}")
    return best_pca, best_lda