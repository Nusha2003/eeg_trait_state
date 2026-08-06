# tune_joint_pca_lda.py

import os
import re
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd

from kneed import KneeLocator
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true"
)

warnings.filterwarnings(
    "ignore",
    category=UndefinedMetricWarning
)


def parse_subjects(fname):
    if "all" in fname:
        return "all"
    m = re.search(r"n(\d+)", fname)
    return m.group(1) if m else None


def run_cv_for_dims(
    X_train_outer,
    y_train_outer,
    pca_range,
    lda_range,
    nfold=5,
    n_repeat=100,
    random_state=42,
):
    """
    repeated K-fold CV over PCA and LDA dimensions.

    returns:
        results_df with columns:
        pca_dim, lda_dim, val_bal_acc_mean, val_bal_acc_std
    """

    y_train_outer = np.asarray(y_train_outer)
    rows = []

    for pca_dim in pca_range:
        print(f"\nPCA dim = {pca_dim}")

        for lda_dim in lda_range:
            fold_scores = []

            for rep in range(n_repeat):
                kf = KFold(
                    n_splits=nfold,
                    shuffle=True,
                    random_state=random_state + rep,
                )

                for inner_train_idx, val_idx in kf.split(X_train_outer):
                    X_inner_train = X_train_outer[inner_train_idx]
                    X_val = X_train_outer[val_idx]

                    y_inner_train = y_train_outer[inner_train_idx]
                    y_val = y_train_outer[val_idx]

                    n_samples = X_inner_train.shape[0]
                    n_features = X_inner_train.shape[1]
                    n_classes = len(np.unique(y_inner_train))

                    actual_pca_dim = min(pca_dim, n_samples - 1, n_features)
                    actual_lda_dim = min(lda_dim, actual_pca_dim, n_classes - 1)

                    if actual_lda_dim < 1:
                        continue

                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=actual_pca_dim)),
                        ("lda", LinearDiscriminantAnalysis(n_components=actual_lda_dim)),
                    ])

                    pipe.fit(X_inner_train, y_inner_train)

                    X_inner_train_t = pipe.transform(X_inner_train)
                    X_val_t = pipe.transform(X_val)

                    clf = LinearSVC(
                        dual=False,
                        max_iter=5000,
                        random_state=random_state,
                    )

                    clf.fit(X_inner_train_t, y_inner_train)
                    y_pred = clf.predict(X_val_t)

                    score = balanced_accuracy_score(y_val, y_pred)
                    fold_scores.append(score)

            if len(fold_scores) > 0:
                rows.append({
                    "pca_dim": pca_dim,
                    "lda_dim": lda_dim,
                    "val_bal_acc_mean": np.mean(fold_scores),
                    "val_bal_acc_std": np.std(fold_scores),
                })

                print(
                    f"  LDA dim = {lda_dim:02d}, "
                    f"val bal acc = {np.mean(fold_scores):.4f}"
                )

    return pd.DataFrame(rows)


def select_pca_lda_dims(results_df):
    """
    1. for each PCA dim, compute AUC over LDA validation accuracy.
    2. Choose PCA knee on AUC curve.
    3. At selected PCA, choose LDA knee on validation accuracy curve.
    """

    auc_rows = []

    for pca_dim, df in results_df.groupby("pca_dim"):
        df = df.sort_values("lda_dim")

        auc = np.trapz(
            df["val_bal_acc_mean"].values,
            df["lda_dim"].values,
        )

        auc_rows.append({
            "pca_dim": pca_dim,
            "accuracy_auc": auc,
        })

    auc_df = pd.DataFrame(auc_rows).sort_values("pca_dim")

    pca_kneedle = KneeLocator(
        auc_df["pca_dim"].values,
        auc_df["accuracy_auc"].values,
        S=1.0,
        curve="concave",
        direction="increasing",
    )

    best_pca = pca_kneedle.knee

    if best_pca is None:
        best_pca = auc_df.loc[auc_df["accuracy_auc"].idxmax(), "pca_dim"]

    best_pca = int(best_pca)

    lda_curve = (
        results_df[results_df["pca_dim"] == best_pca]
        .sort_values("lda_dim")
    )

    lda_kneedle = KneeLocator(
        lda_curve["lda_dim"].values,
        lda_curve["val_bal_acc_mean"].values,
        S=1.0,
        curve="concave",
        direction="increasing",
    )

    best_lda = lda_kneedle.knee

    if best_lda is None:
        best_lda = lda_curve.loc[
            lda_curve["val_bal_acc_mean"].idxmax(),
            "lda_dim"
        ]

    best_lda = int(best_lda)

    return best_pca, best_lda, auc_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--nfold", type=int, default=5)
    parser.add_argument("--n_repeat", type=int, default=100)

    parser.add_argument("--pca_min", type=int, default=5)
    parser.add_argument("--pca_max", type=int, default=30)
    parser.add_argument("--pca_step", type=int, default=5)

    parser.add_argument("--lda_min", type=int, default=1)
    parser.add_argument("--lda_max", type=int, default=20)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    data_cfg = config["data"][dataset]
    feature = config["feature"]
    num_classes = data_cfg["classes"]

    if dataset == "motor":
        if num_classes == 6:
            split_dir = data_cfg["6split_dir"]
            save_dir = data_cfg["save_dir_6"]
        else:
            split_dir = data_cfg["10split_dir"]
            save_dir = data_cfg["save_dir_10"]
    else:
        split_dir = data_cfg["split_dir"]
        save_dir = data_cfg["save_dir"]

    save_dir = os.path.join(save_dir, feature)
    tune_dir = os.path.join(save_dir, "dimension_tuning")
    os.makedirs(tune_dir, exist_ok=True)

    X = pd.read_csv(data_cfg[feature]).values

    split_files = [
        f for f in os.listdir(split_dir)
        if f.startswith("splits") and f.endswith(".pkl")
    ]

    groups = {}
    for f in split_files:
        subj = parse_subjects(f)
        if subj is not None:
            groups.setdefault(subj, []).append(f)

    pca_range = np.arange(args.pca_min, args.pca_max + 1, args.pca_step)
    lda_range = np.arange(args.lda_min, args.lda_max + 1, 1)

    summary_rows = []

    for subj_key, files in groups.items():
        print(f"Tuning n_subjects = {subj_key}")

        for s_file in sorted(files):
            print(f"\nSplit file: {s_file}")

            data = joblib.load(os.path.join(split_dir, s_file))

            metadata = data["metadata"]
            metadata.columns = metadata.columns.str.lower().str.strip()

            trait_split = data["trait_split"]

            seed_id = (
                int(s_file.split("_rep")[-1].replace(".pkl", ""))
                if "_rep" in s_file
                else 0
            )

            y_joint = (
                metadata.subject.astype(str)
                + "_"
                + metadata.condition.astype(str)
            )

            outer_train_idx = trait_split["train"]

            X_train_outer = X[outer_train_idx]
            y_train_outer = y_joint.iloc[outer_train_idx].values

            results_df = run_cv_for_dims(
                X_train_outer=X_train_outer,
                y_train_outer=y_train_outer,
                pca_range=pca_range,
                lda_range=lda_range,
                nfold=args.nfold,
                n_repeat=args.n_repeat,
            )

            best_pca, best_lda, auc_df = select_pca_lda_dims(results_df)

            out_prefix = (
                f"joint_pca_lda_tuning_"
                f"n{subj_key}_seed{seed_id}_"
                f"{num_classes}classes_{feature}"
            )

            results_df.to_csv(
                os.path.join(tune_dir, out_prefix + "_heatmap.csv"),
                index=False,
            )

            auc_df.to_csv(
                os.path.join(tune_dir, out_prefix + "_pca_auc.csv"),
                index=False,
            )

            summary_rows.append({
                "num_subjects": subj_key,
                "seed": seed_id,
                "best_pca_dim": best_pca,
                "best_lda_dim": best_lda,
                "split_file": s_file,
            })

            print(f"\nSelected PCA dim: {best_pca}")
            print(f"Selected LDA dim: {best_lda}")

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        tune_dir,
        f"joint_pca_lda_selected_dims_{num_classes}classes_{feature}.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print("Saved selected dimensions to:")
    print(summary_path)


if __name__ == "__main__":
    main()