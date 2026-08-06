# showing that EEG has a natural hierarchy

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


DATASET_ALL_MAP = {
    "motor": 105,
    "gamma": 14,
    "bci": 14,
    "lee": 54,
    "lemon": 156,
}

KNOWN_FEATURES = {"psd", "entropy", "complexity"}


def parse_num_classes(path):
    fname = os.path.basename(path)
    m = re.search(r"_(\d+)classes", fname)
    return int(m.group(1)) if m else None


def infer_feature(path):
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p.lower() in KNOWN_FEATURES:
            return p.lower()
    return "unknown_feature"


def flatten_multilevel_columns(df):
    flat_cols = []

    for i, (col0, col1) in enumerate(df.columns):
        col0 = str(col0).strip()
        col1 = str(col1).strip()

        if i == 0:
            flat_cols.append("num_subjects")
        elif i == 1:
            flat_cols.append("space")
        elif "Unnamed" in col1 or col1 == "":
            flat_cols.append(col0)
        else:
            flat_cols.append(f"{col0}_{col1}")

    df.columns = flat_cols
    return df


def load_hierarchy_csv(csv_path, dataset):
    df = pd.read_csv(csv_path, header=[0, 1], skiprows=[2])
    df = flatten_multilevel_columns(df)

    df = df.rename(columns={
        "hier_ratio_mean": "mean_hierarchy",
        "hier_ratio_std": "std_hierarchy",
        "inter_mean": "mean_inter",
        "inter_std": "std_inter",
        "intra_mean": "mean_intra",
        "intra_std": "std_intra",
        "hier_pval_mean": "mean_pval",
        "hier_pval_std": "std_pval",
    })

    required = {
        "num_subjects",
        "space",
        "mean_hierarchy",
        "mean_inter",
        "mean_intra"
    }

    missing = required - set(df.columns)
    if missing:
        print(f"Skipping {csv_path}; missing {missing}")
        return None

    df["space"] = df["space"].replace({"Raw": "Unsupervised"})
    df = df[df["space"] == "Unsupervised"].copy()

    if df.empty:
        return None

    df["dataset"] = dataset
    df["feature"] = infer_feature(csv_path)
    df["num_classes"] = parse_num_classes(csv_path)
    df["source_file"] = csv_path

    def parse_subjects(x):
        x = str(x).strip().lower()
        if x in {"all", "nall"}:
            return "all"
        return int(float(x))

    df["n_subjects"] = df["num_subjects"].apply(parse_subjects)

    for col in [
        "mean_hierarchy", "std_hierarchy",
        "mean_inter", "std_inter",
        "mean_intra", "std_intra",
        "mean_pval", "std_pval"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    return df


def collect_unsupervised_results(results_dir, dataset, num_classes=None):
    dataset_dir = os.path.join(results_dir, dataset)
    files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    hierarchy_files = [f for f in files if "hierarchy" in f.lower()]

    if not hierarchy_files:
        raise FileNotFoundError(f"No hierarchy CSVs found under {dataset_dir}")

    dfs = []

    for f in hierarchy_files:
        df = load_hierarchy_csv(f, dataset)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No Unsupervised hierarchy rows found for dataset={dataset}")

    df = pd.concat(dfs, ignore_index=True)

    if num_classes is not None:
        df = df[df["num_classes"] == num_classes].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for dataset={dataset}, num_classes={num_classes}"
        )

    def plot_n(row):
        if row["n_subjects"] == "all":
            return DATASET_ALL_MAP.get(dataset, np.nan)
        return row["n_subjects"]

    df["n_subjects_plot"] = df.apply(plot_n, axis=1)
    df["n_subjects_plot"] = pd.to_numeric(df["n_subjects_plot"], errors="coerce")
    df["n_subjects_label"] = df["n_subjects_plot"].astype(int).astype(str)

    df = df.dropna(subset=["n_subjects_plot", "mean_hierarchy"])

    # Collapse accidental duplicates safely.
    group_cols = [
        "dataset",
        "feature",
        "space",
        "num_classes",
        "n_subjects",
        "n_subjects_plot",
        "n_subjects_label",
    ]

    agg_cols = {
        "mean_hierarchy": "mean",
        "std_hierarchy": "mean",
        "mean_inter": "mean",
        "std_inter": "mean",
        "mean_intra": "mean",
        "std_intra": "mean",
        "mean_pval": "mean",
        "std_pval": "mean",
    }

    existing_agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}

    df = (
        df.groupby(group_cols, dropna=False)
        .agg(existing_agg_cols)
        .reset_index()
        .sort_values(["feature", "n_subjects_plot"])
    )

    return df


def plot_hierarchy_ratio(df, dataset, out_dir, num_classes=None):
    features = [
        f for f in ["psd", "entropy", "complexity"]
        if f in df["feature"].unique()
    ]

    subject_order_df = (
        df[["n_subjects_plot", "n_subjects_label"]]
        .drop_duplicates()
        .sort_values("n_subjects_plot")
    )

    subject_labels = subject_order_df["n_subjects_label"].tolist()
    subject_positions = np.arange(len(subject_labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    total_group_width = 0.8
    bar_width = total_group_width / max(len(features), 1)

    for feat_idx, feature in enumerate(features):
        sub = df[df["feature"] == feature].copy()

        sub = (
            sub.set_index("n_subjects_label")
            .reindex(subject_labels)
            .reset_index()
        )

        x = (
            subject_positions
            - total_group_width / 2
            + feat_idx * bar_width
            + bar_width / 2
        )

        y = sub["mean_hierarchy"].to_numpy(dtype=float)
        yerr = sub["std_hierarchy"].fillna(0.0).to_numpy(dtype=float)
        mask = ~np.isnan(y)

        ax.bar(
            x[mask],
            y[mask],
            width=bar_width,
            yerr=yerr[mask],
            capsize=4,
            alpha=0.9,
            label=feature
        )

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of Subjects")
    ax.set_ylabel("Hierarchy Ratio")
    ax.set_xticks(subject_positions)
    ax.set_xticklabels(subject_labels)
    ax.legend(title="Feature")

    title = f"Unsupervised — {dataset}"
    if num_classes is not None:
        title += f" ({num_classes} classes)"
    ax.set_title(title)

    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(
        out_dir,
        f"{dataset}{suffix}_h1_unsupervised_hierarchy_ratio_bars.png"
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")


def plot_inter_intra(df, dataset, out_dir, num_classes=None):
    features = [
        f for f in ["psd", "entropy", "complexity"]
        if f in df["feature"].unique()
    ]

    n_features = len(features)

    fig, axes = plt.subplots(
        1,
        n_features,
        figsize=(6 * n_features, 4.5),
        squeeze=False,
        sharey=False
    )

    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        sub = df[df["feature"] == feature].copy()
        sub = sub.sort_values("n_subjects_plot")

        x = np.arange(len(sub))
        labels = sub["n_subjects_label"].tolist()
        width = 0.35

        ax.bar(
            x - width / 2,
            sub["mean_inter"],
            yerr=sub["std_inter"].fillna(0.0),
            width=width,
            capsize=4,
            label="Inter-subject"
        )

        ax.bar(
            x + width / 2,
            sub["mean_intra"],
            yerr=sub["std_intra"].fillna(0.0),
            width=width,
            capsize=4,
            label="Intra-subject"
        )

        ax.set_title(feature)
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("Distance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.05)
    )

    title = f"Inter vs Intra Distances — Unsupervised / {dataset}"
    if num_classes is not None:
        title += f" ({num_classes} classes)"

    fig.suptitle(title, y=1.12)
    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(
        out_dir,
        f"{dataset}{suffix}_h1_unsupervised_inter_intra.png"
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = collect_unsupervised_results(
        args.results_dir,
        args.dataset,
        num_classes=args.num_classes
    )

    suffix = f"_{args.num_classes}classes" if args.num_classes is not None else ""
    csv_path = os.path.join(
        args.out_dir,
        f"{args.dataset}{suffix}_h1_unsupervised_loaded.csv"
    )

    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    plot_hierarchy_ratio(
        df,
        args.dataset,
        args.out_dir,
        num_classes=args.num_classes
    )

    plot_inter_intra(
        df,
        args.dataset,
        args.out_dir,
        num_classes=args.num_classes
    )


if __name__ == "__main__":
    main()