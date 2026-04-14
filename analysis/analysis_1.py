import os 
import numpy as np


"""
Plots:
1) Increase number of subjects -> hierarchy value increases even for State LDA 
2) Does this happen across features?
3) Accuracy using joint LDA for trait and state accuracy across number of subjects
4) use sklearn cohen’s kappa (can compare across classifiers w/ dissimilar number of classes)
5) Does have a trait-state hierarchy
- high inter-subject variability
6) Features that have lower trait-state metric - better suited for state classification
- may be observed in deep learning/foundation models
One table/plot for each analysis point
"""

import os
import re
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "/home1/amadapur/projects/eeg_trait_state_geometry"
out_dir = "/home1/amadapur/projects/eeg_trait_state_geometry/plots/hierarchy_by_space"
os.makedirs(out_dir, exist_ok=True)

sns.set_theme(style="whitegrid")


def parse_n_subjects(path):
    """
    Extract subject count from strings like n4, n10, n20, nall.
    """
    m = re.search(r'n(\d+|all)', path)
    if not m:
        return None

    val = m.group(1)
    if val == "all":
        return "all"
    return int(val)

def parse_num_classes(path):
    """
    Extract class count from:
    _6classes
    _6_complexityclasses
    _10_entropyclasses
    etc.
    """
    fname = os.path.basename(path)

    # match: _6classes OR _6_complexityclasses OR _6_entropyclasses
    m = re.search(r'_(\d+)(?:_[a-z]+)?classes', fname)

    if not m:
        return None

    return int(m.group(1))


def infer_dataset_feature(path):
    """
    Assumes paths like:
    results/gamma/complexity/hierarchy/...
    results/lee/psd/hierarchy/...
    results/lemon/entropy/hierarchy/...

    Returns:
        dataset, feature
    """
    parts = os.path.normpath(path).split(os.sep)

    if "results" in parts:
        idx = parts.index("results")
        after = parts[idx + 1:]
    else:
        after = parts

    dataset = after[0] if len(after) > 0 else "unknown_dataset"

    known_features = {"psd", "entropy", "complexity"}
    feature = "unknown_feature"
    for p in after[1:]:
        if p.lower() in known_features:
            feature = p.lower()
            break

    return dataset, feature


def load_hierarchy_csv(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = {"space", "seed", "hier_ratio"}
    if not required_cols.issubset(df.columns):
        print(f"Skipping {csv_path}: missing required columns {required_cols - set(df.columns)}")
        return None

    dataset, feature = infer_dataset_feature(csv_path)
    n_subjects = parse_n_subjects(csv_path)
    if dataset == "motor" and n_subjects == 109:
        n_subjects = "all"
    num_classes = parse_num_classes(csv_path)

    df = df.copy()
    df["dataset"] = dataset
    df["feature"] = feature
    df["n_subjects"] = n_subjects
    df["source_file"] = csv_path
    df["num_classes"] = num_classes

    return df


def collect_all_hierarchy_results(results_dir):
    pattern = os.path.join(results_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)

    hierarchy_files = [
        f for f in files
        if "hierarchy" in os.path.basename(f).lower() or "hierarchy" in f.lower()
    ]

    if not hierarchy_files:
        raise FileNotFoundError(f"No hierarchy CSV files found under {results_dir}")

    dfs = []
    for f in hierarchy_files:
        try:
            df = load_hierarchy_csv(f)
            if df is not None:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        raise ValueError("No valid hierarchy CSVs could be loaded.")

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def add_subject_order_info(df):
    """
    Add numeric ordering for n_subjects so bars appear in the correct order.
    Also create a clean display label.
    """
    df = df.copy()

    

    dataset_all_map = {
        "motor": 105,
        "lee": 54,
        "lemon": 156,
        "gamma": 14,   # add if needed
    }

    def map_plot_value(row):
        if row["n_subjects"] == "all":
            return dataset_all_map.get(row["dataset"], np.nan)
        return row["n_subjects"]

    df["n_subjects_plot"] = df.apply(map_plot_value, axis=1)
    df["n_subjects_plot"] = pd.to_numeric(df["n_subjects_plot"], errors="coerce")

    def make_label(row):
        if row["n_subjects"] == "all":
            mapped = dataset_all_map.get(row["dataset"], "all")
            return f"all ({mapped})"
        return str(row["n_subjects"])

    df["n_subjects_label"] = df.apply(make_label, axis=1)

    return df


def summarize_across_seeds(df):
    group_cols = ["dataset", "feature", "space", "n_subjects", "n_subjects_plot", "n_subjects_label", "num_classes"]

    agg_dict = {
        "mean_hierarchy": ("hier_ratio", "mean"),
        "std_hierarchy": ("hier_ratio", "std"),
        "n_seeds": ("hier_ratio", "count"),
    }

    if "hier_pval" in df.columns:
        agg_dict["mean_pval"] = ("hier_pval", "mean")

    summary = (
        df.groupby(group_cols, dropna=False)
          .agg(**agg_dict)
          .reset_index()
    )

    summary["std_hierarchy"] = summary["std_hierarchy"].fillna(0.0)
    summary["sem_hierarchy"] = summary["std_hierarchy"] / np.sqrt(summary["n_seeds"].clip(lower=1))

    return summary


def save_tables(raw_df, summary_df, out_dir):
    raw_path = os.path.join(out_dir, "all_hierarchy_results_combined.csv")
    summary_path = os.path.join(out_dir, "all_hierarchy_summary_across_seeds.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")


def plot_dataset_bars_by_space(summary_df, dataset, out_dir, num_classes=None):
    """
    One figure per dataset.
    Separate subplot for each space.
    Bars within each subplot are grouped by feature.
    """
    ds = summary_df[summary_df["dataset"] == dataset].copy()
    if ds.empty:
        return

    # enforce ordering
    ds = ds.sort_values(["space", "n_subjects_plot", "feature"])

    space_order = ["Raw", "Trait_LDA", "State_LDA", "Joint_LDA"]
    existing_spaces = [s for s in space_order if s in ds["space"].unique()]
    remaining_spaces = [s for s in sorted(ds["space"].unique()) if s not in existing_spaces]
    spaces = existing_spaces + remaining_spaces

    feature_order = [f for f in ["psd", "entropy", "complexity"] if f in ds["feature"].unique()]
    remaining_features = [f for f in sorted(ds["feature"].unique()) if f not in feature_order]
    features = feature_order + remaining_features

    subject_order_df = (
        ds[["n_subjects_plot", "n_subjects_label"]]
        .drop_duplicates()
        .sort_values("n_subjects_plot")
    )
    subject_labels = subject_order_df["n_subjects_label"].tolist()
    subject_positions = np.arange(len(subject_labels))

    n_spaces = len(spaces)
    ncols = 2
    nrows = math.ceil(n_spaces / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()

    total_group_width = 0.8
    bar_width = total_group_width / max(len(features), 1)

    for ax_idx, space in enumerate(spaces):
        ax = axes[ax_idx]
        sub = ds[ds["space"] == space].copy()

        for feat_idx, feature in enumerate(features):
            feat_sub = sub[sub["feature"] == feature].copy()

            # align to all subject labels so bars stay in the right positions
            feat_sub = feat_sub.set_index("n_subjects_label").reindex(subject_labels).reset_index()

            x = subject_positions - total_group_width / 2 + feat_idx * bar_width + bar_width / 2
            y = feat_sub["mean_hierarchy"].to_numpy(dtype=float)
            yerr = feat_sub["sem_hierarchy"].fillna(0.0).to_numpy(dtype=float)

            mask = ~np.isnan(y)

            ax.bar(
                x[mask],
                y[mask],
                width=bar_width,
                label=feature,
                yerr=yerr[mask],
                capsize=4,
                alpha=0.9
            )

        ax.set_title(space)
        ax.set_xticks(subject_positions)
        ax.set_xticklabels(subject_labels)
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("Hierarchy Ratio")

    # remove unused axes
    for j in range(n_spaces, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(features), bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"Hierarchy vs Number of Subjects — Dataset: {dataset}", y=1.06, fontsize=16)
    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(out_dir, f"{dataset}{suffix}_hierarchy_bars_by_space.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_dataset_feature_bars(summary_df, dataset, feature, out_dir, num_classes=None):
    """
    Optional: one figure per dataset-feature.
    Separate subplot for each space.
    Bars are just subject counts.
    """
    sub = summary_df[
        (summary_df["dataset"] == dataset) &
        (summary_df["feature"] == feature)
    ].copy()

    if sub.empty:
        return

    sub = sub.sort_values(["space", "n_subjects_plot"])

    space_order = ["Raw", "Trait_LDA", "State_LDA", "Joint_LDA"]
    existing_spaces = [s for s in space_order if s in sub["space"].unique()]
    remaining_spaces = [s for s in sorted(sub["space"].unique()) if s not in existing_spaces]
    spaces = existing_spaces + remaining_spaces

    n_spaces = len(spaces)
    ncols = 2
    nrows = math.ceil(n_spaces / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, space in enumerate(spaces):
        ax = axes[ax_idx]
        s = sub[sub["space"] == space].copy().sort_values("n_subjects_plot")

        x = np.arange(len(s))
        ax.bar(
            x,
            s["mean_hierarchy"],
            yerr=s["sem_hierarchy"],
            capsize=4,
            alpha=0.9
        )
        ax.set_title(space)
        ax.set_xticks(x)
        ax.set_xticklabels(s["n_subjects_label"].tolist())
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("Hierarchy Ratio")

    for j in range(n_spaces, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Hierarchy vs Number of Subjects — {dataset} / {feature}", y=1.03, fontsize=16)
    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(out_dir, f"{dataset}_{feature}{suffix}_hierarchy_bars.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
def main():
    raw_df = collect_all_hierarchy_results(results_dir)
    raw_df = add_subject_order_info(raw_df)

    raw_df = raw_df.dropna(subset=["n_subjects_plot", "hier_ratio", "space", "dataset", "feature"])

    summary_df = summarize_across_seeds(raw_df)
    save_tables(raw_df, summary_df, out_dir)

    # only motor, and make separate plots for 6-class and 10-class
    dataset = "motor"

    for num_classes in [6, 10]:
        class_summary = summary_df[
            (summary_df["dataset"] == dataset) &
            (summary_df["num_classes"] == num_classes)
        ].copy()

        if class_summary.empty:
            print(f"No data found for {dataset}, {num_classes} classes")
            continue

        plot_dataset_bars_by_space(class_summary, dataset, out_dir, num_classes=num_classes)

        dataset_features = sorted(class_summary["feature"].unique())
        for feature in dataset_features:
            plot_dataset_feature_bars(
                class_summary,
                dataset,
                feature,
                out_dir,
                num_classes=num_classes
            )

    print("Done.")

"""
def main():
    
    
    raw_df = collect_all_hierarchy_results(results_dir)
    raw_df = add_subject_order_info(raw_df)

    raw_df = raw_df.dropna(subset=["n_subjects_plot", "hier_ratio", "space", "dataset", "feature"])

    summary_df = summarize_across_seeds(raw_df)
    summary_df = summary_df[
        (summary_df["dataset"] == "motor") &
        (summary_df["num_classes"].isin([6, 10]))
    ].copy()

    raw_df = raw_df[
        (raw_df["dataset"] == "motor") &
        (raw_df["num_classes"].isin([6, 10]))
    ].copy()
    
    save_tables(raw_df, summary_df, out_dir)

    datasets = sorted(summary_df["dataset"].unique())
    datasets = ["motor"]
    for dataset in datasets:
        plot_dataset_bars_by_space(summary_df, dataset, out_dir)

        # optional extra plots: one per feature
        dataset_features = sorted(summary_df[summary_df["dataset"] == dataset]["feature"].unique())
        for feature in dataset_features:
            plot_dataset_feature_bars(summary_df, dataset, feature, out_dir)

    print("Done.")

"""
if __name__ == "__main__":
    main()