import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "/home1/amadapur/projects/eeg_trait_state_geometry"
out_dir = "/home1/amadapur/projects/eeg_trait_state_geometry/plots/all_hierarchy_metrics"
os.makedirs(out_dir, exist_ok=True)

sns.set_theme(style="whitegrid")


def parse_n_subjects(path):
    m = re.search(r'n(\d+|all)', path)
    if not m:
        return None
    val = m.group(1)
    return "all" if val == "all" else int(val)


def parse_num_classes(path):
    fname = os.path.basename(path)
    m = re.search(r'_(\d+)(?:_[a-z]+)?classes', fname)
    if not m:
        return None
    return int(m.group(1))


def infer_dataset_feature(path):
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


def resolve_distance_columns(df):
    """
    Try to detect inter/intra distance column names automatically.
    Adjust/add candidates here if your CSVs use other names.
    """
    inter_candidates = [
        "inter_dist", "inter_distance", "mean_inter_dist", "mean_inter_distance",
        "inter", "avg_inter", "inter_subject_dist", "inter_subject_distance"
    ]
    intra_candidates = [
        "intra_dist", "intra_distance", "mean_intra_dist", "mean_intra_distance",
        "intra", "avg_intra", "intra_subject_dist", "intra_subject_distance"
    ]

    inter_col = next((c for c in inter_candidates if c in df.columns), None)
    intra_col = next((c for c in intra_candidates if c in df.columns), None)

    return inter_col, intra_col


def load_hierarchy_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "space" not in df.columns or "seed" not in df.columns:
        print(f"Skipping {csv_path}: missing required columns")
        return None

    if "hier_ratio" not in df.columns:
        print(f"Skipping {csv_path}: missing hier_ratio")
        return None

    dataset, feature = infer_dataset_feature(csv_path)
    n_subjects = parse_n_subjects(csv_path)
    if dataset == "motor" and n_subjects == 109:
        n_subjects = "all"
    num_classes = parse_num_classes(csv_path)

    inter_col, intra_col = resolve_distance_columns(df)

    df = df.copy()
    df["dataset"] = dataset
    df["feature"] = feature
    df["n_subjects"] = n_subjects
    df["source_file"] = csv_path
    df["num_classes"] = num_classes

    # standardize names so downstream code is simple
    if inter_col is not None:
        df["inter_dist_value"] = pd.to_numeric(df[inter_col], errors="coerce")
    else:
        df["inter_dist_value"] = np.nan

    if intra_col is not None:
        df["intra_dist_value"] = pd.to_numeric(df[intra_col], errors="coerce")
    else:
        df["intra_dist_value"] = np.nan

    df["hier_ratio"] = pd.to_numeric(df["hier_ratio"], errors="coerce")

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

    return pd.concat(dfs, ignore_index=True)


def add_subject_order_info(df):
    df = df.copy()

    dataset_all_map = {
        "motor": 105,
        "lee": 54,
        "lemon": 156,
        "gamma": 14,
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
    group_cols = [
        "dataset", "feature", "space",
        "n_subjects", "n_subjects_plot", "n_subjects_label", "num_classes"
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
          .agg(
              mean_hierarchy=("hier_ratio", "mean"),
              std_hierarchy=("hier_ratio", "std"),
              n_seeds=("hier_ratio", "count"),

              mean_inter=("inter_dist_value", "mean"),
              std_inter=("inter_dist_value", "std"),

              mean_intra=("intra_dist_value", "mean"),
              std_intra=("intra_dist_value", "std"),
          )
          .reset_index()
    )

    for col in ["std_hierarchy", "std_inter", "std_intra"]:
        summary[col] = summary[col].fillna(0.0)

    summary["sem_hierarchy"] = summary["std_hierarchy"] / np.sqrt(summary["n_seeds"].clip(lower=1))
    summary["sem_inter"] = summary["std_inter"] / np.sqrt(summary["n_seeds"].clip(lower=1))
    summary["sem_intra"] = summary["std_intra"] / np.sqrt(summary["n_seeds"].clip(lower=1))

    return summary


def save_tables(raw_df, summary_df, out_dir):
    raw_path = os.path.join(out_dir, "all_hierarchy_results_combined.csv")
    summary_path = os.path.join(out_dir, "all_hierarchy_summary_across_seeds.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")


def plot_inter_intra_by_space(summary_df, dataset, out_dir, num_classes=None):
    """
    For each space:
      - line for mean inter distance
      - line for mean intra distance
    This directly shows how each component changes with subject count.
    """
    ds = summary_df[summary_df["dataset"] == dataset].copy()
    if ds.empty:
        return

    ds = ds.sort_values(["space", "n_subjects_plot", "feature"])

    space_order = ["Raw", "Trait_LDA", "State_LDA", "Joint_LDA"]
    existing_spaces = [s for s in space_order if s in ds["space"].unique()]
    remaining_spaces = [s for s in sorted(ds["space"].unique()) if s not in existing_spaces]
    spaces = existing_spaces + remaining_spaces

    feature_order = [f for f in ["psd", "entropy", "complexity"] if f in ds["feature"].unique()]
    remaining_features = [f for f in sorted(ds["feature"].unique()) if f not in feature_order]
    features = feature_order + remaining_features

    n_spaces = len(spaces)
    ncols = 2
    nrows = math.ceil(n_spaces / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, space in enumerate(spaces):
        ax = axes[ax_idx]
        sub = ds[ds["space"] == space].copy()

        for feature in features:
            feat_sub = sub[sub["feature"] == feature].copy().sort_values("n_subjects_plot")

            x = feat_sub["n_subjects_plot"].to_numpy(dtype=float)

            y_inter = feat_sub["mean_inter"].to_numpy(dtype=float)
            y_intra = feat_sub["mean_intra"].to_numpy(dtype=float)

            e_inter = feat_sub["sem_inter"].to_numpy(dtype=float)
            e_intra = feat_sub["sem_intra"].to_numpy(dtype=float)

            if not np.all(np.isnan(y_inter)):
                ax.errorbar(
                    x, y_inter, yerr=e_inter,
                    marker="o", capsize=4,
                    label=f"{feature} inter"
                )

            if not np.all(np.isnan(y_intra)):
                ax.errorbar(
                    x, y_intra, yerr=e_intra,
                    marker="s", linestyle="--", capsize=4,
                    label=f"{feature} intra"
                )

        ax.set_title(space)
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("Distance")
        ax.set_xticks(sorted(sub["n_subjects_plot"].dropna().unique()))

    for j in range(n_spaces, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))

    fig.suptitle(f"Inter vs Intra Distance — Dataset: {dataset}", y=1.08, fontsize=16)
    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(out_dir, f"{dataset}{suffix}_inter_intra_by_space.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_distance_gap_by_space(summary_df, dataset, out_dir, num_classes=None):
    """
    Plot (inter - intra) instead of ratio.
    Helpful for seeing separation directly.
    """
    ds = summary_df[summary_df["dataset"] == dataset].copy()
    if ds.empty:
        return

    ds = ds.copy()
    ds["mean_gap"] = ds["mean_inter"] - ds["mean_intra"]

    space_order = ["Raw", "Trait_LDA", "State_LDA", "Joint_LDA"]
    existing_spaces = [s for s in space_order if s in ds["space"].unique()]
    remaining_spaces = [s for s in sorted(ds["space"].unique()) if s not in existing_spaces]
    spaces = existing_spaces + remaining_spaces

    feature_order = [f for f in ["psd", "entropy", "complexity"] if f in ds["feature"].unique()]
    remaining_features = [f for f in sorted(ds["feature"].unique()) if f not in feature_order]
    features = feature_order + remaining_features

    n_spaces = len(spaces)
    ncols = 2
    nrows = math.ceil(n_spaces / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, space in enumerate(spaces):
        ax = axes[ax_idx]
        sub = ds[ds["space"] == space].copy()

        for feature in features:
            feat_sub = sub[sub["feature"] == feature].copy().sort_values("n_subjects_plot")
            x = feat_sub["n_subjects_plot"].to_numpy(dtype=float)
            y = feat_sub["mean_gap"].to_numpy(dtype=float)

            ax.plot(x, y, marker="o", label=feature)

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(space)
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("Inter - Intra Distance")
        ax.set_xticks(sorted(sub["n_subjects_plot"].dropna().unique()))

    for j in range(n_spaces, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(features), bbox_to_anchor=(0.5, 1.03))

    fig.suptitle(f"Distance Gap vs Number of Subjects — Dataset: {dataset}", y=1.08, fontsize=16)
    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(out_dir, f"{dataset}{suffix}_distance_gap_by_space.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_hierarchy_by_space(summary_df, dataset, out_dir, num_classes=None):
    """
    Keep the old hierarchy plot too, for comparison.
    """
    ds = summary_df[summary_df["dataset"] == dataset].copy()
    if ds.empty:
        return

    ds = ds.sort_values(["space", "n_subjects_plot", "feature"])

    space_order = ["Raw", "Trait_LDA", "State_LDA", "Joint_LDA"]
    existing_spaces = [s for s in space_order if s in ds["space"].unique()]
    remaining_spaces = [s for s in sorted(ds["space"].unique()) if s not in existing_spaces]
    spaces = existing_spaces + remaining_spaces

    feature_order = [f for f in ["psd", "entropy", "complexity"] if f in ds["feature"].unique()]
    remaining_features = [f for f in sorted(ds["feature"].unique()) if f not in feature_order]
    features = feature_order + remaining_features

    n_spaces = len(spaces)
    ncols = 2
    nrows = math.ceil(n_spaces / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, space in enumerate(spaces):
        ax = axes[ax_idx]
        sub = ds[ds["space"] == space].copy()

        for feature in features:
            feat_sub = sub[sub["feature"] == feature].copy().sort_values("n_subjects_plot")
            x = feat_sub["n_subjects_plot"].to_numpy(dtype=float)
            y = feat_sub["mean_hierarchy"].to_numpy(dtype=float)
            e = feat_sub["sem_hierarchy"].to_numpy(dtype=float)

            ax.errorbar(x, y, yerr=e, marker="o", capsize=4, label=feature)

        ax.set_title(space)
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("Hierarchy Ratio")
        ax.set_xticks(sorted(sub["n_subjects_plot"].dropna().unique()))

    for j in range(n_spaces, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(features), bbox_to_anchor=(0.5, 1.03))

    fig.suptitle(f"Hierarchy Ratio vs Number of Subjects — Dataset: {dataset}", y=1.08, fontsize=16)
    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(out_dir, f"{dataset}{suffix}_hierarchy_ratio_by_space.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    raw_df = collect_all_hierarchy_results(results_dir)
    raw_df = add_subject_order_info(raw_df)

    raw_df = raw_df.dropna(subset=["n_subjects_plot", "hier_ratio", "space", "dataset", "feature"])

    summary_df = summarize_across_seeds(raw_df)
    save_tables(raw_df, summary_df, out_dir)

    dataset = "motor"

    for num_classes in [6, 10]:
        class_summary = summary_df[
            (summary_df["dataset"] == dataset) &
            (summary_df["num_classes"] == num_classes)
        ].copy()

        if class_summary.empty:
            print(f"No data found for {dataset}, {num_classes} classes")
            continue

        plot_hierarchy_by_space(class_summary, dataset, out_dir, num_classes=num_classes)
        plot_inter_intra_by_space(class_summary, dataset, out_dir, num_classes=num_classes)
        plot_distance_gap_by_space(class_summary, dataset, out_dir, num_classes=num_classes)

    print("Done.")


if __name__ == "__main__":
    main()