#decoding accuracy analysis

import os
import re
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "/home1/amadapur/projects/eeg_trait_state_geometry/results"
out_dir = "/home1/amadapur/projects/eeg_trait_state_geometry/plots/geometry_accuracy_by_space"
os.makedirs(out_dir, exist_ok=True)

sns.set_theme(style="whitegrid")


def parse_n_subjects(path):
    m = re.search(r'n(\d+|all)', path)
    if not m:
        return None
    val = m.group(1)
    return "all" if val == "all" else int(val)


def parse_num_classes(path):
    m = re.search(r'_(\d+)classes', path)
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


def flatten_tworow_header(columns):
    flat = []
    for c in columns:
        if isinstance(c, tuple):
            left, right = c
            left = "" if left is None else str(left).strip()
            right = "" if right is None else str(right).strip()

            if left.startswith("Unnamed"):
                left = ""
            if right.startswith("Unnamed"):
                right = ""

            if left and right:
                flat.append(f"{left}_{right}")
            elif left:
                flat.append(left)
            elif right:
                flat.append(right)
            else:
                flat.append("")
        else:
            flat.append(str(c).strip())
    return flat


def load_geometry_csv(csv_path):
    df = pd.read_csv(csv_path, header=[0, 1])
    df.columns = flatten_tworow_header(df.columns)

    # if first column is unnamed, it actually contains the space labels
    if "space" not in df.columns:
        first_col = df.columns[0]
        if first_col == "" or str(first_col).startswith("Unnamed"):
            df = df.rename(columns={first_col: "space"})

    # sometimes there's a bogus first row where the value is literally "space"
    if "space" in df.columns:
        df = df[df["space"].astype(str).str.strip().str.lower() != "space"]

    if "space" not in df.columns:
        print(f"Skipping {csv_path}: no 'space' column after flattening")
        return None

    dataset, feature = infer_dataset_feature(csv_path)
    n_subjects = parse_n_subjects(csv_path)
    num_classes = parse_num_classes(csv_path)

    if dataset == "motor" and n_subjects == 109:
        n_subjects = "all"

    df = df.copy()
    df["dataset"] = dataset
    df["feature"] = feature
    df["n_subjects"] = n_subjects
    df["num_classes"] = num_classes
    df["source_file"] = csv_path

    return df


def collect_all_geometry_results(results_dir):
    pattern = os.path.join(results_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)

    geometry_files = [
        f for f in files
        if "geometry" in os.path.basename(f).lower() or "geometry" in f.lower()
    ]

    if not geometry_files:
        raise FileNotFoundError(f"No geometry CSV files found under {results_dir}")

    dfs = []
    for f in geometry_files:
        try:
            df = load_geometry_csv(f)
            if df is not None:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        raise ValueError("No valid geometry CSVs could be loaded.")

    return pd.concat(dfs, ignore_index=True)


def reshape_geometry_long(df):
    """
    Turn wide columns like:
      trait_ident_acc_mean, trait_ident_acc_std, within_state_acc_mean, ...
    into rows:
      metric, stat, value
    """
    metric_cols = []
    for c in df.columns:
        if c.endswith("_mean") or c.endswith("_std"):
            metric_cols.append(c)

    id_vars = [c for c in df.columns if c not in metric_cols]

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=metric_cols,
        var_name="metric_stat",
        value_name="value"
    )

    long_df["stat"] = long_df["metric_stat"].str.extract(r'(mean|std)$')
    long_df["metric"] = long_df["metric_stat"].str.replace(r'_(mean|std)$', '', regex=True)

    long_df = long_df.drop(columns=["metric_stat"])
    return long_df


def plot_geometry_metric_by_space(long_df, dataset, metric, num_classes, out_dir):
    if num_classes is None:
        ds = long_df[
            (long_df["dataset"] == dataset) &
            (long_df["metric"] == metric) &
            (long_df["num_classes"].isna()) &
            (long_df["stat"] == "mean")
        ].copy()
        std_df = long_df[
            (long_df["dataset"] == dataset) &
            (long_df["metric"] == metric) &
            (long_df["num_classes"].isna()) &
            (long_df["stat"] == "std")
        ].copy()
        title = f"{metric} vs Number of Subjects — {dataset}"
        out_name = f"{dataset}_{metric}_by_space.png"
    else:
        ds = long_df[
            (long_df["dataset"] == dataset) &
            (long_df["metric"] == metric) &
            (long_df["num_classes"] == num_classes) &
            (long_df["stat"] == "mean")
        ].copy()
        std_df = long_df[
            (long_df["dataset"] == dataset) &
            (long_df["metric"] == metric) &
            (long_df["num_classes"] == num_classes) &
            (long_df["stat"] == "std")
        ].copy()
        title = f"{metric} vs Number of Subjects — {dataset} ({int(num_classes)} classes)"
        out_name = f"{dataset}_{int(num_classes)}classes_{metric}_by_space.png"

    if ds.empty:
        return

    ds = ds.rename(columns={"value": "mean_value"})
    std_df = std_df.rename(columns={"value": "std_value"})

    merged = ds.merge(
        std_df[["dataset", "feature", "space", "n_subjects_label", "n_subjects_plot", "metric", "std_value"]],
        on=["dataset", "feature", "space", "n_subjects_label", "n_subjects_plot", "metric"],
        how="left"
    )

    merged = merged.sort_values(["space", "n_subjects_plot", "feature"])

    space_order = ["Raw", "Trait_LDA", "State_LDA", "Joint_LDA"]
    existing_spaces = [s for s in space_order if s in merged["space"].unique()]
    remaining_spaces = [s for s in sorted(merged["space"].unique()) if s not in existing_spaces]
    spaces = existing_spaces + remaining_spaces

    feature_order = [f for f in ["psd", "entropy", "complexity"] if f in merged["feature"].unique()]
    remaining_features = [f for f in sorted(merged["feature"].unique()) if f not in feature_order]
    features = feature_order + remaining_features

    subject_order_df = (
        merged[["n_subjects_plot", "n_subjects_label"]]
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
        sub = merged[merged["space"] == space].copy()

        for feat_idx, feature in enumerate(features):
            feat_sub = sub[sub["feature"] == feature].copy()
            feat_sub = feat_sub.set_index("n_subjects_label").reindex(subject_labels).reset_index()

            x = subject_positions - total_group_width / 2 + feat_idx * bar_width + bar_width / 2
            y = feat_sub["mean_value"].to_numpy(dtype=float)
            yerr = feat_sub["std_value"].fillna(0.0).to_numpy(dtype=float)

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
        ax.set_ylabel(metric)

        # optional if all these are accuracies
        if "acc" in metric:
            ax.set_ylim(0, 1.05)

    for j in range(n_spaces, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(features), bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(title, y=1.06, fontsize=16)
    fig.tight_layout()

    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    raw_df = collect_all_geometry_results(results_dir)
    raw_df = add_subject_order_info(raw_df)
    raw_df = raw_df.dropna(subset=["n_subjects_plot", "space", "dataset", "feature"])

    long_df = reshape_geometry_long(raw_df)

    datasets = sorted(long_df["dataset"].dropna().unique())
    metrics = sorted(long_df["metric"].dropna().unique())

    print("Datasets:", datasets)
    print("Metrics:", metrics)

    for dataset in datasets:
        dataset_df = long_df[long_df["dataset"] == dataset]
        class_values = sorted(dataset_df["num_classes"].dropna().unique())

        if len(class_values) == 0:
            for metric in metrics:
                plot_geometry_metric_by_space(long_df, dataset, metric, None, out_dir)
        else:
            for num_classes in class_values:
                for metric in metrics:
                    plot_geometry_metric_by_space(long_df, dataset, metric, num_classes, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()