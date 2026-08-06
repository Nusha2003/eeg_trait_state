import os
import re
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def parse_n_subjects_from_filename(path):
    fname = os.path.basename(path)
    m = re.search(r"_n(\d+|all)\b", fname)
    if not m:
        return None
    val = m.group(1)
    return "all" if val == "all" else int(val)


def parse_num_classes(path):
    fname = os.path.basename(path)
    m = re.search(r"_(\d+)classes", fname)
    if not m:
        return None
    return int(m.group(1))


def infer_dataset_feature(path):
    parts = os.path.normpath(path).split(os.sep)

    dataset = "unknown_dataset"
    feature = "unknown_feature"

    known_datasets = {"motor", "lee", "lemon", "gamma", "bci"}
    known_features = {"psd", "entropy", "complexity"}

    for p in parts:
        p_low = p.lower()
        if p_low in known_datasets:
            dataset = p_low
        if p_low in known_features:
            feature = p_low

    return dataset, feature


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


def normalize_loaded_columns(df):
    rename_map = {
        "num_subjects": "num_subjects",
        "space": "space",
        "hier_ratio_mean": "mean_hierarchy",
        "hier_ratio_std": "std_hierarchy",
        "inter_mean": "mean_inter",
        "inter_std": "std_inter",
        "intra_mean": "mean_intra",
        "intra_std": "std_intra",
        "hier_pval_mean": "mean_pval",
        "hier_pval_std": "std_pval",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def load_hierarchy_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, header=[0, 1], skiprows=[2])
    except Exception as e:
        print(f"Skipping {csv_path}: failed to read with 2-row header: {e}")
        return None
    df = flatten_multilevel_columns(df)
    df["space"] = df["space"].replace({
        "Raw": "Unsupervised"
    })

    rename_map = {
        "hier_ratio_mean": "mean_hierarchy",
        "hier_ratio_std": "std_hierarchy",
        "inter_mean": "mean_inter",
        "inter_std": "std_inter",
        "intra_mean": "mean_intra",
        "intra_std": "std_intra",
        "hier_pval_mean": "mean_pval",
        "hier_pval_std": "std_pval",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_cols = {"num_subjects", "space", "mean_hierarchy"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Skipping {csv_path}: missing required columns {missing}")
        print("Columns found:", df.columns.tolist())
        return None

    dataset, feature = infer_dataset_feature(csv_path)
    num_classes = parse_num_classes(csv_path)
    n_from_fname = parse_n_subjects_from_filename(csv_path)

    df = df.copy()
    df["dataset"] = dataset
    df["feature"] = feature
    df["num_classes"] = num_classes
    df["source_file"] = csv_path

    def parse_subject_value(x):
        if pd.isna(x):
            return n_from_fname

        x_str = str(x).strip().lower()
        if x_str in {"all", "nall"}:
            return "all"

        try:
            return int(float(x))
        except Exception:
            return n_from_fname

    df["n_subjects"] = df["num_subjects"].apply(parse_subject_value)

    numeric_cols = [
        "mean_hierarchy", "std_hierarchy",
        "mean_inter", "std_inter",
        "mean_intra", "std_intra",
        "mean_pval", "std_pval"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "std_hierarchy" not in df.columns:
        df["std_hierarchy"] = 0.0
    if "mean_inter" not in df.columns:
        df["mean_inter"] = np.nan
    if "std_inter" not in df.columns:
        df["std_inter"] = np.nan
    if "mean_intra" not in df.columns:
        df["mean_intra"] = np.nan
    if "std_intra" not in df.columns:
        df["std_intra"] = np.nan
    if "mean_pval" not in df.columns:
        df["mean_pval"] = np.nan
    if "std_pval" not in df.columns:
        df["std_pval"] = np.nan

    return df


def collect_all_hierarchy_results(results_dir, dataset):
    dataset_dir = os.path.join(results_dir, dataset)
    pattern = os.path.join(dataset_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)

    hierarchy_files = [f for f in files if "hierarchy" in f.lower()]

    if not hierarchy_files:
        raise FileNotFoundError(f"No hierarchy CSV files found under {dataset_dir}")

    dfs = []
    for f in hierarchy_files:
        df = load_hierarchy_csv(f)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid hierarchy CSVs could be loaded for dataset={dataset}.")

    return pd.concat(dfs, ignore_index=True)


def add_subject_order_info(df):
    df = df.copy()

    dataset_all_map = {
        "motor": 105,
        "lee": 54,
        "lemon": 156,
        "gamma": 14,
        "bci": 14
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
        return str(int(row["n_subjects"]))

    df["n_subjects_label"] = df.apply(make_label, axis=1)
    return df


def clean_summary_df(df):
    df = df.copy()

    numeric_cols = [
        "mean_hierarchy", "std_hierarchy",
        "mean_inter", "std_inter",
        "mean_intra", "std_intra",
        "mean_pval", "std_pval",
        "n_subjects_plot"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sem_hierarchy"] = df["std_hierarchy"].fillna(0.0)

    dedup_cols = [
        "dataset", "feature", "space", "n_subjects",
        "n_subjects_plot", "n_subjects_label", "num_classes"
    ]

    df = (
        df.sort_values(dedup_cols)
        .drop_duplicates(subset=dedup_cols, keep="first")
        .reset_index(drop=True)
    )
    return df


def save_tables(raw_df, summary_df, out_dir, dataset):
    raw_path = os.path.join(out_dir, f"{dataset}_all_hierarchy_results_combined.csv")
    summary_path = os.path.join(out_dir, f"{dataset}_all_hierarchy_summary_loaded.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")


def get_space_order(values):
    preferred = ["Unsupervised", "Trait_LDA", "State_LDA", "Joint_LDA"]
    existing = [v for v in preferred if v in values]
    remaining = [v for v in sorted(values) if v not in existing]
    return existing + remaining


def get_feature_order(values):
    preferred = ["psd", "entropy", "complexity"]
    existing = [v for v in preferred if v in values]
    remaining = [v for v in sorted(values) if v not in existing]
    return existing + remaining


def plot_dataset_bars_by_space(summary_df, dataset, out_dir, num_classes=None):
    ds = summary_df[summary_df["dataset"] == dataset].copy()
    if num_classes is not None:
        ds = ds[ds["num_classes"] == num_classes].copy()
    if ds.empty:
        return

    ds = ds.sort_values(["space", "n_subjects_plot", "feature"])
    spaces = get_space_order(ds["space"].dropna().unique())
    features = get_feature_order(ds["feature"].dropna().unique())

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

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
        sharey=True
    )
    axes = axes.flatten()

    total_group_width = 0.8
    bar_width = total_group_width / max(len(features), 1)

    for ax_idx, space in enumerate(spaces):
        ax = axes[ax_idx]
        sub = ds[ds["space"] == space].copy()

        for feat_idx, feature in enumerate(features):
            feat_sub = sub[sub["feature"] == feature].copy()
            feat_sub = feat_sub.set_index("n_subjects_label").reindex(subject_labels).reset_index()

            x = subject_positions - total_group_width / 2 + feat_idx * bar_width + bar_width / 2
            y = feat_sub["mean_hierarchy"].to_numpy(dtype=float)
            yerr = feat_sub["sem_hierarchy"].fillna(0.0).to_numpy(dtype=float)
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

        ax.set_title(space)
        ax.set_xticks(subject_positions)
        ax.set_xticklabels(subject_labels)
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("Hierarchy Ratio")

    for j in range(n_spaces, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(features), bbox_to_anchor=(0.5, 1.02))

    title = f"Hierarchy vs Number of Subjects — Dataset: {dataset}"
    if num_classes is not None:
        title += f" ({num_classes} classes)"
    fig.suptitle(title, y=1.06, fontsize=16)

    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(out_dir, f"{dataset}{suffix}_hierarchy_bars_by_space.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_dataset_feature_bars(summary_df, dataset, feature, out_dir, num_classes=None):
    sub = summary_df[
        (summary_df["dataset"] == dataset) &
        (summary_df["feature"] == feature)
    ].copy()

    if num_classes is not None:
        sub = sub[sub["num_classes"] == num_classes].copy()

    if sub.empty:
        return

    sub = sub.sort_values(["space", "n_subjects_plot"])
    spaces = get_space_order(sub["space"].dropna().unique())

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
            yerr=s["sem_hierarchy"].fillna(0.0),
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

    title = f"Hierarchy vs Number of Subjects — {dataset} / {feature}"
    if num_classes is not None:
        title += f" ({num_classes} classes)"
    fig.suptitle(title, y=1.03, fontsize=16)

    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(out_dir, f"{dataset}_{feature}{suffix}_hierarchy_bars.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_inter_intra_bars_per_feature(summary_df, dataset, feature, out_dir, num_classes=None):
    """
    One figure per feature.
    2x2 subplots for spaces.
    Each subplot shows grouped bars:
        - inter
        - intra
    across subject counts.
    """

    sub = summary_df[
        (summary_df["dataset"] == dataset) &
        (summary_df["feature"] == feature)
    ].copy()

    if num_classes is not None:
        sub = sub[sub["num_classes"] == num_classes].copy()

    if sub.empty:
        return

    sub = sub.sort_values(["space", "n_subjects_plot"])

    space_order = ["Unsupervised", "Trait_LDA", "State_LDA", "Joint_LDA"]
    existing_spaces = [s for s in space_order if s in sub["space"].unique()]
    remaining_spaces = [s for s in sorted(sub["space"].unique()) if s not in existing_spaces]
    spaces = existing_spaces + remaining_spaces

    n_spaces = len(spaces)
    ncols = 2
    nrows = math.ceil(n_spaces / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
        sharey=True
    )
    axes = axes.flatten()

    for ax_idx, space in enumerate(spaces):
        ax = axes[ax_idx]
        s = sub[sub["space"] == space].copy().sort_values("n_subjects_plot")

        subject_order_df = (
            s[["n_subjects_plot", "n_subjects_label"]]
            .drop_duplicates()
            .sort_values("n_subjects_plot")
        )
        subject_labels = subject_order_df["n_subjects_label"].tolist()
        x = np.arange(len(subject_labels))

        s = (
            s.set_index("n_subjects_label")
             .reindex(subject_labels)
             .reset_index()
        )

        inter_vals = s["mean_inter"].to_numpy(dtype=float)
        intra_vals = s["mean_intra"].to_numpy(dtype=float)

        bar_width = 0.35

        ax.bar(
            x - bar_width / 2,
            inter_vals,
            width=bar_width,
            label="inter",
            alpha=0.9
        )
        ax.bar(
            x + bar_width / 2,
            intra_vals,
            width=bar_width,
            label="intra",
            alpha=0.75
        )

        ax.set_title(space)
        ax.set_xticks(x)
        ax.set_xticklabels(subject_labels)
        ax.set_xlabel("Number of Subjects")
        ax.set_ylabel("Distance")

    for j in range(n_spaces, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02)
        )

    title = f"Inter vs Intra Distance — {dataset} / {feature}"
    if num_classes is not None:
        title += f" ({num_classes} classes)"
    fig.suptitle(title, y=1.04, fontsize=16)

    fig.tight_layout()

    suffix = f"_{num_classes}classes" if num_classes is not None else ""
    out_path = os.path.join(out_dir, f"{dataset}_{feature}{suffix}_inter_intra_bars.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["motor", "lee", "lemon", "gamma", "bci"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--plot_feature_level", action="store_true")
    parser.add_argument("--plot_inter_intra", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_df = collect_all_hierarchy_results(args.results_dir, args.dataset)
    raw_df = add_subject_order_info(raw_df)

    raw_df = raw_df.dropna(
        subset=["n_subjects_plot", "mean_hierarchy", "space", "dataset", "feature"]
    )

    if raw_df.empty:
        raise ValueError(f"No hierarchy data found for dataset={args.dataset}")

    summary_df = clean_summary_df(raw_df)
    save_tables(raw_df, summary_df, args.out_dir, args.dataset)

    class_values = sorted(summary_df["num_classes"].dropna().unique())

    if len(class_values) == 0:
        print("Plotting main bars...")
        plot_dataset_bars_by_space(summary_df, args.dataset, args.out_dir)
        if args.plot_feature_level:
            for feature in sorted(summary_df["feature"].dropna().unique()):
                print(f"Plotting feature bars for {feature}...")
                plot_dataset_feature_bars(summary_df, args.dataset, feature, args.out_dir)
        if args.plot_inter_intra:
            print("Plotting inter/intra bars...")
            for feature in sorted(class_summary["feature"].dropna().unique()):
                    plot_inter_intra_bars_per_feature(
                        class_summary,
                        args.dataset,
                        feature,
                        args.out_dir,
                        num_classes=num_classes
                    )
    else:
        for num_classes in class_values:
            class_summary = summary_df[summary_df["num_classes"] == num_classes].copy()
            if class_summary.empty:
                continue
            print("Plotting main bars...")
            plot_dataset_bars_by_space(class_summary, args.dataset, args.out_dir, num_classes=num_classes)

            if args.plot_feature_level:
                for feature in sorted(class_summary["feature"].dropna().unique()):
                    print(f"Plotting feature bars for {feature}...")
                    plot_dataset_feature_bars(
                        class_summary,
                        args.dataset,
                        feature,
                        args.out_dir,
                        num_classes=num_classes
                    )

            if args.plot_inter_intra:
                print("Plotting inter/intra bars...")
                for feature in sorted(class_summary["feature"].dropna().unique()):
                    plot_inter_intra_bars_per_feature(
                        class_summary,
                        args.dataset,
                        feature,
                        args.out_dir,
                        num_classes=num_classes
                    )

    print("Done.")


if __name__ == "__main__":
    main()