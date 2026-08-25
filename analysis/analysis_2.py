# ============================================================
# Decoding accuracy analysis
#
# Standard representations:
#   PSD / entropy / complexity
#   Trait_LDA / State_LDA / Joint_LDA
#
# Autoencoder:
#   Loaded from decoding_all_seeds.csv
#   Used as a shared baseline in every feature panel.
#
# Raw / Unsupervised is NOT plotted.
# ============================================================

from __future__ import annotations

import argparse
import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Defaults
# ============================================================

DEFAULT_RESULTS_DIR = (
    "/home1/amadapur/projects/"
    "eeg_trait_state_geometry/results_dir"
)

DEFAULT_OUT_DIR = (
    "/home1/amadapur/projects/"
    "eeg_trait_state_geometry/plots/"
    "decoding_accuracy_bars"
)


STANDARD_FEATURES = [
    "psd",
    "entropy",
    "complexity",
]

STANDARD_SPACES = [
    "Trait_LDA",
    "State_LDA",
    "Joint_LDA",
]

REPRESENTATION_ORDER = [
    "Autoencoder",
    "Trait_LDA",
    "State_LDA",
    "Joint_LDA",
]

METRICS = [
    "Bal_Acc",
    "F1",
    "Kappa",
]

SPLIT_ORDER = [
    "trait",
    "within_state",
    "between_state",
]


# Actual number of subjects represented by "all".
DATASET_ALL_MAP = {
    "gamma": 14,
    "bci": 14,
    "motor": 105,
}


# ============================================================
# Path parsing
# ============================================================

def parse_n_subjects(path: str) -> str | None:
    """
    Examples:

    geometry_results_trait_n4_4classes_psd.csv
        -> "4"

    geometry_results_trait_nall_4classes_psd.csv
        -> "all"
    """

    fname = os.path.basename(path)

    match = re.search(
        r"_n(\d+|all)(?=_|\.|$)",
        fname,
    )

    if match is None:
        return None

    return match.group(1)


def parse_num_classes(path: str) -> int | None:
    fname = os.path.basename(path)

    match = re.search(
        r"_(\d+)classes",
        fname,
    )

    if match is None:
        return None

    return int(match.group(1))


def parse_split_type(path: str) -> str | None:
    """
    Determine decoding protocol from path / filename.
    """

    path_lower = path.lower()

    if "within_state" in path_lower:
        return "within_state"

    if "between_state" in path_lower:
        return "between_state"

    if "trait" in path_lower:
        return "trait"

    return None


def infer_feature(path: str) -> str:
    """
    Infer feature family from directory path.
    """

    parts = os.path.normpath(path).split(os.sep)

    known_features = {
        "psd",
        "entropy",
        "complexity",
        "autoencoder",
    }

    for part in parts:
        part_lower = part.lower()

        if part_lower in known_features:
            return part_lower

    return "unknown_feature"


# ============================================================
# Subject-count helpers
# ============================================================

def subject_plot_value(
    dataset: str,
    n_subjects: str,
) -> float:
    """
    Numerical value used only for ordering subject counts.
    """

    if str(n_subjects).lower() == "all":
        return float(
            DATASET_ALL_MAP.get(
                dataset,
                np.inf,
            )
        )

    return float(n_subjects)


def subject_label(
    dataset: str,
    n_subjects: str,
) -> str:
    """
    Pretty x-axis label.
    """

    if str(n_subjects).lower() == "all":

        total = DATASET_ALL_MAP.get(
            dataset
        )

        if total is None:
            return "all"

        return f"all ({total})"

    return str(n_subjects)


def sort_subject_counts(
    dataset: str,
    values,
) -> list[str]:

    unique_values = list(
        dict.fromkeys(
            str(value)
            for value in values
            if pd.notna(value)
        )
    )

    return sorted(
        unique_values,
        key=lambda value: subject_plot_value(
            dataset,
            value,
        ),
    )


# ============================================================
# Standard representation results
# ============================================================

def load_standard_geometry_csv(
    csv_path: str,
    dataset: str,
) -> pd.DataFrame | None:
    """
    Load a standard geometry summary CSV.

    Expected format is something like:

        ,Bal_Acc,Bal_Acc,F1,F1,Kappa,Kappa
        ,mean,std,mean,std,mean,std
        space,,,,,,
        Raw,...
        Trait_LDA,...
        State_LDA,...
        Joint_LDA,...
    """

    try:
        df = pd.read_csv(
            csv_path,
            header=[0, 1],
            index_col=0,
        )

    except Exception as exc:

        print(
            f"Skipping {csv_path}: "
            f"could not read ({exc})"
        )

        return None

    # --------------------------------------------------------
    # Drop the extra "space" row generated when dataframe
    # index name was written to CSV.
    # --------------------------------------------------------

    index_text = (
        pd.Index(df.index)
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        index_text != "space"
    ].copy()

    # --------------------------------------------------------
    # Flatten metric/stat MultiIndex columns.
    #
    # ('Bal_Acc', 'mean') -> Bal_Acc_mean
    # ('Bal_Acc', 'std')  -> Bal_Acc_std
    # --------------------------------------------------------

    flat_columns = []

    for metric, stat in df.columns:

        metric = str(metric).strip()
        stat = str(stat).strip()

        if (
            not stat
            or stat.startswith("Unnamed")
        ):
            flat_columns.append(
                metric
            )

        else:
            flat_columns.append(
                f"{metric}_{stat}"
            )

    df.columns = flat_columns

    df.index.name = "space"

    df = (
        df
        .reset_index()
    )

    # --------------------------------------------------------
    # Metadata from path
    # --------------------------------------------------------

    feature = infer_feature(
        csv_path
    )

    split_type = parse_split_type(
        csv_path
    )

    n_subjects = parse_n_subjects(
        csv_path
    )

    num_classes = parse_num_classes(
        csv_path
    )

    if feature == "autoencoder":
        # AE is loaded separately from decoding_all_seeds.csv.
        return None

    if feature not in STANDARD_FEATURES:
        return None

    if split_type is None:
        print(
            f"Skipping {csv_path}: "
            f"could not determine split type."
        )
        return None

    if n_subjects is None:
        print(
            f"Skipping {csv_path}: "
            f"could not determine subject count."
        )
        return None

    df["dataset"] = dataset
    df["feature"] = feature
    df["split_type"] = split_type
    df["n_subjects"] = str(
        n_subjects
    )
    df["num_classes"] = num_classes
    df["source_file"] = csv_path

    # --------------------------------------------------------
    # Normalize representation names
    # --------------------------------------------------------

    df["space"] = (
        df["space"]
        .astype(str)
        .str.strip()
    )

    # Remove Raw / Unsupervised completely.
    df = df[
        ~df["space"].isin(
            [
                "Raw",
                "Unsupervised",
            ]
        )
    ].copy()

    # Only keep LDA representations we actually want.
    df = df[
        df["space"].isin(
            STANDARD_SPACES
        )
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for metric in METRICS:

        for stat in [
            "mean",
            "std",
        ]:

            column = (
                f"{metric}_{stat}"
            )

            if column in df.columns:

                df[column] = (
                    pd.to_numeric(
                        df[column],
                        errors="coerce",
                    )
                )

    return df


def collect_standard_results(
    results_dir: str,
    dataset: str,
) -> pd.DataFrame:
    """
    Only collect geometry_results_*.csv files.

    Does NOT load:
        hierarchy files
        dimension tuning files
        decoding_all_seeds.csv
    """

    dataset_dir = os.path.join(
        results_dir,
        dataset,
    )

    pattern = os.path.join(
        dataset_dir,
        "**",
        "geometry_results_*.csv",
    )

    files = glob.glob(
        pattern,
        recursive=True,
    )

    print(
        f"Found {len(files)} standard decoding "
        f"summary files for dataset={dataset}"
    )

    frames = []

    for csv_path in files:

        df = load_standard_geometry_csv(
            csv_path,
            dataset,
        )

        if (
            df is not None
            and not df.empty
        ):
            frames.append(
                df
            )

    if not frames:
        raise ValueError(
            f"No valid standard decoding summaries "
            f"found for dataset={dataset}."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# Convert standard summaries to long format
# ============================================================

def standard_results_to_long(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in df.iterrows():

        for metric in METRICS:

            mean_col = (
                f"{metric}_mean"
            )

            std_col = (
                f"{metric}_std"
            )

            if mean_col not in df.columns:
                continue

            mean_value = row.get(
                mean_col,
                np.nan,
            )

            std_value = row.get(
                std_col,
                np.nan,
            )

            rows.append({
                "dataset": row["dataset"],
                "feature": row["feature"],
                "split_type": row["split_type"],
                "n_subjects": str(
                    row["n_subjects"]
                ),
                "num_classes": row[
                    "num_classes"
                ],
                "space": row["space"],
                "metric": metric,
                "mean_value": mean_value,
                "std_value": std_value,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Autoencoder decoding results
# ============================================================

def find_autoencoder_decoding_file(
    results_dir: str,
    dataset: str,
) -> str:

    dataset_dir = os.path.join(
        results_dir,
        dataset,
    )

    pattern = os.path.join(
        dataset_dir,
        "autoencoder",
        "**",
        "decoding_all_seeds.csv",
    )

    files = glob.glob(
        pattern,
        recursive=True,
    )

    if not files:
        raise FileNotFoundError(
            "Could not find autoencoder "
            "decoding_all_seeds.csv under "
            f"{dataset_dir}/autoencoder"
        )

    if len(files) > 1:
        print(
            "Warning: multiple AE decoding files found. "
            "Using:"
        )

        for path in files:
            print(
                f"    {path}"
            )

    return files[0]


def load_autoencoder_results(
    results_dir: str,
    dataset: str,
) -> pd.DataFrame:
    """
    Load flat AE decoding results.

    Expected columns:

        num_subjects
        seed
        split_type
        subject
        space
        Bal_Acc
        F1
        Kappa
    """

    csv_path = (
        find_autoencoder_decoding_file(
            results_dir,
            dataset,
        )
    )

    print(
        f"Loading Autoencoder results: "
        f"{csv_path}"
    )

    df = pd.read_csv(
        csv_path
    )

    required = {
        "num_subjects",
        "seed",
        "split_type",
        "space",
        "Bal_Acc",
        "F1",
        "Kappa",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Autoencoder file is missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Normalize subject count
    # --------------------------------------------------------

    def normalize_n_subjects(value):

        value_string = (
            str(value)
            .strip()
            .lower()
        )

        if value_string in {
            "all",
            "nall",
        }:
            return "all"

        try:
            return str(
                int(
                    float(value)
                )
            )

        except Exception:
            return value_string

    df["n_subjects"] = (
        df["num_subjects"]
        .apply(
            normalize_n_subjects
        )
    )

    df["dataset"] = dataset

    df["split_type"] = (
        df["split_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["space"] = "Autoencoder"

    for metric in METRICS:

        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

    return df


# ============================================================
# Correct AE aggregation
# ============================================================

def summarize_autoencoder_results(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Important:

    trait:
        Usually one result per seed.

    within_state / between_state:
        Multiple target subjects can exist per seed.

    Therefore:

        1. Average subjects WITHIN each seed.
        2. Calculate mean/std ACROSS seeds.

    This prevents subjects from being treated as
    independent repetitions.
    """

    # --------------------------------------------------------
    # Step 1:
    # Average target subjects within each seed.
    #
    # For trait this is effectively unchanged.
    # --------------------------------------------------------

    seed_level = (
        df
        .groupby(
            [
                "dataset",
                "split_type",
                "n_subjects",
                "seed",
            ],
            dropna=False,
            sort=False,
        )[METRICS]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------------
    # Step 2:
    # Mean/std across seeds
    # --------------------------------------------------------

    summary = (
        seed_level
        .groupby(
            [
                "dataset",
                "split_type",
                "n_subjects",
            ],
            dropna=False,
            sort=False,
        )[METRICS]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    rows = []

    for _, row in summary.iterrows():

        for metric in METRICS:

            rows.append({
                "dataset": row[
                    ("dataset", "")
                ],
                "feature": "autoencoder",
                "split_type": row[
                    ("split_type", "")
                ],
                "n_subjects": str(
                    row[
                        ("n_subjects", "")
                    ]
                ),
                "num_classes": None,
                "space": "Autoencoder",
                "metric": metric,
                "mean_value": row[
                    (metric, "mean")
                ],
                "std_value": row[
                    (metric, "std")
                ],
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plotting
# ============================================================

def pretty_split_name(
    split_type: str,
) -> str:

    names = {
        "trait": "Trait Identification",
        "within_state": "Within-State Decoding",
        "between_state": "Between-State Decoding",
    }

    return names.get(
        split_type,
        split_type,
    )


def pretty_metric_name(
    metric: str,
) -> str:

    names = {
        "Bal_Acc": "Balanced Accuracy",
        "F1": "F1 Score",
        "Kappa": "Cohen's Kappa",
    }

    return names.get(
        metric,
        metric,
    )


def plot_decoding_bars(
    standard_df: pd.DataFrame,
    autoencoder_df: pd.DataFrame,
    dataset: str,
    split_type: str,
    metric: str,
    out_dir: str,
    num_classes: int | None = None,
) -> None:
    """
    One figure with panels:

        PSD
        Entropy
        Complexity

    In every panel:

        Autoencoder
        Trait_LDA
        State_LDA
        Joint_LDA

    x-axis:
        number of subjects

    Autoencoder is a shared baseline and is therefore
    repeated visually in each feature panel.
    """

    std = standard_df[
        (standard_df["dataset"] == dataset)
        & (
            standard_df["split_type"]
            == split_type
        )
        & (
            standard_df["metric"]
            == metric
        )
    ].copy()

    if num_classes is not None:

        std = std[
            std["num_classes"]
            == num_classes
        ].copy()

    ae = autoencoder_df[
        (autoencoder_df["dataset"] == dataset)
        & (
            autoencoder_df["split_type"]
            == split_type
        )
        & (
            autoencoder_df["metric"]
            == metric
        )
    ].copy()

    if std.empty:
        print(
            f"No standard data: "
            f"{dataset} / {split_type} / {metric}"
        )
        return

    if ae.empty:
        print(
            f"Warning: no Autoencoder data for "
            f"{dataset} / {split_type} / {metric}"
        )

    # --------------------------------------------------------
    # Available handcrafted features
    # --------------------------------------------------------

    features = [
        feature
        for feature in STANDARD_FEATURES
        if feature in std["feature"].unique()
    ]

    if not features:
        return

    # --------------------------------------------------------
    # Subject-count union
    #
    # Include counts present in either standard or AE data.
    # --------------------------------------------------------

    subject_values = list(
        std["n_subjects"]
    ) + list(
        ae["n_subjects"]
    )

    subject_counts = (
        sort_subject_counts(
            dataset,
            subject_values,
        )
    )

    subject_labels = [
        subject_label(
            dataset,
            n_subjects,
        )
        for n_subjects
        in subject_counts
    ]

    x = np.arange(
        len(subject_counts)
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    n_features = len(
        features
    )

    ncols = min(
        3,
        n_features,
    )

    nrows = math.ceil(
        n_features / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            6.3 * ncols,
            5.0 * nrows,
        ),
        squeeze=False,
        sharey=True,
    )

    axes = axes.flatten()

    total_width = 0.82

    bar_width = (
        total_width
        / len(
            REPRESENTATION_ORDER
        )
    )

    # --------------------------------------------------------
    # Each feature panel
    # --------------------------------------------------------

    for ax_idx, feature in enumerate(
        features
    ):

        ax = axes[
            ax_idx
        ]

        feature_df = std[
            std["feature"]
            == feature
        ].copy()

        for rep_idx, representation in enumerate(
            REPRESENTATION_ORDER
        ):

            # ------------------------------------------------
            # Autoencoder:
            # shared reference baseline
            # ------------------------------------------------

            if representation == "Autoencoder":

                rep_df = (
                    ae
                    .groupby(
                        "n_subjects",
                        as_index=False,
                    )
                    .agg({
                        "mean_value": "mean",
                        "std_value": "mean",
                    })
                )

            # ------------------------------------------------
            # Feature-specific LDA space
            # ------------------------------------------------

            else:

                rep_df = feature_df[
                    feature_df["space"]
                    == representation
                ].copy()

                # Safety in case duplicate files exist.
                rep_df = (
                    rep_df
                    .groupby(
                        "n_subjects",
                        as_index=False,
                    )
                    .agg({
                        "mean_value": "mean",
                        "std_value": "mean",
                    })
                )

            # ------------------------------------------------
            # Align subject counts
            # ------------------------------------------------

            value_map = {
                str(row["n_subjects"]): row[
                    "mean_value"
                ]
                for _, row
                in rep_df.iterrows()
            }

            error_map = {
                str(row["n_subjects"]): row[
                    "std_value"
                ]
                for _, row
                in rep_df.iterrows()
            }

            y = np.array(
                [
                    value_map.get(
                        str(n),
                        np.nan,
                    )
                    for n
                    in subject_counts
                ],
                dtype=float,
            )

            yerr = np.array(
                [
                    error_map.get(
                        str(n),
                        0.0,
                    )
                    for n
                    in subject_counts
                ],
                dtype=float,
            )

            yerr = np.nan_to_num(
                yerr,
                nan=0.0,
            )

            mask = ~np.isnan(
                y
            )

            offset = (
                rep_idx
                - (
                    len(
                        REPRESENTATION_ORDER
                    ) - 1
                ) / 2
            ) * bar_width

            ax.bar(
                x[mask] + offset,
                y[mask],
                width=bar_width,
                yerr=yerr[mask],
                capsize=3,
                label=representation,
                alpha=0.9,
            )

        # ----------------------------------------------------
        # Panel formatting
        # ----------------------------------------------------

        ax.set_title(
            feature.upper()
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            subject_labels
        )

        ax.set_xlabel(
            "Number of Subjects"
        )

        ax.set_ylabel(
            pretty_metric_name(
                metric
            )
        )

        if metric in {
            "Bal_Acc",
            "F1",
        }:
            ax.set_ylim(
                0,
                1.05,
            )

        elif metric == "Kappa":
            ax.set_ylim(
                -0.05,
                1.05,
            )

    # --------------------------------------------------------
    # Remove unused axes
    # --------------------------------------------------------

    for ax_idx in range(
        n_features,
        len(axes),
    ):
        fig.delaxes(
            axes[ax_idx]
        )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    if handles:

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(
                0.5,
                1.02,
            ),
        )

    split_title = (
        pretty_split_name(
            split_type
        )
    )

    metric_title = (
        pretty_metric_name(
            metric
        )
    )

    title = (
        f"{split_title}: "
        f"{metric_title} vs Number of Subjects — "
        f"{dataset.capitalize()}"
    )

    if num_classes is not None:

        title += (
            f" ({int(num_classes)} classes)"
        )

    fig.suptitle(
        title,
        fontsize=16,
        y=1.06,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    class_suffix = (
        f"_{int(num_classes)}classes"
        if num_classes is not None
        else ""
    )

    filename = (
        f"{dataset}_"
        f"{split_type}_"
        f"{metric}"
        f"{class_suffix}_"
        f"decoding_bars.png"
    )

    out_path = os.path.join(
        out_dir,
        filename,
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved {out_path}"
    )


# ============================================================
# Save combined data used for plotting
# ============================================================

def save_analysis_tables(
    standard_df: pd.DataFrame,
    autoencoder_df: pd.DataFrame,
    out_dir: str,
    dataset: str,
) -> None:

    standard_path = os.path.join(
        out_dir,
        f"{dataset}_standard_decoding_summary.csv",
    )

    ae_path = os.path.join(
        out_dir,
        f"{dataset}_autoencoder_decoding_summary.csv",
    )

    standard_df.to_csv(
        standard_path,
        index=False,
    )

    autoencoder_df.to_csv(
        ae_path,
        index=False,
    )

    print(
        f"Saved {standard_path}"
    )

    print(
        f"Saved {ae_path}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Dataset to analyze, "
            "for example: gamma"
        ),
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=[
            "all",
            "Bal_Acc",
            "F1",
            "Kappa",
        ],
    )

    parser.add_argument(
        "--split_type",
        type=str,
        default="all",
        choices=[
            "all",
            "trait",
            "within_state",
            "between_state",
        ],
    )

    args = parser.parse_args()

    dataset = (
        args.dataset
        .strip()
        .lower()
    )

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Standard feature decoding summaries
    # --------------------------------------------------------

    standard_wide = (
        collect_standard_results(
            results_dir=args.results_dir,
            dataset=dataset,
        )
    )

    standard_long = (
        standard_results_to_long(
            standard_wide
        )
    )

    # --------------------------------------------------------
    # Autoencoder seed-level results
    # --------------------------------------------------------

    autoencoder_raw = (
        load_autoencoder_results(
            results_dir=args.results_dir,
            dataset=dataset,
        )
    )

    autoencoder_summary = (
        summarize_autoencoder_results(
            autoencoder_raw
        )
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()
    print(
        "Standard features:",
        sorted(
            standard_long[
                "feature"
            ]
            .dropna()
            .unique()
        ),
    )

    print(
        "Standard spaces:",
        sorted(
            standard_long[
                "space"
            ]
            .dropna()
            .unique()
        ),
    )

    print(
        "AE split types:",
        sorted(
            autoencoder_summary[
                "split_type"
            ]
            .dropna()
            .unique()
        ),
    )

    print(
        "AE subject counts:",
        sort_subject_counts(
            dataset,
            autoencoder_summary[
                "n_subjects"
            ].dropna(),
        ),
    )

    # --------------------------------------------------------
    # Save processed tables
    # --------------------------------------------------------

    save_analysis_tables(
        standard_df=standard_long,
        autoencoder_df=autoencoder_summary,
        out_dir=args.out_dir,
        dataset=dataset,
    )

    # --------------------------------------------------------
    # Determine class configurations
    # --------------------------------------------------------

    class_values = sorted(
        standard_long[
            "num_classes"
        ]
        .dropna()
        .unique()
    )

    if not class_values:
        class_values = [
            None
        ]

    # --------------------------------------------------------
    # Which plots?
    # --------------------------------------------------------

    if args.metric == "all":
        metrics = METRICS
    else:
        metrics = [
            args.metric
        ]

    if args.split_type == "all":
        split_types = SPLIT_ORDER
    else:
        split_types = [
            args.split_type
        ]

    # --------------------------------------------------------
    # Generate bar charts
    # --------------------------------------------------------

    for num_classes in class_values:

        for split_type in split_types:

            for metric in metrics:

                plot_decoding_bars(
                    standard_df=standard_long,
                    autoencoder_df=autoencoder_summary,
                    dataset=dataset,
                    split_type=split_type,
                    metric=metric,
                    out_dir=args.out_dir,
                    num_classes=num_classes,
                )

    print()
    print("Done.")


if __name__ == "__main__":
    main()