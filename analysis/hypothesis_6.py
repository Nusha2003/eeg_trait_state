import os
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ============================================================
# Hierarchy ratio vs Cohen's kappa
# Includes:
#   - motor / 10_classes only
#   - bci
#   - gamma
# Excludes:
#   - lemon
#   - lee
#   - motor / 6_classes
# ============================================================

RESULTS_ROOT = "results"

DATASET_ROOTS = {
    "motor": os.path.join(RESULTS_ROOT, "motor", "10_classes"),
    "bci": os.path.join(RESULTS_ROOT, "bci"),
    "gamma": os.path.join(RESULTS_ROOT, "gamma"),
}

FEATURES = ["complexity", "entropy", "psd"]

SPACES_ORDER = ["Raw", "Joint_LDA", "Trait_LDA", "State_LDA"]

KAPPA_METRICS = [
    "kappa_trait",
    "kappa_within",
    "kappa_between",
]

OUT_DIR = os.path.join(RESULTS_ROOT, "hierarchy_kappa_plots")
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Helpers
# -------------------------

def clean_cols(path):
    df = pd.read_csv(path, header=[0, 1, 2])

    cols = []
    for col in df.columns:
        parts = [str(x).strip() for x in col if "Unnamed" not in str(x)]
        name = "_".join(parts).strip("_")
        cols.append(name)

    df.columns = cols

    # first two columns are always num_subjects and space in your files
    df = df.rename(columns={
        df.columns[0]: "num_subjects",
        df.columns[1]: "space",
    })

    # remove possible fully-empty rows
    df = df.dropna(subset=["num_subjects", "space"])

    df["num_subjects"] = df["num_subjects"].astype(str)
    df["space"] = df["space"].astype(str)

    return df


def extract_n(path):
    match = re.search(r"_n(.*?)_", os.path.basename(path))
    if match is None:
        return "unknown"
    return match.group(1)


def find_matching_geometry_file(geom_dir, n, feature):
    candidates = glob.glob(
        os.path.join(geom_dir, f"geometry_results_n{n}_*_{feature}.csv")
    )

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print(f"Warning: multiple geometry files for n={n}, feature={feature}")
        print(candidates)

    return candidates[0]


# -------------------------
# Load data
# -------------------------

rows = []

for dataset, dataset_root in DATASET_ROOTS.items():
    for feature in FEATURES:

        feature_root = os.path.join(dataset_root, feature)
        hier_dir = os.path.join(feature_root, "hierarchy")
        geom_dir = os.path.join(feature_root, "geometry")

        if not os.path.exists(hier_dir):
            print(f"Skipping missing hierarchy dir: {hier_dir}")
            continue

        if not os.path.exists(geom_dir):
            print(f"Skipping missing geometry dir: {geom_dir}")
            continue

        hier_files = sorted(
            glob.glob(os.path.join(hier_dir, f"hierarchy_results_n*_*_{feature}.csv"))
        )

        if len(hier_files) == 0:
            print(f"No hierarchy files found in {hier_dir}")
            continue

        for hpath in hier_files:
            n = extract_n(hpath)
            gpath = find_matching_geometry_file(geom_dir, n, feature)

            if gpath is None:
                print(f"Missing geometry file for: dataset={dataset}, feature={feature}, n={n}")
                continue

            hier = clean_cols(hpath)
            geom = clean_cols(gpath)

            merged = pd.merge(
                hier,
                geom,
                on=["num_subjects", "space"],
                how="inner",
                suffixes=("_hier", "_geom")
            )

            merged["dataset"] = dataset
            merged["feature"] = feature
            merged["n"] = n

            rows.append(merged)

if len(rows) == 0:
    raise RuntimeError("No data loaded. Check RESULTS_ROOT and folder structure.")

df = pd.concat(rows, ignore_index=True)


# -------------------------
# Clean / order categories
# -------------------------

df = df[df["space"].isin(SPACES_ORDER)].copy()

df["space"] = pd.Categorical(
    df["space"],
    categories=SPACES_ORDER,
    ordered=True
)

df["feature"] = pd.Categorical(
    df["feature"],
    categories=FEATURES,
    ordered=True
)

df["dataset"] = pd.Categorical(
    df["dataset"],
    categories=list(DATASET_ROOTS.keys()),
    ordered=True
)

df = df.sort_values(["dataset", "feature", "n", "space"]).reset_index(drop=True)

print("\nLoaded points:")
print(df.groupby(["dataset", "feature", "space"]).size())


# -------------------------
# Save merged dataframe
# -------------------------

merged_path = os.path.join(OUT_DIR, "merged_hierarchy_geometry_results.csv")
df.to_csv(merged_path, index=False)
print(f"\nSaved merged data to: {merged_path}")


# -------------------------
# Plot 1: pooled across datasets/features
# -------------------------

sns.set(style="whitegrid", context="talk")

for kappa in KAPPA_METRICS:
    y = f"{kappa}_mean"

    if y not in df.columns:
        print(f"Skipping {kappa}: column not found: {y}")
        continue

    g = sns.lmplot(
        data=df,
        x="hier_ratio_mean",
        y=y,
        hue="dataset",
        col="feature",
        height=4.2,
        aspect=1.1,
        scatter_kws={"s": 85, "alpha": 0.85},
        line_kws={"linewidth": 2}
    )

    g.set_axis_labels("Hierarchy ratio", kappa.replace("_", " ").title())
    g.fig.suptitle(
        f"Hierarchy Ratio vs {kappa.replace('_', ' ').title()}",
        y=1.05
    )

    out_path = os.path.join(OUT_DIR, f"hierarchy_vs_{kappa}_pooled_by_feature.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")


# -------------------------
# Plot 2: facet by dataset
# -------------------------

for kappa in KAPPA_METRICS:
    y = f"{kappa}_mean"

    if y not in df.columns:
        continue

    g = sns.lmplot(
        data=df,
        x="hier_ratio_mean",
        y=y,
        hue="space",
        col="dataset",
        row="feature",
        height=3.6,
        aspect=1.1,
        scatter_kws={"s": 75, "alpha": 0.85},
        line_kws={"linewidth": 2}
    )

    g.set_axis_labels("Hierarchy ratio", kappa.replace("_", " ").title())
    g.fig.suptitle(
        f"Within-Dataset Hierarchy Ratio vs {kappa.replace('_', ' ').title()}",
        y=1.02
    )

    out_path = os.path.join(OUT_DIR, f"hierarchy_vs_{kappa}_by_dataset_and_feature.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")


# -------------------------
# Correlations
# -------------------------

corr_rows = []

for kappa in KAPPA_METRICS:
    y = f"{kappa}_mean"

    if y not in df.columns:
        continue

    # pooled
    sub = df[["hier_ratio_mean", y]].dropna()

    if len(sub) >= 3:
        pr, pp = pearsonr(sub["hier_ratio_mean"], sub[y])
        sr, sp = spearmanr(sub["hier_ratio_mean"], sub[y])

        corr_rows.append({
            "group": "pooled",
            "dataset": "all",
            "feature": "all",
            "kappa_metric": kappa,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n_points": len(sub),
        })

    # by dataset
    for dataset, group in df.groupby("dataset"):
        sub = group[["hier_ratio_mean", y]].dropna()

        if len(sub) < 3:
            continue

        pr, pp = pearsonr(sub["hier_ratio_mean"], sub[y])
        sr, sp = spearmanr(sub["hier_ratio_mean"], sub[y])

        corr_rows.append({
            "group": "by_dataset",
            "dataset": dataset,
            "feature": "all",
            "kappa_metric": kappa,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n_points": len(sub),
        })

    # by dataset and feature
    for (dataset, feature), group in df.groupby(["dataset", "feature"]):
        sub = group[["hier_ratio_mean", y]].dropna()

        if len(sub) < 3:
            continue

        pr, pp = pearsonr(sub["hier_ratio_mean"], sub[y])
        sr, sp = spearmanr(sub["hier_ratio_mean"], sub[y])

        corr_rows.append({
            "group": "by_dataset_feature",
            "dataset": dataset,
            "feature": feature,
            "kappa_metric": kappa,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n_points": len(sub),
        })

corr_df = pd.DataFrame(corr_rows)

corr_path = os.path.join(OUT_DIR, "hierarchy_kappa_correlations.csv")
corr_df.to_csv(corr_path, index=False)

print(f"\nSaved correlations to: {corr_path}")
print(corr_df)