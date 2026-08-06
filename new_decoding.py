import numpy as np
import pandas as pd
import joblib
import argparse
import os
import yaml
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
from sklearn.decomposition import PCA
from hierarchy_metric import HierarchyMetric
from joblib import Parallel, delayed
import re
from sklearn.model_selection import StratifiedKFold
from kneed import KneeLocator
import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def fit_space(X, y, train_idx, space_type, pca_dim=30, lda_dim=10):
    X_train = X[train_idx]
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_comp = min(pca_dim, n_samples - 1, n_features)

    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp)) #need to also decide this
    ]

    if space_type != 'Raw' and y is not None:
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        n_classes = len(np.unique(y_train))
        n_lda = min(lda_dim, n_classes - 1, n_comp) #need to change this part
        if n_lda >= 1:
            steps.append(('lda', LinearDiscriminantAnalysis(n_components=n_lda)))

    pipe = Pipeline(steps)
    if space_type == 'Raw':
        pipe.fit(X_train)
    else:
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        pipe.fit(X_train, y_train)
    return pipe


def run_decoding_experiment(X_train, X_test, y_train, y_test):
    clf = LinearSVC(dual=False, max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        'Bal_Acc': balanced_accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'Kappa': cohen_kappa_score(y_test, y_pred)
    }

"""
helper functions for finding the optimal PCA and LDA dimensions
"""

def tune_joint_dims_for_split(X, y_joint, train_idx, pca_range, lda_range,nfold=5,n_repeat=1):
    X_train_outer = X[train_idx]
    y_train_outer = y_joint.iloc[train_idx].values
    rows = []
    for pca_dim in pca_range:
        for lda_dim in lda_range:
            scores = []

            for rep in range(n_repeat):
                cv = StratifiedKFold(
                    n_splits=nfold,
                    shuffle=True,
                    random_state=42 + rep
                )

                for inner_train_idx, val_idx in cv.split(X_train_outer, y_train_outer):
                    X_inner = X_train_outer[inner_train_idx]
                    X_val = X_train_outer[val_idx]

                    y_inner = y_train_outer[inner_train_idx]
                    y_val = y_train_outer[val_idx]

                    n_samples, n_features = X_inner.shape
                    n_classes = len(np.unique(y_inner))

                    actual_pca = min(pca_dim, n_samples - 1, n_features)
                    actual_lda = min(lda_dim, actual_pca, n_classes - 1)

                    if actual_lda < 1:
                        continue

                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=actual_pca)),
                        ('lda', LinearDiscriminantAnalysis(n_components=actual_lda))
                    ])

                    pipe.fit(X_inner, y_inner)

                    X_inner_t = pipe.transform(X_inner)
                    X_val_t = pipe.transform(X_val)

                    clf = LinearSVC(dual=False, max_iter=5000, random_state=42)
                    clf.fit(X_inner_t, y_inner)

                    y_pred = clf.predict(X_val_t)
                    scores.append(balanced_accuracy_score(y_val, y_pred))

            if scores:
                rows.append({
                    "pca_dim": pca_dim,
                    "lda_dim": lda_dim,
                    "val_bal_acc": np.mean(scores)
                })

    return pd.DataFrame(rows)


def select_joint_dims(results_df):
    auc_rows = []

    for pca_dim, df in results_df.groupby("pca_dim"):
        df = df.sort_values("lda_dim")

        auc = np.trapezoid(
            df["val_bal_acc"].values,
            df["lda_dim"].values
        )

        auc_rows.append({
            "pca_dim": pca_dim,
            "accuracy_auc": auc
        })

    auc_df = pd.DataFrame(auc_rows).sort_values("pca_dim")

    pca_knee = KneeLocator(
        auc_df["pca_dim"].values,
        auc_df["accuracy_auc"].values,
        curve="concave",
        direction="increasing"
    ).knee

    if pca_knee is None:
        pca_knee = auc_df.loc[auc_df["accuracy_auc"].idxmax(), "pca_dim"]

    pca_knee = int(pca_knee)

    lda_curve = (
        results_df[results_df["pca_dim"] == pca_knee]
        .groupby("lda_dim")["val_bal_acc"]
        .mean()
        .reset_index()
        .sort_values("lda_dim")
    )

    lda_knee = KneeLocator(
        lda_curve["lda_dim"].values,
        lda_curve["val_bal_acc"].values,
        curve="concave",
        direction="increasing"
    ).knee

    if lda_knee is None:
        lda_knee = lda_curve.loc[lda_curve["val_bal_acc"].idxmax(), "lda_dim"]

    return int(pca_knee), int(lda_knee), auc_df


def tune_joint_dims_for_group(files, split_dir, X, subj_key, tune_dir, num_classes, feature):
    print(f"Tuning Joint_LDA dims for n_subjects = {subj_key}")

    pca_range = range(5, 31, 10)
    lda_range = range(1, 21)

    all_results = []

    for s_file in sorted(files):
        data = joblib.load(os.path.join(split_dir, s_file))
        metadata = data["metadata"]
        metadata.columns = metadata.columns.str.lower().str.strip()

        trait_split = data["trait_split"]

        y_joint = (
            metadata.subject.astype(str)
            + "_"
            + metadata.condition.astype(str)
        )

        df = tune_joint_dims_for_split(
            X,
            y_joint,
            trait_split["train"],
            pca_range,
            lda_range,
            nfold=3,
            n_repeat=1
        )

        seed_id = int(s_file.split("_rep")[-1].replace(".pkl", "")) if "_rep" in s_file else 0
        df["seed"] = seed_id
        df["num_subjects"] = subj_key
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)

    avg_results = (
        results_df
        .groupby(["pca_dim", "lda_dim"])["val_bal_acc"]
        .mean()
        .reset_index()
    )

    best_pca, best_lda, auc_df = select_joint_dims(avg_results)

    os.makedirs(tune_dir, exist_ok=True)

    results_df.to_csv(
        os.path.join(tune_dir, f"joint_tuning_allseeds_n{subj_key}_{num_classes}classes_{feature}.csv"),
        index=False
    )

    avg_results.to_csv(
        os.path.join(tune_dir, f"joint_tuning_avg_n{subj_key}_{num_classes}classes_{feature}.csv"),
        index=False
    )

    auc_df.to_csv(
        os.path.join(tune_dir, f"joint_tuning_auc_n{subj_key}_{num_classes}classes_{feature}.csv"),
        index=False
    )

    print(f"Selected Joint_LDA dims for n={subj_key}: PCA={best_pca}, LDA={best_lda}")

    return best_pca, best_lda


def process_split_file(s_file, split_dir, X, subj_key, joint_pca_dim=30, joint_lda_dim=10):
    print(f"Processing Seed: {s_file}")

    data = joblib.load(os.path.join(split_dir, s_file))
    metadata = data['metadata']
    metadata.columns = metadata.columns.str.lower().str.strip()

    splits = data['splits']
    trait_split = data['trait_split']

    seed_id = int(s_file.split("_rep")[-1].replace(".pkl", "")) if "_rep" in s_file else 0

    y_subject = metadata.subject
    y_state = metadata.condition
    y_joint = metadata.subject.astype(str) + "_" + metadata.condition.astype(str)

    all_space_results = []
    hierarchy_results = []

    for space_name in ['Raw', 'Trait_LDA', 'State_LDA', 'Joint_LDA']:
        if space_name == 'Raw':
            y_for_geom = None
        elif space_name == 'Trait_LDA':
            y_for_geom = y_subject
        elif space_name == 'State_LDA':
            y_for_geom = y_state
        else:
            y_for_geom = y_joint

        if space_name == "Joint_LDA":
            transformer = fit_space(
                X,
                y_for_geom,
                trait_split['train'],
                space_name,
                pca_dim=joint_pca_dim,
                lda_dim=joint_lda_dim
            )
        else:
            transformer = fit_space(
                X,
                y_for_geom,
                trait_split['train'],
                space_name
    )
        X_transformed = transformer.transform(X)

        test_indices = trait_split['test']
        test_labels = metadata.iloc[test_indices][['subject', 'condition']].reset_index(drop=True)

        hm = HierarchyMetric(X_transformed[test_indices], test_labels)
        h_results = hm.evaluate(n_perm=100)

        hierarchy_results.append({
            "num_subjects": subj_key,
            "space": space_name,
            "seed": seed_id,
            "hier_ratio": h_results['ratio'],
            "inter": h_results["inter"],
            "intra": h_results["intra"],
            "hier_pval": h_results['p_value']
        })

        t_metrics = run_decoding_experiment(
            X_transformed[trait_split['train']],
            X_transformed[trait_split['test']],
            y_subject.iloc[trait_split['train']],
            y_subject.iloc[trait_split['test']]
        )

        for split in splits:
            w_metrics = run_decoding_experiment(
                X_transformed[split['within_state']['train']],
                X_transformed[split['within_state']['test']],
                y_state.iloc[split['within_state']['train']],
                y_state.iloc[split['within_state']['test']]
            )

            b_metrics = run_decoding_experiment(
                X_transformed[split['between_state']['train']],
                X_transformed[split['between_state']['test']],
                y_state.iloc[split['between_state']['train']],
                y_state.iloc[split['between_state']['test']]
            )

            all_space_results.append({
                "num_subjects": subj_key,
                "space": space_name,
                "seed": seed_id,
                "subject": split['subject'],
                "trait_ident_acc": t_metrics['Bal_Acc'],
                "within_state_acc": w_metrics['Bal_Acc'],
                "between_acc": b_metrics['Bal_Acc'],
                "trait_ident_f1": t_metrics['F1'],
                "within_state_f1": w_metrics['F1'],
                "between_f1": b_metrics['F1'],
                "kappa_trait": t_metrics["Kappa"],
                "kappa_within": w_metrics["Kappa"],
                "kappa_between": b_metrics["Kappa"]


            })

    return all_space_results, hierarchy_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
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
    geom_path = os.path.join(save_dir, "geometry")
    hier_path = os.path.join(save_dir, "hierarchy")
    tune_path = os.path.join(save_dir, "dimension_tuning")
    os.makedirs(geom_path, exist_ok=True)
    os.makedirs(hier_path, exist_ok=True)

    def parse_subjects(fname):
        if "all" in fname:
            return "all"
        m = re.search(r'n(\d+)', fname)
        return m.group(1) if m else None

    split_files = [f for f in os.listdir(split_dir) if f.startswith("splits")]

    groups = {}
    for f in split_files:
        subj = parse_subjects(f)
        if subj is None:
            continue
        groups.setdefault(subj, []).append(f)

    print("Detected subject groups:", groups.keys())
    X = pd.read_csv(data_cfg[feature]).values

    for subj_key, files in groups.items():
        print(f"Processing n_subjects = {subj_key}")
        joint_pca_dim, joint_lda_dim = tune_joint_dims_for_group(
                files,
                split_dir,
                X,
                subj_key,
                tune_path,
                num_classes,
                feature
            )
        parallel_results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(process_split_file)(
                s_file,
                split_dir,
                X,
                subj_key,
                joint_pca_dim,
                joint_lda_dim
            )
            for s_file in sorted(files)
        )

        all_space_results = []
        hierarchy_results = []

        for split_result, hier_result in parallel_results:
            all_space_results.extend(split_result)
            hierarchy_results.extend(hier_result)

        results_df = pd.DataFrame(all_space_results)
        hier_df = pd.DataFrame(hierarchy_results)

        geom_summary = (
            results_df
            .groupby(['num_subjects', 'space', 'seed'])
            .mean(numeric_only=True)
            .groupby(['num_subjects', 'space'])
            .agg(['mean', 'std'])
        )

        hier_summary = (
            hier_df
            .groupby(['num_subjects', 'space'])
            .agg({
                'hier_ratio': ['mean', 'std'],
                'inter': ['mean', 'std'],
                'intra': ['mean', 'std'],
                'hier_pval': ['mean', 'std']
            })
        )

        geom_summary.to_csv(
            os.path.join(geom_path, f"geometry_results_n{subj_key}_{num_classes}classes_{feature}.csv")
        )

        hier_summary.to_csv(
            os.path.join(hier_path, f"hierarchy_results_n{subj_key}_{num_classes}classes_{feature}.csv")
        )