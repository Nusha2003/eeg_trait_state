import numpy as np
import pandas as pd
import joblib
import argparse
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yaml
from hierarchy_metric import HierarchyMetric


def build_representation_space(X, y, train_idx, test_idx, space_type):
    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=10)
    pca.fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)


    #LDA 
    if space_type == 'Raw':
        return X_train_pca, X_test_pca
    n_classes = len(np.unique(y_train))
    n_lda = min(10, n_classes - 1)

    if n_lda < 1:
        return X_train_pca, X_test_pca


    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda.fit(X_train_pca, y_train)

    X_train_lda = lda.transform(X_train_pca)
    X_test_lda  = lda.transform(X_test_pca)

    return X_train_lda, X_test_lda
    
"""
def lda_full_projection(X, y, max_components=9):
    n_classes = len(np.unique(y))
    k = min(max_components, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=k, solver="svd")
    return lda.fit_transform(X, y)


"""

def run_decoding_experiment(X_train, X_test, y_train, y_test):
    clf = LinearSVC(dual=False, max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        'Bal_Acc': balanced_accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }

"""
def run_decoding_experiment(X, y, train_idx, test_idx, is_already_lda=False):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    n_classes_train = len(np.unique(y_train))
    if n_classes_train < 2:
        return {'Bal_Acc': np.nan, 'F1': np.nan}
    
    steps = [('scaler', StandardScaler())]
    
    if not is_already_lda:
        n_features = X.shape[1]
        n_comp = min(10, n_features, n_classes_train - 1) 
        if n_comp >= 1:
            steps.append(('lda', LinearDiscriminantAnalysis(n_components=n_comp)))
        
    steps.append(('clf', LinearSVC(dual=False, max_iter=2000, random_state=42)))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return {
        'Bal_Acc': balanced_accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
"""
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, required=True, choices=['4', '10', '20', '109'])
    parser.add_argument("--num_classes", type=str, required=True, choices=['6', '10'])
    args = parser.parse_args()

    split_dir = f"splits_{args.num_classes}classes"
    if args.subjects in ['4', '10', '20']:
        split_files = sorted([
            f for f in os.listdir(split_dir)
            if f.startswith(f"splits_n{args.subjects}_rep")
        ])
    else:
        split_files = [f"splits_n{args.subjects}.pkl"]

    all_space_results = []
    hierarchy_results = []
    for s_file in split_files:
        print(f"Processing split file: {s_file}")
        data = joblib.load(os.path.join(split_dir, s_file))
        metadata = data['metadata']
        splits = data['splits']
        trait_split = data['trait_split'] 
        if "_rep" in s_file:
            seed_id = int(s_file.split("_rep")[-1].replace(".pkl", ""))
        else:
            seed_id = 0

        metadata.columns = metadata.columns.str.lower().str.strip()
        X = pd.read_csv(config['data']['feature_path']).values
        if args.num_classes == '6':
            y = pd.read_csv(config['data']['labels_6_path']).values
        else:
            y = pd.read_csv(config['data']['labels_10_path']).values
    
        for space_name in ['Raw', 'Trait_LDA', 'State_LDA', 'Joint_LDA']:
            X_train, X_test = build_representation_space(
                X,
                metadata.subject if space_name == 'Trait_LDA'
                else metadata.condition if space_name == 'State_LDA'
                else metadata.subject + "_" + metadata.condition if space_name == 'Joint_LDA'
                else metadata.subject,  
                trait_split['train'],
                trait_split['test'],
                space_name
            )

            y_train = metadata.subject.iloc[trait_split['train']]
            y_test  = metadata.subject.iloc[trait_split['test']]


            test_labels = metadata.iloc[trait_split['test']][['subject', 'condition']].reset_index(drop=True)

            hm = HierarchyMetric(X_test, test_labels)

            h_results = hm.evaluate(n_perm=1000)

            hier_ratio = h_results['ratio']
            hier_pval = h_results['p_value']
            hierarchy_results.append({
            "space": space_name,
            "seed": seed_id,
            "hier_ratio": hier_ratio,
            "hier_pval": hier_pval
        })

            t_metrics = run_decoding_experiment(X_train, X_test, y_train, y_test)

            for split in splits:

                # Within
                X_train, X_test = build_representation_space(
                    X,
                    metadata.condition if space_name == 'State_LDA'
                    else metadata.subject if space_name == 'Trait_LDA'
                    else metadata.subject + "_" + metadata.condition if space_name == 'Joint_LDA'
                    else metadata.condition,
                    split['within_state']['train'],
                    split['within_state']['test'],
                    space_name
                )

                y_train = metadata.condition.iloc[split['within_state']['train']]
                y_test  = metadata.condition.iloc[split['within_state']['test']]

                w_metrics = run_decoding_experiment(X_train, X_test, y_train, y_test)

                # Between
                X_train, X_test = build_representation_space(
                    X,
                    metadata.condition if space_name == 'State_LDA'
                    else metadata.subject if space_name == 'Trait_LDA'
                    else metadata.subject + "_" + metadata.condition if space_name == 'Joint_LDA'
                    else metadata.condition,
                    split['between_state']['train'],
                    split['between_state']['test'],
                    space_name
                )

                y_train = metadata.condition.iloc[split['between_state']['train']]
                y_test  = metadata.condition.iloc[split['between_state']['test']]

                b_metrics = run_decoding_experiment(X_train, X_test, y_train, y_test)

                all_space_results.append({
                    'space': space_name,
                    'seed': seed_id,
                    'subject': split['subject'],
                    'trait_ident_acc': t_metrics['Bal_Acc'],
                    'within_state_acc': w_metrics['Bal_Acc'],
                    'between_acc': b_metrics['Bal_Acc'],
                    'trait_ident_f1': t_metrics['F1'],
                    'within_state_f1': w_metrics['F1'],
                    'between_f1': b_metrics['F1'],
                })
    results_df = pd.DataFrame(all_space_results)
    hierarchy_results = pd.DataFrame(hierarchy_results)

    hier_seed_level = (
    hierarchy_results
    .groupby(['space', 'seed'])
    .mean(numeric_only=True)
    .reset_index()
)


    seed_level = (
        results_df
        .groupby(['space', 'seed'])
        .mean(numeric_only=True)
        .reset_index()
    )

    # Step 2: compute mean + std across seeds
    final_summary = (
        seed_level
        .groupby('space')
        .agg(['mean', 'std'])
    )

    hier_final_summary = (
    hier_seed_level
    .groupby('space')
    .agg(['mean','std'])
)

    print("\n--- Final Aggregated Results (Mean ± Std across seeds) ---")
    print(final_summary)
    print("\n--- Hierarchy Results (Mean ± Std across seeds) ---")
    print(hier_final_summary)       
    final_summary.to_csv(f"geometry_results_n{args.subjects}_{args.num_classes}classes.csv")
    hier_final_summary.to_csv(
    f"hierarchy_results_n{args.subjects}_{args.num_classes}classes.csv"
)
