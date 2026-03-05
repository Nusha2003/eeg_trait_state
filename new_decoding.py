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
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA
from hierarchy_metric import HierarchyMetric

def fit_space(X, y, train_idx, space_type):
    X_train = X[train_idx]
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_comp = min(30, n_samples - 1, n_features)
    
    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp))
    ]

    if space_type != 'Raw' and y is not None:
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        n_classes = len(np.unique(y_train))
        n_lda = min(10, n_classes - 1)
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
        'F1': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]

    data_cfg = config["data"][dataset]
    feature = config["feature"]
    save_dir = os.path.join(data_cfg["save_dir"], feature)
    geom_path = os.path.join(save_dir, "geometry")
    hier_path = os.path.join(save_dir, "hierarchy")

    os.makedirs(geom_path, exist_ok=True)
    os.makedirs(hier_path, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, required=True, choices=['4', '10', '20', 'all'])
    parser.add_argument("--num_classes", type=str, required=True, choices=['2','6', '10'])
    args = parser.parse_args()

    if dataset == "motor":
        if args.num_classes == '6':
            split_dir = data_cfg["6split_dir"]
        else:
            split_dir = data_cfg["10split_dir"]
    else:
        split_dir = data_cfg["split_dir"]
    if args.subjects in ['4', '10', '20']:
        split_files = sorted([f for f in os.listdir(split_dir) if f.startswith(f"splits_n{args.subjects}_rep")])
    else:
        split_files = [f"splits_{args.subjects}.pkl"]

    all_space_results = []
    hierarchy_results = []

    for s_file in split_files:
        print(f"Processing Seed: {s_file}")
        data = joblib.load(os.path.join(split_dir, s_file))
        metadata = data['metadata']
        metadata.columns = metadata.columns.str.lower().str.strip()
        
        splits = data['splits']
        trait_split = data['trait_split'] 
        seed_id = int(s_file.split("_rep")[-1].replace(".pkl", "")) if "_rep" in s_file else 0

        
        X = pd.read_csv(data_cfg[feature]).values
        
        y_subject = metadata.subject
        y_state = metadata.condition
        y_joint = metadata.subject.astype(str) + "_" + metadata.condition.astype(str)

        for space_name in ['Raw', 'Trait_LDA', 'State_LDA', 'Joint_LDA']:
            if space_name == 'Raw':
                y_for_geom = None 
            elif space_name == 'Trait_LDA':
                y_for_geom = y_subject
            elif space_name == 'State_LDA':
                y_for_geom = y_state
            else: 
                y_for_geom = y_joint

            transformer = fit_space(X, y_for_geom, trait_split['train'], space_name)
            X_transformed = transformer.transform(X)

            test_indices = trait_split['test']
            test_labels = metadata.iloc[test_indices][['subject', 'condition']].reset_index(drop=True)
            hm = HierarchyMetric(X_transformed[test_indices], test_labels)
            h_results = hm.evaluate(n_perm=1000)

            hierarchy_results.append({
                "space": space_name, "seed": seed_id,
                "hier_ratio": h_results['ratio'], "hier_pval": h_results['p_value']
            })

            t_metrics = run_decoding_experiment(
                X_transformed[trait_split['train']], X_transformed[trait_split['test']],
                y_subject.iloc[trait_split['train']], y_subject.iloc[trait_split['test']]
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
                    'space': space_name, 'seed': seed_id, 'subject': split['subject'],
                    'trait_ident_acc': t_metrics['Bal_Acc'],
                    'within_state_acc': w_metrics['Bal_Acc'],
                    'between_acc': b_metrics['Bal_Acc'],
                    'trait_ident_f1': t_metrics['F1'],
                    'within_state_f1': w_metrics['F1'],
                    'between_f1': b_metrics['F1'],
                })


    results_df = pd.DataFrame(all_space_results)
    hier_df = pd.DataFrame(hierarchy_results)
    
    final_summary = results_df.groupby(['space', 'seed']).mean(numeric_only=True).groupby('space').agg(['mean', 'std'])
    print("\n--- Final Results ---")
    print(final_summary)
    final_summary.to_csv(os.path.join(geom_path, f"geometry_results_n{args.subjects}_{args.num_classes}classes_{feature}.csv"))
    hier_df.to_csv(
    os.path.join(hier_path, f"hierarchy_results_n{args.subjects}_{args.num_classes}_{feature}classes.csv"
    ))
