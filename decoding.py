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
from pca_embed import perform_pca

def lda_full_projection(X, y, max_components=9):
    n_classes = len(np.unique(y))
    k = min(max_components, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=k, solver="svd")
    return lda.fit_transform(X, y)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, required=True, choices=['4', '10', '20', '109'])
    args = parser.parse_args()

    split_dir = "splits"
    if args.subjects == '10':
        split_files = [f for f in os.listdir(split_dir) if f.startswith(f"splits_n{args.subjects}_rep")]
    else:
        split_files = [f"splits_n{args.subjects}.pkl"]

    all_space_results = []

    for s_file in split_files:
        print(f"Processing split file: {s_file}")
        data = joblib.load(os.path.join(split_dir, s_file))
        metadata = data['metadata']
        splits = data['splits']
        trait_split = data['trait_split'] 

        metadata.columns = metadata.columns.str.lower().str.strip()

        _, Xz_full, _, _, _ = perform_pca() 
        Xz = Xz_full[metadata.index] 

        # create representation spaces
        Z_trait = lda_full_projection(Xz, metadata.subject, max_components=10)
        Z_state = lda_full_projection(Xz, metadata.condition, max_components=10)
        Z_joint = lda_full_projection(Xz, metadata.subject + "_" + metadata.condition, max_components=10)
        
        spaces = {'Raw': Xz, 'Trait_LDA': Z_trait, 'State_LDA': Z_state, 'Joint_LDA': Z_joint}

        for space_name, X_space in spaces.items():
            is_lda = (space_name != 'Raw')
            
            # between-subject trait
            t_metrics = run_decoding_experiment(X_space, metadata.subject, 
                                               trait_split['train'], trait_split['test'], 
                                               is_already_lda=is_lda)

            for split in splits:
                # within subject state
                w_metrics = run_decoding_experiment(X_space, metadata.condition, 
                                                   split['within_state']['train'], split['within_state']['test'], 
                                                   is_already_lda=is_lda)
                
                # generalization - between subject state
                bp_metrics = run_decoding_experiment(X_space, metadata.condition, 
                                                    split['between_state_pure']['train'], split['between_state_pure']['test'], 
                                                    is_already_lda=is_lda)
                
                # personalization - between subject state
                bnp_metrics = run_decoding_experiment(X_space, metadata.condition, 
                                                     split['between_state_nonpure']['train'], split['between_state_nonpure']['test'], 
                                                     is_already_lda=is_lda)
                
                all_space_results.append({
                    'space': space_name,
                    'subject': split['subject'],

                    # Accuracy
                    'trait_ident_acc': t_metrics['Bal_Acc'],
                    'within_state_acc': w_metrics['Bal_Acc'],
                    'between_pure_acc': bp_metrics['Bal_Acc'],
                    'between_nonpure_acc': bnp_metrics['Bal_Acc'],

                    # F1
                    'trait_ident_f1': t_metrics['F1'],
                    'within_state_f1': w_metrics['F1'],
                    'between_pure_f1': bp_metrics['F1'],
                    'between_nonpure_f1': bnp_metrics['F1']
                })

    results_df = pd.DataFrame(all_space_results)
    final_summary = results_df.groupby('space')[['trait_ident_acc', 'within_state_acc', 'between_pure_acc', 'between_nonpure_acc']].mean()
    
    print("\n--- Final Aggregated Results ---")
    print(final_summary)
    final_summary.to_csv(f"geometry_results_n{args.subjects}.csv")