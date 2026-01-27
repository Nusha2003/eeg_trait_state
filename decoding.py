import numpy as np
import pandas as pd
import joblib
import argparse
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from pca_embed import perform_pca
from sklearn.utils.multiclass import unique_labels


def lda_full_projection(X, y, max_components=10):
    n_classes = len(np.unique(y))
    k = min(max_components, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=k, solver="svd")
    return lda.fit_transform(X, y)

def run_stress_decoding(X, y, train_idx, test_idx):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    all_labels = np.unique(y)
    n_classes_train = len(np.unique(y_train))
    n_features = X.shape[1]
    
    # we take the minimum of 10, the features, or (classes - 1)
    n_comp = min(10, n_features, n_classes_train - 1) 
    
    if n_comp < 1:
        n_comp = None 
        
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(n_components=n_comp)),
        ('clf', LinearSVC(dual=False, max_iter=2000, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return {
        'Acc': accuracy_score(y_test, y_pred),
        'Bal_Acc': balanced_accuracy_score(y_test, y_pred),
        'F1_Macro': f1_score(y_test, y_pred, labels=all_labels, average='macro', zero_division=0)

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

        _, Xz_full, _, _, _ = perform_pca() 
        Xz = Xz_full[metadata.index] 

        #create representation spaces
        Z_trait = lda_full_projection(Xz, metadata.subject, max_components=10)
        Z_state = lda_full_projection(Xz, metadata.condition, max_components=10)
        joint_labels = metadata.subject + "_" + metadata.condition
        Z_joint = lda_full_projection(Xz, joint_labels, max_components=10)
        
        spaces = {'Raw': Xz, 'Trait_LDA': Z_trait, 'State_LDA': Z_state, 'Joint_LDA': Z_joint}

        for space_name, X_space in spaces.items():
            for split in splits:
                # within test
                w_metrics = run_stress_decoding(X_space, metadata.condition, 
                                                split['within']['train'], split['within']['test'])
                # between test
                b_metrics = run_stress_decoding(X_space, metadata.condition, 
                                                split['between']['train'], split['between']['test'])
                
                all_space_results.append({
                    'space': space_name,
                    'split_file': s_file,
                    'subject': split['subject'],
                    'within_bal_acc': w_metrics['Bal_Acc'],
                    'within_f1': w_metrics['F1_Macro'],
                    'between_bal_acc': b_metrics['Bal_Acc'],
                    'between_f1': b_metrics['F1_Macro']
                })

    # aggregation of final results
    results_df = pd.DataFrame(all_space_results)
    final_summary = results_df.groupby('space')[['within_bal_acc', 'between_bal_acc', 'within_f1', 'between_f1']].mean()
    
    print("\n--- Final Aggregated Results ---")
    print(final_summary)
    final_summary.to_csv(f"stress_results_n{args.subjects}_final.csv")