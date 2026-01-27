import pandas as pd
import numpy as np
import joblib
import argparse
import os

def make_splits(labels_path, n_subjects=109, seed=42):
    """
    - Within-Subject: 
        Baseline (Runs 1-2): 1st half train, 2nd half test.
        Tasks: 3-10 Train, 11-14 Test.
    - Between-Subject:
        Train: All data except Runs 11-14 of the target subject.
        Test: Runs 11-14 of the target subject.
    """
    labels_df = pd.read_csv(labels_path)
    rng = np.random.default_rng(seed)
    
    all_subs = labels_df['subject'].unique()
    selected_subs = rng.choice(all_subs, size=min(n_subjects, len(all_subs)), replace=False)
    
    # Filter to current stress-test group
    df = labels_df[labels_df['subject'].isin(selected_subs)].reset_index(drop=True)
    
    split_results = []

    for target_sub in selected_subs:
        subj_mask = (df['subject'] == target_sub)
        
        #within subject
        # Baseline trials (Runs 1-2)
        b_indices = df[subj_mask & df['run'].isin([1, 2])].index.tolist()
        mid = len(b_indices) // 2
        b_train_w = b_indices[:mid]
        b_test_w  = b_indices[mid:]
        
        # Task trials (Runs 3-14)
        t_train_w = df[subj_mask & df['run'].between(3, 10)].index.tolist()
        t_test_w  = df[subj_mask & df['run'].between(11, 14)].index.tolist()
        
        within_train = b_train_w + t_train_w
        within_test  = b_test_w + t_test_w
        
       #between subject
        between_train = df[~(subj_mask & df['run'].between(11, 14))].index.tolist()
        between_test  = t_test_w 
        
        split_results.append({
            'subject': target_sub,
            'within': {'train': within_train, 'test': within_test},
            'between': {'train': between_train, 'test': between_test}
        })
        
    return split_results, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--out_dir", type=str, default="splits", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for size in [109, 20, 10, 4]:
        iterations = 10 if size == 10 else 1
        for i in range(iterations):
            seed = 42 + i
            splits, metadata = make_splits(args.labels, n_subjects=size, seed=seed)
            
            suffix = f"_n{size}_rep{i}" if iterations > 1 else f"_n{size}"
            output_file = os.path.join(args.out_dir, f"splits{suffix}.pkl")
            
            joblib.dump({'splits': splits, 'metadata': metadata}, output_file)
            
    print(f"Splits generated. Within-subject uses 50/50 baseline split; Between uses everything except 11-14.")