import pandas as pd
import numpy as np
import joblib
import argparse
import os

# Define subjects to exclude
# Handling as both ints and strings to be safe
IGNORED_IDS = [88, 90, 92, 100]
IGNORED_STRS = [f"S{i:03d}" for i in IGNORED_IDS]
IGNORED_SUBJECTS = IGNORED_IDS + IGNORED_STRS

def make_splits(labels_path, n_subjects=109, seed=42):
    labels_df = pd.read_csv(labels_path)
    
    labels_df = labels_df[~labels_df['subject'].isin(IGNORED_SUBJECTS)].reset_index(drop=True)
    
    rng = np.random.default_rng(seed)
    
    all_subs = labels_df['subject'].unique()

    selected_subs = rng.choice(all_subs, size=min(n_subjects, len(all_subs)), replace=False)
    df = labels_df[labels_df['subject'].isin(selected_subs)].reset_index(drop=True)
    
    split_results = []
    
    # between subject trait decoding
    trait_train = df[df['run'].between(1, 10)].index.tolist()
    trait_test  = df[df['run'].between(11, 14)].index.tolist()

    for target_sub in selected_subs:
        subj_mask = (df['subject'] == target_sub)
        
        # Within-subject state decoding
        b_indices = df[subj_mask & df['run'].isin([1, 2])].index.tolist()
        mid = len(b_indices) // 2
        within_train = b_indices[:mid] + df[subj_mask & df['run'].between(3, 10)].index.tolist()
        within_test  = b_indices[mid:] + df[subj_mask & df['run'].between(11, 14)].index.tolist()
        
        # test set for between-state experiments
        test_runs_idx = df[subj_mask & df['run'].between(11, 14)].index.tolist()

        # between subject decoding
        pure_train = df[~subj_mask].index.tolist() 
        
        # between subject w/ leakage
        nonpure_train = df[~(subj_mask & df['run'].between(11, 14))].index.tolist()
        
        split_results.append({
            'subject': target_sub,
            'within_state': {'train': within_train, 'test': within_test},
            'between_state_pure': {'train': pure_train, 'test': test_runs_idx},
            'between_state_nonpure': {'train': nonpure_train, 'test': test_runs_idx}
        })
        
    return split_results, {'train': trait_train, 'test': trait_test}, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="/home1/amadapur/projects/eeg_trait_state_geometry/data/motor_psd_labels.csv")
    parser.add_argument("--out_dir", type=str, default="/home1/amadapur/projects/eeg_trait_state_geometry/splits")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for size in [109, 20, 10, 4]:
        iterations = 10 if size == 10 else 1
        for i in range(iterations):
            seed = 42 + i
            splits, trait_split, metadata = make_splits(args.labels, n_subjects=size, seed=seed)
            
            suffix = f"_n{size}_rep{i}" if iterations > 1 else f"_n{size}"
            output_file = os.path.join(args.out_dir, f"splits{suffix}.pkl")
            
            joblib.dump({
                'splits': splits, 
                'trait_split': trait_split, 
                'metadata': metadata
            }, output_file)
            
    print(f"Splits generated")