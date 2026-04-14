"""
within-subject (state decoding): use current splits (train, test)
between subject (state decoding):all other subjects + last run of held out subject 
trait decoding: use existing train and test splits
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os

def make_splits(labels_path, n_subjects, seed = 42):
    labels_df = pd.read_csv(labels_path)
    rng = np.random.default_rng(seed)
    all_subs = labels_df['subject'].unique()
    selected_subs = rng.choice(all_subs, size=min(n_subjects, len(all_subs)), replace=False)
    df = labels_df[labels_df['subject'].isin(selected_subs)].reset_index(drop=True)
    split_results = []

    #between subject trait decoding
    trait_train = df[df['original_part'] == "train"].index.tolist()
    trait_test = df[df['original_part'] == "test"].index.tolist()

    for target_sub in selected_subs:
        subj_mask = (df['subject'] == target_sub)
        within_train = (
            df[subj_mask & (df['original_part']=="train")].index.tolist()
        )
        within_test = (
            df[subj_mask & (df['original_part']=="test")].index.tolist()
        )

        #between subject test
        between_train = df[~subj_mask].index.tolist()
        between_test  = df[subj_mask & (df['original_part'] == "test")].index.tolist()


        split_results.append({
            'subject': target_sub,
            'within_state': {
                'train': within_train,
                'test': within_test
            },
            'between_state': {
                'train': between_train,
                'test': between_test
            }
        })
        
    return split_results, {'train': trait_train, 'test': trait_test}, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="/home1/amadapur/projects/eeg_trait_state_geometry/data/gamma/gamma_psd_labels.csv")
    parser.add_argument("--out_dir", type=str, default="/home1/amadapur/projects/eeg_trait_state_geometry/splits/gamma")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for size in [14, 7, 4, 10]:
        iterations = 10 if size in [7, 4, 10] else 1
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
