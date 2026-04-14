import pandas as pd
import numpy as np
import joblib
import argparse
import os



def make_splits(labels_path, n_subjects=None, seed=42):

    labels_df = pd.read_csv(labels_path)


    rng = np.random.default_rng(seed)

    all_subs = labels_df['subject'].unique()

    if n_subjects is None:
        selected_subs = all_subs
    else:
        selected_subs = rng.choice(all_subs, size=min(n_subjects, len(all_subs)), replace=False)

    df = labels_df[labels_df['subject'].isin(selected_subs)].reset_index(drop=True)

    split_results = []

    rng.shuffle(selected_subs)
    mid = len(selected_subs) // 2

    train_subs = selected_subs[:mid]
    test_subs  = selected_subs[mid:]

    trait_train = df[df['subject'].isin(train_subs)].index.tolist()
    trait_test  = df[df['subject'].isin(test_subs)].index.tolist()

    for target_sub in selected_subs:

        subj_mask = (df['subject'] == target_sub)

        subj_indices = df[subj_mask].index.tolist()

        rng.shuffle(subj_indices)
        half = len(subj_indices) // 2

        within_train = subj_indices[:half]
        within_test  = subj_indices[half:]

        between_train = df[~subj_mask].index.tolist()
        between_test  = df[subj_mask].index.tolist()

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

    parser.add_argument(
        "--labels",
        type=str,
        default="/home1/amadapur/projects/eeg_trait_state_geometry/data/lemon/lemon_labels.csv"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home1/amadapur/projects/eeg_trait_state_geometry/splits/lemon"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sizes = [None, 20, 10, 4]   # None = all subjects

    for size in sizes:

        iterations = 1 if size is None else 10

        for i in range(iterations):

            seed = 42 + i

            splits, trait_split, metadata = make_splits(
                args.labels,
                n_subjects=size,
                seed=seed
            )

            if size is None:
                filename = "splits_all.pkl"
            else:
                filename = f"splits_n{size}_rep{i}.pkl"

            output_file = os.path.join(args.out_dir, filename)

            joblib.dump({
                'splits': splits,
                'trait_split': trait_split,
                'metadata': metadata
            }, output_file)

    print("Splits generated")