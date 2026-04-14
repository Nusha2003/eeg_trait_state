import joblib
import os
import pandas as pd
import numpy as np

def investigate_splits(split_dir):
    files = [f for f in os.listdir(split_dir) if f.endswith('.pkl')]
    files.sort()

    for file in files:
        file_path = os.path.join(split_dir, file)
        data = joblib.load(file_path)
        
        print(f"\n{'='*60}")
        print(f"FILE: {file}")
        print(f"{'='*60}")

        # 1. Top Level Keys
        print(f"Keys in pkl: {list(data.keys())}")
        
        # 2. Metadata Metadata
        metadata = data['metadata']
        print(f"Metadata Rows: {len(metadata)}")
        print(f"Subjects present: {metadata['subject'].nunique()}")

        # 👇 THIS is what you need
        n_classes = metadata['condition'].nunique()
        print(f"State Classes Present: {n_classes}")
        print("Example condition labels:")
        print(metadata['condition'].unique()[:12])
        
        # 3. Global Trait Split
        trait = data['trait_split']
        print(f"\nGLOBAL TRAIT SPLIT (Identification):")
        print(f"  - Train samples: {len(trait['train'])}")
        print(f"  - Test samples:  {len(trait['test'])}")

        # 4. Subject-Specific Splits (Peeking at the first subject)
        splits = data['splits']
        print(f"\nSUBJECT-SPECIFIC SPLITS (Total subjects in list: {len(splits)}):")
        
        example = splits[0]
        sub_id = example['subject']
        print(f"Example Subject: {sub_id}")
        
        # Helper to print split sizes
        for split_type in ['within_state', 'between_state']:
            train_sz = len(example[split_type]['train'])
            test_sz = len(example[split_type]['test'])
            print(f"  - {split_type:22} | Train: {train_sz:6} | Test: {test_sz:4}")


if __name__ == "__main__":
    SPLIT_DIR = "/home1/amadapur/projects/eeg_trait_state_geometry/splits/gamma"
    investigate_splits(SPLIT_DIR)