import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import joblib

"""
trait still survives state supervision
even when the model is trained to ignore subject - samples from the same person remain closer together
even when optimized for task, the model cannot destroy subject odentity
"""

# Assuming your perform_pca and lda_full_projection functions exist
from pca_embed import perform_pca

def lda_full_projection(X, y, max_components=10):
    n_classes = len(np.unique(y))
    k = min(max_components, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=k, solver="svd")
    return lda.fit_transform(X, y)

def run_safe_decoding(X, y, groups=None, mode='trait'):
    n_classes = len(np.unique(y))
    n_features = X.shape[1]  # Get the actual width of the current space
    
    # NEW LOGIC: components must be < n_classes AND <= n_features
    # We use n_features if it's smaller than our target of 10
    n_comp = min(10, n_classes - 1, n_features) 
    
    # If the space is already very small (like State-LDA), 
    # we might not even need an extra LDA in the pipeline.
    # But for consistency, we'll keep it and just use the smaller n_comp.
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(n_components=n_comp)),
        ('clf', LinearSVC(dual=False, max_iter=2000, random_state=42))
    ])

    if mode == 'trait':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=-1)
    elif mode == 'state_logo':
        cv = LeaveOneGroupOut()
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, n_jobs=-1)
        
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    print("Loading data and generating representation spaces...")
    Z_pca, Xz, labels, scaler, pca = perform_pca()

    # 1. Create the specialized spaces
    print("Projecting into Trait, State, and Joint spaces...")
    Z_trait = lda_full_projection(Xz, labels.subject, max_components=10)
    Z_state = lda_full_projection(Xz, labels.condition, max_components=3)
    
    joint_labels = labels.subject + "_" + labels.condition
    Z_joint = lda_full_projection(Xz, joint_labels, max_components=10)

    spaces = {
        'PCA (Unsupervised)': Z_pca,
        'Trait-LDA (Identity-Optimized)': Z_trait,
        'State-LDA (Task-Optimized)': Z_state,
        'Joint-LDA (Combined)': Z_joint
    }

    all_results = {}
    le = LabelEncoder()
    y_trait = le.fit_transform(labels.subject)

    # 2. Iterate through every space to test Trait and State access
    for space_name, X_space in spaces.items():
        print(f"\n--- Testing Space: {space_name} ---")
        
        # Test Trait (Subject ID)
        t_acc, t_std = run_safe_decoding(X_space, y_trait, mode='trait')
        
        # Test State (Task Condition) via Cross-Subject LOGO
        s_acc, s_std = run_safe_decoding(X_space, labels.condition, groups=labels.subject, mode='state_logo')
        
        all_results[space_name] = {
            'Trait_Acc': t_acc,
            'Trait_Std': t_std,
            'State_Acc': s_acc,
            'State_Std': s_std
        }
        
        print(f"  Trait Accuracy: {t_acc:.4f}")
        print(f"  State Accuracy (LOGO): {s_acc:.4f}")

    # 3. Final Comparison: Save results
    joblib.dump(all_results, "full_space_stress_test.pkl")
    print("\nAll results saved to full_space_stress_test.pkl")