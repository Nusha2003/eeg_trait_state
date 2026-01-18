import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pca_embed import perform_pca
from joblib import Parallel, delayed

# --- Core Functions ---

def lda_full_projection(X, y, max_components=10):
    n_classes = len(np.unique(y))
    k = min(max_components, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=k, solver="svd")
    return lda.fit_transform(X, y)

def balanced_trait_coherence(Z, subjects, min_pairs=100):
    scores = []
    for s in subjects.unique():
        idx_s = subjects == s
        if idx_s.sum() < 5: continue
        Zs = Z[idx_s]
        within = pdist(Zs).mean()
        between_vals = []
        for t in subjects.unique():
            if t == s: continue
            idx_t = subjects == t
            if idx_t.sum() < 5: continue
            Zt = Z[idx_t]
            D = cdist(Zs, Zt).ravel()
            if len(D) > min_pairs:
                D = np.random.choice(D, size=min_pairs, replace=False)
            between_vals.append(D.mean())
        scores.append(within / np.mean(between_vals))
    return np.mean(scores), np.std(scores)

# --- Parallel Worker Functions ---

def coherence_worker(Z, subjects):
    """Worker for a single coherence permutation."""
    shuffled_subjects = np.random.permutation(subjects.values)
    score, _ = balanced_trait_coherence(Z, pd.Series(shuffled_subjects))
    return score

def rsa_worker(R_A, Z_B, iu):
    """Worker for a single RSA permutation."""
    idx = np.random.permutation(len(Z_B))
    R_B_shuffled = pairwise_distances(Z_B[idx])
    corr, _ = spearmanr(R_A[iu], R_B_shuffled[iu])
    return corr

# --- Parallel Test Orchestrators ---

def permutation_test_coherence_parallel(Z, subjects, n_permutations=1000, n_jobs=-1):
    observed_mean, _ = balanced_trait_coherence(Z, subjects)
    print(f"Running {n_permutations} Coherence permutations on {n_jobs} cores...")
    
    null_means = Parallel(n_jobs=n_jobs)(
        delayed(coherence_worker)(Z, subjects) for _ in range(n_permutations)
    )
    
    p_value = np.sum(np.array(null_means) <= observed_mean) / n_permutations
    return observed_mean, p_value, null_means

def permutation_test_rsa_parallel(Z_A, Z_B, n_permutations=1000, n_jobs=-1):
    R_A = pairwise_distances(Z_A)
    R_B = pairwise_distances(Z_B)
    iu = np.triu_indices(R_A.shape[0], k=1)
    observed_corr, _ = spearmanr(R_A[iu], R_B[iu])
    print(f"Running {n_permutations} RSA permutations on {n_jobs} cores...")
    
    null_corrs = Parallel(n_jobs=n_jobs)(
        delayed(rsa_worker)(R_A, Z_B, iu) for _ in range(n_permutations)
    )
    
    p_value = np.sum(np.array(null_corrs) >= observed_corr) / n_permutations
    return observed_corr, p_value, null_corrs

# --- Utilities ---

def plot_permutation_results(null_dist, observed_val, title, save_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(null_dist, kde=True, color='gray', label='Null Distribution')
    plt.axvline(observed_val, color='red', linestyle='--', label='Observed')
    plt.title(title)
    plt.xlabel("Metric Value")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Data
    Z_pca, Xz, labels, scaler, pca = perform_pca()
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
    
    full_results = {}

    for name, Z_space in spaces.items():
        print(f"\n--- Parallel Analysis: {name} ---")
        
        # Run Coherence Test
        obs_c, p_c, null_c = permutation_test_coherence_parallel(
            Z_space, labels.subject, n_permutations=1000, n_jobs=-1
        )
        
        # Run RSA Test (compared to natural PCA geometry)
        obs_r, p_r, null_r = permutation_test_rsa_parallel(
            Z_pca, Z_space, n_permutations=1000, n_jobs=-1
        )
        
        full_results[name] = {
            'coherence': {'obs': obs_c, 'p': p_c},
            'rsa': {'obs': obs_r, 'p': p_r}
        }
        
        plot_permutation_results(null_c, obs_c, f"Coherence: {name}", f"perm_coherence_{name}.png")

    joblib.dump(full_results, "permutation_across_spaces.pkl")
    print("\nAll parallel analyses complete. Results saved.")