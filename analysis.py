import sys
from pca_embed import perform_pca
from sklearn.metrics import pairwise_distances
import numpy as np
import itertools
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.spatial.distance import pdist, cdist
from joblib import parallel_config

from sklearn.model_selection import GroupKFold

def lda_cv_projection(X, y, groups, max_components=10, n_splits=5):
    Z_list = []
    folds = []
    
    gkf = GroupKFold(n_splits=n_splits)

    for train, test in gkf.split(X, y, groups):
        n_classes = len(np.unique(y[train]))
        k = min(max_components, n_classes - 1)
        
        lda = LinearDiscriminantAnalysis(n_components=k, solver="svd")
        lda.fit(X[train], y[train])
        Z_list.append(lda.transform(X[test]))
        folds.append(test)

    K = max(z.shape[1] for z in Z_list)
    Z = np.zeros((len(X), K))
    
    for z, idx in zip(Z_list, folds):
        Z[idx, :z.shape[1]] = z

    return Z

"""
def lda_full_projection(X, y, max_components=10):

    Simpler approach: train LDA on ALL data, transform ALL data

    n_classes = len(np.unique(y))
    k = min(max_components, n_classes - 1)
    
    lda = LinearDiscriminantAnalysis(n_components=k, solver="svd")
    lda.fit(X, y)  # ← Train on EVERYTHING
    Z = lda.transform(X)  # ← Transform EVERYTHING
    
    return Z

"""
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                        help="Which dataset we are using")
args = parser.parse_args()

#Defining representation spaces

Z, Xz, labels, scaler, pca = perform_pca()
"""
Xz: original EEG features after standardization
PCA embedding of EEG
Who the person is
What task they were doing
"""

#geometry maximally aligned to trait
#maximizes variance between subjects - looks at between-subject difference
Z_trait = lda_cv_projection(Xz, labels.subject, labels.subject)
Z_state = lda_cv_projection(Xz, labels.condition, labels.subject)
joint_labels = labels.subject + "_" + labels.condition
Z_joint = lda_cv_projection(Xz, joint_labels, labels.subject)


"""
Trait coherence: within subject distance/between subject distance
smaller value = more clustered by subject
"""
#change to balanced trait coherence
def balanced_trait_coherence(Z, subjects, min_pairs=100):
    scores = []

    for s in subjects.unique():
        idx_s = subjects == s
        if idx_s.sum() < 5:
            continue

        Zs = Z[idx_s]
        within = pdist(Zs).mean()

        between_vals = []
        for t in subjects.unique():
            if t == s:
                continue
            idx_t = subjects == t
            if idx_t.sum() < 5:
                continue

            Zt = Z[idx_t]
            D = cdist(Zs, Zt).ravel()
            if len(D) > min_pairs:
                D = np.random.choice(D, size=min_pairs, replace=False)
            between_vals.append(D.mean())

        scores.append(within / np.mean(between_vals))

    return np.mean(scores), np.std(scores)


print("\n=== Trait coherence across spaces ===")
print("PCA:   ", balanced_trait_coherence(Z, labels.subject))
print("Trait: ", balanced_trait_coherence(Z_trait, labels.subject))
print("State: ", balanced_trait_coherence(Z_state, labels.subject))
print("Joint: ", balanced_trait_coherence(Z_joint, labels.subject))

"""
Identity is geometrically encoded
"""

#state separability
#how much task warps EEG within the same person
if args.dataset == "motor_simplified":
    states = ["resting", "real", "imaginary"]
elif args.dataset == "motor_all":
    states = ["baseline_eyes_open",
"baseline_eyes_closed",
"task1_real_left_fist",
"task1_real_right_fist",
"task2_imag_left_fist",
"task2_imag_right_fist",
"task3_real_both_fists",
"task3_real_both_feet",
"task4_imag_both_fists",
"task4_imag_both_feet"]

"""

is the distance between states larger than the natural variability within a state?
state modulates geometry only weakly


does task warp EEG more than natural noise?
Task differences are about the same size as natural variati
removing the confound of subject differences
How well different task states are separated within the same subject in each embedding space
"""


spaces = {
    "PCA":   Z,
    "Trait": Z_trait,
    "State": Z_state,
    "Joint": Z_joint
}

def compute_within_subject_df(Zspace):
    rows = []

    for s in labels.subject.unique():

        sub = labels[labels.subject == s]
        if len(sub) < 20:
            continue

        for s1 in states:
            for s2 in states:
                if s1 >= s2:
                    continue

                idx1 = (labels.subject == s) & (labels.condition == s1)
                idx2 = (labels.subject == s) & (labels.condition == s2)

                if idx1.sum() < 3 or idx2.sum() < 3:
                    continue

                Z1 = Zspace[idx1]
                Z2 = Zspace[idx2]
                #mean distance b/w states s1 and s2 within same subject/mean distance within state s1 for same subject
                sep = (
                    pairwise_distances(Z1, Z2).mean() /
                    pdist(Z1).mean()
                )

                rows.append({
                    "subject": s,
                    "pair": f"{s1}-{s2}",
                    "separability": sep
                })

    return pd.DataFrame(rows)

for name, Zspace in spaces.items():

    df = compute_within_subject_df(Zspace)

    summary = (
        df.groupby("pair")["separability"]
          .agg(["mean", "std", "count"])
          .sort_values("mean", ascending=False)
    )

    print(f"\n=== Population-averaged within-subject separability ({name}) ===")
    print(summary.to_string(float_format="%.3f"))

    summary.to_csv(f"within_subject_separability_{name}.csv")

"""
plt.figure(figsize=(7,5))
sns.violinplot(x="pair", y="separability", data=df, inner="box")
plt.axhline(1.0, linestyle="--", color="gray")
plt.ylabel("Within-subject state separability")
plt.title("State geometry within individual subjects")
plt.tight_layout()
plt.savefig("fig3_within_subject_state_geometry.png", dpi=300)
plt.close()
"""
"""
State modulates each person only weakly
Restrict test to within a single subject
"""





#effect of supervision
from scipy.stats import spearmanr
def rdm(Z):
    return pairwise_distances(Z)


def rdm_similarity(A, B):
    iu = np.triu_indices(A.shape[0], k=1)
    return spearmanr(A[iu], B[iu])[0]
R_pca   = rdm(Z)
R_trait = rdm(Z_trait)
R_state = rdm(Z_state)
R_joint = rdm(Z_joint)


#does supervision change geometry or preserve it?
print("\n=== Representational Geometry Preservation ===")
print("PCA → Trait-LDA:",  rdm_similarity(R_pca, R_trait))
print("PCA → State-LDA:",  rdm_similarity(R_pca, R_state))
print("PCA → Joint-LDA:",  rdm_similarity(R_pca, R_joint))


#use cluster for this


#define hierarchy metric - intraindividual/inter-individual

def calculate_hierarchy_metric(features, labels):
    # 1. Inter-Subject Distance (Trait Signal)
    sub_centroids = []
    for sub in labels.subject.unique():
        sub_centroids.append(features[labels.subject == sub].mean(axis=0))
    
    trait_dist = np.mean(np.linalg.norm(np.array(sub_centroids)[:, None] - sub_centroids, axis=2))

    # 2. Inter-Task Distance (State Signal)
    task_centroids = []
    for cond in labels.condition.unique():
        task_centroids.append(features[labels.condition == cond].mean(axis=0))
    
    state_dist = np.mean(np.linalg.norm(np.array(task_centroids)[:, None] - task_centroids, axis=2))

    return trait_dist / state_dist

spaces = {
    "PCA (Baseline)": Z,
    "Trait-LDA": Z_trait,
    "State-LDA": Z_state,
    "Joint-LDA": Z_joint
}

hierarchy_results = {name: calculate_hierarchy_metric(sp, labels) for name, sp in spaces.items()}

for space, score in hierarchy_results.items():
    print(f"Hierarchy Metric in {space}: {score:.4f}")


import matplotlib.pyplot as plt

# We use a subset of points (e.g., 500) to keep the plot readable
subset_idx = np.random.choice(len(Z), 500, replace=False)
d_pca = pairwise_distances(Z[subset_idx]).ravel()
d_state = pairwise_distances(Z_state[subset_idx]).ravel()

plt.figure(figsize=(6, 6))
plt.scatter(d_pca, d_state, alpha=0.1, s=1, color='teal')
plt.title(f"Geometry Preservation (r={rdm_similarity(R_pca, R_state):.2f})")
plt.xlabel("Distance in PCA (Trait-Dominant)")
plt.ylabel("Distance in State-LDA (Task-Optimized)")
plt.tight_layout()
plt.savefig("geometry_preservation_scatter.png")



