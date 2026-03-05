import numpy as np
import itertools


class HierarchyMetric:
    def __init__(self, Z, labels):
        self.Z = Z
        self.labels = labels.reset_index(drop=True)

    def compute_class_centroids(self, Z, labels):

        centroids = {}

        subjects = labels["subject"].values
        conditions = labels["condition"].values

        unique_subjects = np.unique(subjects)
        unique_conditions = np.unique(conditions)

        for s in unique_subjects:
            for c in unique_conditions:

                idx = np.where((subjects == s) & (conditions == c))[0]

                if len(idx) == 0:
                    continue

                centroids[(s, c)] = Z[idx].mean(axis=0)

        return centroids

    def compute_hierarchy(self, centroids):

        by_subject = {}
        by_condition = {}

        for (s, c), vec in centroids.items():

            by_subject.setdefault(s, {})[c] = vec
            by_condition.setdefault(c, {})[s] = vec

        intra_subject_means = []

        for s, conds in by_subject.items():

            dists = []

            for (c1, v1), (c2, v2) in itertools.combinations(conds.items(), 2):
                dists.append(np.linalg.norm(v1 - v2))

            if len(dists) > 0:
                intra_subject_means.append(np.mean(dists))

        intra = np.mean(intra_subject_means)

        inter_condition_means = []

        for c, subs in by_condition.items():

            dists = []

            for (s1, v1), (s2, v2) in itertools.combinations(subs.items(), 2):
                dists.append(np.linalg.norm(v1 - v2))

            if len(dists) > 0:
                inter_condition_means.append(np.mean(dists))

        inter = np.mean(inter_condition_means)

        ratio = inter / intra

        return ratio, inter, intra

    def permute_subject_within_condition(self, labels):

        permuted = labels.copy()

        for c in permuted["condition"].unique():

            idx = permuted["condition"] == c

            permuted.loc[idx, "subject"] = np.random.permutation(
                permuted.loc[idx, "subject"].values
            )

        return permuted


    def compute_null_distribution(self, n_perm=1000):

        null_ratios = []

        for i in range(n_perm):

            perm_labels = self.permute_subject_within_condition(self.labels)

            centroids = self.compute_class_centroids(self.Z, perm_labels)

            r, _, _ = self.compute_hierarchy(centroids)

            null_ratios.append(r)

        return np.array(null_ratios)

    def evaluate(self, n_perm=1000, verbose=True):

        if verbose:
            print("\nEvaluating hierarchy structure...")

        centroids = self.compute_class_centroids(self.Z, self.labels)

        real_ratio, inter, intra = self.compute_hierarchy(centroids)

        null_ratios = self.compute_null_distribution(n_perm)


        pval = np.mean(null_ratios >= real_ratio)

        if verbose:

            print(f"INTER distance : {inter:.4f}")
            print(f"INTRA distance : {intra:.4f}")
            print(f"RATIO (inter/intra): {real_ratio:.4f}")

            print("\nNull distribution:")
            print(f"mean : {null_ratios.mean():.4f}")
            print(f"std  : {null_ratios.std():.4f}")

            print(f"\np-value : {pval:.6f}")

        return {
            "ratio": real_ratio,
            "inter": inter,
            "intra": intra,
            "null": null_ratios,
            "p_value": pval,
        }




















"""
Trait coherence: within subject distance/between subject distance
smaller value = more clustered by subject

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


Identity is geometrically encoded


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



is the distance between states larger than the natural variability within a state?
state modulates geometry only weakly


does task warp EEG more than natural noise?
Task differences are about the same size as natural variati
removing the confound of subject differences
How well different task states are separated within the same subject in each embedding space



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


plt.figure(figsize=(7,5))
sns.violinplot(x="pair", y="separability", data=df, inner="box")
plt.axhline(1.0, linestyle="--", color="gray")
plt.ylabel("Within-subject state separability")
plt.title("State geometry within individual subjects")
plt.tight_layout()
plt.savefig("fig3_within_subject_state_geometry.png", dpi=300)
plt.close()

State modulates each person only weakly
Restrict test to within a single subject






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


def calculate_state_vector_alignment(Zspace, labels, state_list=None):

    if state_list is None:
        # Default to comparing a baseline to all 'real' task states
        baseline = "baseline_eyes_open"
        comparison_states = [s for s in labels.condition.unique() if "real" in s]
    else:
        baseline = state_list[0]
        comparison_states = state_list[1:]

    all_pair_similarities = []

    for target_state in comparison_states:
        subject_vectors = []
        
        for s in labels.subject.unique():
            # Get masks for the specific subject and condition pair
            idx_a = (labels.subject == s) & (labels.condition == baseline)
            idx_b = (labels.subject == s) & (labels.condition == target_state)

            if idx_a.sum() > 5 and idx_b.sum() > 5:
                # Calculate the 'State Vector' (the displacement on the manifold)
                vec = Zspace[idx_b].mean(axis=0) - Zspace[idx_a].mean(axis=0)
                
                # Normalize to unit length for Cosine Similarity
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    subject_vectors.append(vec / norm)

        if len(subject_vectors) >= 2:
            V = np.array(subject_vectors)
            # Dot product of normalized vectors = Cosine Similarity
            sim_matrix = V @ V.T
            # Take mean of upper triangle (similarities between different subjects)
            upper_idx = np.triu_indices(len(sim_matrix), k=1)
            all_pair_similarities.append(sim_matrix[upper_idx].mean())

    return np.mean(all_pair_similarities) if all_pair_similarities else 0.0

# --- Execution across the Representation Spaces ---

# Your specific task labels
states_to_compare = [
    "baseline_eyes_open",
    "task1_real_left_fist",
    "task1_real_right_fist",
    "task3_real_both_fists",
    "task3_real_both_feet"
]

print("\n=== Geometric Parallelism (State Vector Alignment) ===")
print("Measures if the 'direction' of a task is consistent across subjects.")
print("-" * 55)

spaces = {
    "PCA (Euclidean)": Z,
    "Trait-LDA": Z_trait,
    "State-LDA": Z_state,
    "Joint-LDA": Z_joint
}

for name, space in spaces.items():
    score = calculate_state_vector_alignment(space, labels, states_to_compare)
    print(f"{name:<20} : {score:.4f}")
    """