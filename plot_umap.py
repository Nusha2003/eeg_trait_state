import umap
from joblib import parallel_config
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pca_embed import perform_pca
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


def lda_full_projection(X, y, max_components=10):
    n_classes = len(np.unique(y))
    k = min(max_components, n_classes - 1)
    
    lda = LinearDiscriminantAnalysis(n_components=k, solver="svd")
    lda.fit(X, y)  
    Z = lda.transform(X)  
    
    return Z


def simplify_condition(cond):
    """Simplify 10 conditions into 3 categories for cleaner visualization"""
    if 'baseline' in cond:
        return 'Baseline'
    elif 'real' in cond:
        return 'Real Movement'
    elif 'imag' in cond:
        return 'Imagined Movement'
    return cond


def plot_2d_single_space(Z_2d, labels, filename, title, n_subs=8):
    """
    Plot a single 2D representation space
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add simplified condition column
    labels_copy = labels.copy()
    labels_copy['cond_simple'] = labels_copy.condition.apply(simplify_condition)
    
    # Marker map for simplified conditions
    marker_map = {
        'Baseline': 'o',
        'Real Movement': '^',
        'Imagined Movement': 's'
    }
    
    # Select top subjects
    top_subs = labels_copy.subject.value_counts().nlargest(n_subs).index
    colors = plt.cm.tab10(np.linspace(0, 1, n_subs))
    
    # Plot each subject
    for i, sub in enumerate(top_subs):
        sub_mask = labels_copy.subject == sub
        
        for cond_simple in marker_map.keys():
            mask = sub_mask & (labels_copy.cond_simple == cond_simple)
            if mask.sum() == 0: continue
            
            ax.scatter(
                Z_2d[mask, 0], 
                Z_2d[mask, 1],
                color=colors[i],
                marker=marker_map[cond_simple],
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5,
                label=f"Sub {sub}" if cond_simple == 'Baseline' else None
            )
    
    # Subject legend (colors)
    sub_legend = ax.legend(loc='upper left', title="Subjects", 
                          bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)
    ax.add_artist(sub_legend)
    
    # Condition legend (markers)
    legend_elements = [
        Line2D([0], [0], marker=marker_map[c], color='w', 
               label=c, markerfacecolor='gray', markersize=10, 
               markeredgecolor='white', markeredgewidth=0.5)
        for c in marker_map.keys()
    ]
    ax.legend(handles=legend_elements, loc='lower left', 
             title="Task Conditions", bbox_to_anchor=(1.02, 0), 
             frameon=True, fontsize=9)
    
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_2d_four_panel(Z_2d_dict, labels, filename="all_spaces_2x2.png", n_subs=6):
    """
    2x2 grid showing all four representation spaces
    Main figure for paper
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Add simplified condition column
    labels_copy = labels.copy()
    labels_copy['cond_simple'] = labels_copy.condition.apply(simplify_condition)
    
    marker_map = {
        'Baseline': 'o',
        'Real Movement': '^',
        'Imagined Movement': 's'
    }
    
    # Select top subjects
    top_subs = labels_copy.subject.value_counts().nlargest(n_subs).index
    colors = plt.cm.tab10(np.linspace(0, 1, n_subs))
    
    space_info = [
        ('PCA', 'A) PCA (Unsupervised Baseline)'),
        ('Trait-LDA', 'B) Trait-LDA (Subject-Optimized)'),
        ('State-LDA', 'C) State-LDA (Task-Optimized)'),
        ('Joint-LDA', 'D) Joint-LDA (Subject×Task)')
    ]
    
    for idx, (space_name, title) in enumerate(space_info):
        ax = axes[idx]
        Z_2d = Z_2d_dict[space_name]
        
        # Plot each subject
        for i, sub in enumerate(top_subs):
            sub_mask = labels_copy.subject == sub
            
            for cond_simple in marker_map.keys():
                mask = sub_mask & (labels_copy.cond_simple == cond_simple)
                if mask.sum() == 0: continue
                
                ax.scatter(
                    Z_2d[mask, 0], 
                    Z_2d[mask, 1],
                    color=colors[i],
                    marker=marker_map[cond_simple],
                    alpha=0.7,
                    s=40,
                    edgecolors='white',
                    linewidth=0.5,
                    label=f"Sub {sub}" if cond_simple == 'Baseline' and idx == 0 else None
                )
        
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title(title, fontsize=12, pad=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle='--')
    
    # Shared legends outside plot area
    # Subject legend
    handles_sub = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=colors[i], markersize=10,
                              markeredgecolor='white', markeredgewidth=0.5,
                              label=f'Sub {sub}')
                   for i, sub in enumerate(top_subs)]
    
    fig.legend(handles=handles_sub, loc='center', 
              bbox_to_anchor=(0.92, 0.75), title="Subjects",
              frameon=True, fontsize=9)
    
    # Condition legend
    handles_cond = [plt.Line2D([0], [0], marker=marker_map[c], color='w', 
                               markerfacecolor='gray', markersize=10,
                               markeredgecolor='white', markeredgewidth=0.5,
                               label=c)
                    for c in marker_map.keys()]
    
    fig.legend(handles=handles_cond, loc='center',
              bbox_to_anchor=(0.92, 0.25), title="Task Conditions",
              frameon=True, fontsize=9)
    
    plt.suptitle("Subject Manifold Structure Across Representation Spaces", 
                 fontsize=15, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_2d_with_hulls_comparison(Z_2d_dict, labels, 
                                   filename="trait_vs_state_hulls.png", n_subs=5):
    """
    Side-by-side: Trait-LDA vs State-LDA with convex hulls
    Shows trait structure persists even in state-optimized space
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    labels_copy = labels.copy()
    labels_copy['cond_simple'] = labels_copy.condition.apply(simplify_condition)
    
    marker_map = {
        'Baseline': 'o',
        'Real Movement': '^',
        'Imagined Movement': 's'
    }
    
    # Select subjects
    top_subs = labels_copy.subject.value_counts().nlargest(n_subs).index
    colors = plt.cm.Set2(np.linspace(0, 1, n_subs))
    
    for ax, space_name, title in [(ax1, 'Trait-LDA', 'Trait-LDA (Subject-Optimized)'),
                                   (ax2, 'State-LDA', 'State-LDA (Task-Optimized)')]:
        
        Z_2d = Z_2d_dict[space_name]
        
        # Plot hulls and points for each subject
        for i, sub in enumerate(top_subs):
            sub_mask = labels_copy.subject == sub
            points = Z_2d[sub_mask]
            
            # Draw convex hull
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    # Fill hull
                    ax.fill(hull_points[:, 0], hull_points[:, 1], 
                           color=colors[i], alpha=0.15)
                    # Hull boundary
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 
                               color=colors[i], alpha=0.4, linewidth=2)
                except:
                    pass
            
            # Plot points
            for cond_simple in marker_map.keys():
                mask = sub_mask & (labels_copy.cond_simple == cond_simple)
                if mask.sum() == 0: continue
                
                ax.scatter(
                    Z_2d[mask, 0], 
                    Z_2d[mask, 1],
                    color=colors[i],
                    marker=marker_map[cond_simple],
                    alpha=0.8,
                    s=60,
                    edgecolors='white',
                    linewidth=0.8,
                    label=f"Sub {sub}" if cond_simple == 'Baseline' and ax == ax1 else None,
                    zorder=3
                )
        
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(title, fontsize=13, pad=15, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle='--')
    
    # Shared legends
    sub_legend = ax1.legend(loc='upper left', title="Subjects", 
                           bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)
    
    legend_elements = [
        Line2D([0], [0], marker=marker_map[c], color='w', 
               label=c, markerfacecolor='gray', markersize=10, 
               markeredgecolor='white', markeredgewidth=0.8)
        for c in marker_map.keys()
    ]
    ax2.legend(handles=legend_elements, loc='upper left', 
              title="Task Conditions", bbox_to_anchor=(1.02, 1), 
              frameon=True, fontsize=10)
    
    plt.suptitle("Trait Structure Persists Under Task-Optimization", 
                 fontsize=15, y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def compute_manifold_metrics(Z_2d_dict, labels):
    """
    Quantify manifold structure across all spaces
    """
    from scipy.spatial.distance import pdist, cdist
    
    print("\n" + "="*60)
    print("MANIFOLD STRUCTURE METRICS (2D UMAP Space)")
    print("="*60)
    
    top_subs = labels.subject.value_counts().nlargest(8).index
    
    for space_name, Z_2d in Z_2d_dict.items():
        within_dists = []
        between_dists = []
        
        for i, sub1 in enumerate(top_subs):
            mask1 = labels.subject == sub1
            if mask1.sum() > 1:
                within_dists.append(pdist(Z_2d[mask1]).mean())
            
            for sub2 in top_subs[i+1:]:
                mask2 = labels.subject == sub2
                between_dists.append(cdist(Z_2d[mask1], Z_2d[mask2]).mean())
        
        within_mean = np.mean(within_dists)
        between_mean = np.mean(between_dists)
        ratio = between_mean / within_mean
        
        print(f"\n{space_name}:")
        print(f"  Within-subject distance:  {within_mean:.3f}")
        print(f"  Between-subject distance: {between_mean:.3f}")
        print(f"  Separation ratio:         {ratio:.2f}x")
    
    print("\n" + "="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("Loading data...")
Z, Xz, labels, scaler, pca = perform_pca()

print("Creating LDA embeddings...")
Z_trait = lda_full_projection(Xz, labels.subject)
Z_state = lda_full_projection(Xz, labels.condition)
joint_labels = labels.subject + "_" + labels.condition
Z_joint = lda_full_projection(Xz, joint_labels)

# Store all embedding spaces
spaces = {
    'PCA': Z,
    'Trait-LDA': Z_trait,
    'State-LDA': Z_state,
    'Joint-LDA': Z_joint
}

n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

print(f"\nRunning UMAP transformations on {n_cpus} core(s)...")
Z_2d_dict = {}

with parallel_config(backend='threading', n_jobs=n_cpus):
    for space_name, Z_space in spaces.items():
        print(f"  Processing {space_name}...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            n_jobs=n_cpus,
            low_memory=True,
            force_approximation_algorithm=True,
            random_state=42
        )
        Z_2d_dict[space_name] = reducer.fit_transform(Z_space)

print("\nGenerating plots...")

# 1. Individual plots for each space
print("\n1. Creating individual space plots...")
for space_name in spaces.keys():
    filename = f"{space_name.replace('-','_').lower()}_2d.png"
    title = f"{space_name} Representation Space"
    plot_2d_single_space(Z_2d_dict[space_name], labels, filename, title, n_subs=8)

# 2. Four-panel comparison (MAIN FIGURE)
print("\n2. Creating 2x2 comparison plot (MAIN FIGURE)...")
plot_2d_four_panel(Z_2d_dict, labels, filename="all_spaces_2x2.png", n_subs=6)

# 3. Trait vs State with hulls (KEY FINDING)
print("\n3. Creating Trait vs State comparison with hulls...")
plot_2d_with_hulls_comparison(Z_2d_dict, labels, 
                              filename="trait_vs_state_hulls.png", n_subs=5)

# 4. Compute quantitative metrics
print("\n4. Computing manifold metrics...")
compute_manifold_metrics(Z_2d_dict, labels)

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)
print("\nMain Figures for Paper:")
print("  📊 all_spaces_2x2.png          - 2x2 grid of all spaces (MAIN FIGURE)")
print("  🎯 trait_vs_state_hulls.png    - Key finding with manifold boundaries")
print("\nIndividual Space Plots:")
print("  📈 pca_2d.png                  - PCA space")
print("  📈 trait_lda_2d.png            - Trait-LDA space")
print("  📈 state_lda_2d.png            - State-LDA space")
print("  📈 joint_lda_2d.png            - Joint-LDA space")
print("\n" + "="*60)