from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

def perform_pca():
    X = pd.read_csv("/home1/amadapur/projects/eeg_trait_state_geometry/data/motor_imagery_balanced_features.csv").values
    labels = pd.read_csv("/home1/amadapur/projects/eeg_trait_state_geometry/data/motor_imagery_balanced_labels.csv")
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=10)
    Z = pca.fit_transform(Xz)
    return Z, Xz, labels, scaler, pca



"""
plotting for initial analysis


FIG_DIR = "/home1/amadapur/projects/eeg_trait_state_geometry/figures"
os.makedirs(FIG_DIR, exist_ok=True)

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection="3d")

colors = pd.factorize(labels.subject)[0]

ax.scatter(
    Z[:,0],
    Z[:,1],
    Z[:,2],
    c=colors,
    s=20,
    alpha=0.7
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA Trait Manifold")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/pca_3d_trait_manifold.png", dpi=300)
plt.close()
"""