import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def plot_pca_2d(X, y, title, save_path):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=y,
        alpha=0.5,       # transparency
        s=10,            # smaller points
        cmap='coolwarm'  # better colors
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Class")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_explained_variance(X, save_path):
    pca = PCA().fit(X)

    plt.figure()
    plt.plot(pca.explained_variance_ratio_)
    plt.title("Explained Variance Ratio")
    plt.xlabel("Components")
    plt.ylabel("Variance")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()