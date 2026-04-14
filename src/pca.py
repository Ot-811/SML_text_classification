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

    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
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