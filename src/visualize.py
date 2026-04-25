import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



def sample_data(X, labels, max_samples=2000):
    if len(X) > max_samples:
        idx = np.random.choice(len(X), size=max_samples, replace=False)
        return X[idx], np.array(labels)[idx]
    return X, labels


def apply_pca(X, n_components=50):
    print("[INFO] Applying PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def run_tsne(X, perplexity=30, n_iter=1000):
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        verbose=1
    )
    return tsne.fit_transform(X)


def plot_tsne(X_2d, labels, output_path="tsne.png", title="t-SNE Visualization"):
    print("Plotting t-SNE...")

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=15,
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter)
    plt.grid(True)

    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved plot → {output_path}")

    plt.show()


def generate_tsne_plot(X, labels, output_path, use_pca=True, max_samples=2000):

    print("t-SNE")

    if hasattr(X, "toarray"):
        X = X.toarray()

    X, labels = sample_data(X, labels, max_samples)

    if use_pca:
        X = apply_pca(X)

    X_2d = run_tsne(X)

    plot_tsne(X_2d, labels, output_path)



def main():
    parser = argparse.ArgumentParser(description="t-SNE Visualization")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .npz")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels .npy")
    parser.add_argument("--output", type=str, default="tsne.png", help="Output image path")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--no_pca", action="store_true")

    args = parser.parse_args()

    print("[INFO] Loading data...")
    X = np.load(args.embeddings)["X"]
    labels = np.load(args.labels)

    generate_tsne_plot(
        X,
        labels,
        args.output,
        use_pca=not args.no_pca,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()