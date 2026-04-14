import argparse
from datasets import load_dataset
import random
from src.preprocessing import clean_text
from src.tfidf import get_tfidf_features
from src.models import train_logistic, train_svm
from src.evaluate import evaluate_model
from src.embeddings import get_embeddings
from src.models import train_knn
from src.models import train_kmeans
from src.evaluate import evaluate_kmeans
from src.pca import plot_pca_2d, plot_explained_variance
from src.pca import accuracy_vs_pca
import numpy as np

def run_tfidf_pipeline():
    dataset = load_dataset("imdb")

    train_data = list(dataset["train"])
    test_data = list(dataset["test"])

    random.seed(42)

    random.shuffle(train_data)
    random.shuffle(test_data)

    train_texts = [clean_text(x["text"]) for x in train_data]
    test_texts = [clean_text(x["text"]) for x in test_data]

    y_train = [x["label"] for x in train_data]
    y_test = [x["label"] for x in test_data]

    X_train, X_test = get_tfidf_features(train_texts, test_texts)

    model_lr = train_logistic(X_train, y_train)
    acc, f1 = evaluate_model(model_lr, X_test, y_test)
    print(f"Logistic Regression → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    model_svm = train_svm(X_train, y_train)
    acc, f1 = evaluate_model(model_svm, X_test, y_test)
    print(f"SVM → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    model_knn = train_knn(X_train, y_train)
    acc, f1 = evaluate_model(model_knn, X_test, y_test)
    print(f"KNN → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    kmeans = train_kmeans(X_train)
    acc = evaluate_kmeans(kmeans, X_test, y_test)
    print(f"KMeans → Accuracy: {acc:.4f}")

    print("PCA:")

    X_small = X_train[:2000].toarray()
    y_small = y_train[:2000]

    plot_pca_2d(
        X_small,
        y_small,
        "TF-IDF PCA (2D)",
        "results/plots/tfidf_pca.png"
    )

    plot_explained_variance(
        X_small,
        "results/plots/tfidf_variance.png"
    )

    accuracy_vs_pca(
        X_small,
        y_small,
        train_logistic,
        dims_list=[50, 100, 200, 300],
        title="TF-IDF: Accuracy vs PCA Dimensions",
        save_path="results/plots/tfidf_acc_pca.png"
    )

    

def run_embedding_pipeline():

    dataset = load_dataset("imdb")

    train_data = list(dataset["train"])
    test_data = list(dataset["test"])

    random.shuffle(train_data)
    random.shuffle(test_data)

    train_texts = [clean_text(x["text"]) for x in train_data]
    test_texts = [clean_text(x["text"]) for x in test_data]

    y_train = [x["label"] for x in train_data]
    y_test = [x["label"] for x in test_data]

    print("Extracting embedding (takes time)...")

    X_train = get_embeddings(train_texts, max_samples=2000)
    X_test = get_embeddings(test_texts, max_samples=2000)

    print("Embeddings extracted. Evaluating models...")

    y_train = y_train[:2000]
    y_test = y_test[:2000]

    model_lr = train_logistic(X_train, y_train)
    acc, f1 = evaluate_model(model_lr, X_test, y_test)
    print(f"Embeddings + Logistic → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    model_svm = train_svm(X_train, y_train)
    acc, f1 = evaluate_model(model_svm, X_test, y_test)
    print(f"Embeddings + SVM → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    model_knn = train_knn(X_train, y_train)
    acc, f1 = evaluate_model(model_knn, X_test, y_test)
    print(f"Embeddings + KNN → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    kmeans = train_kmeans(X_train)
    acc = evaluate_kmeans(kmeans, X_test, y_test)
    print(f"Embeddings + KMeans → Accuracy: {acc:.4f}")

    print("PCA:")

    plot_pca_2d(
        X_train,
        y_train,
        "Embeddings PCA (2D)",
        "results/plots/embedding_pca.png"
    )
    plot_explained_variance(
        X_train,
        "results/plots/embedding_variance.png"
    )

    X_small = X_train[:2000]
    y_small = y_train[:2000]

    accuracy_vs_pca(
        X_small,
        y_small,
        train_logistic,
        dims_list=[50, 100, 200, 300],
        title="TF-IDF: Accuracy vs PCA Dimensions",
        save_path="results/plots/tfidf_acc_pca.png"
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tfidf", "embedding"], required=True)

    args = parser.parse_args()

    if args.mode == "tfidf":
        run_tfidf_pipeline()
    elif args.mode == "embedding":
        run_embedding_pipeline()