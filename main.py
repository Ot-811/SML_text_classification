import argparse
from datasets import load_dataset

from src.preprocessing import clean_text
from src.tfidf import get_tfidf_features
from src.models import train_logistic, train_svm
from src.evaluate import evaluate_model

def run_tfidf_pipeline():
    print("Loading dataset...")
    dataset = load_dataset("imdb")

    train_data = dataset["train"]
    test_data = dataset["test"]

    train_texts = [clean_text(x["text"]) for x in train_data]
    test_texts = [clean_text(x["text"]) for x in test_data]

    y_train = [x["label"] for x in train_data]
    y_test = [x["label"] for x in test_data]

    print("Extracting TF-IDF features...")
    X_train, X_test = get_tfidf_features(train_texts, test_texts)

    print("Training Logistic Regression...")
    model_lr = train_logistic(X_train, y_train)
    acc, f1 = evaluate_model(model_lr, X_test, y_test)
    print(f"Logistic Regression → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    print("Training SVM...")
    model_svm = train_svm(X_train, y_train)
    acc, f1 = evaluate_model(model_svm, X_test, y_test)
    print(f"SVM → Accuracy: {acc:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tfidf"], required=True)

    args = parser.parse_args()

    if args.mode == "tfidf":
        run_tfidf_pipeline()