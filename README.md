# Text Classification using Classical Machine Learning  
## Sparse Features vs Contextual Embeddings

---

## Overview

This project compares classical machine learning models for sentiment classification using two types of feature representations:

- TF-IDF (Sparse Features)
- Contextual Embeddings generated using DistilBERT

The objective is to analyze how different feature representations affect classification performance on textual data.

---

## Project Structure

.
├── src/
│   ├── preprocessing.py
│   ├── tfidf.py
│   ├── embeddings.py
│   ├── models.py
│   ├── evaluate.py
│   ├── pca.py
│   ├── visualize.py
│
├── results/
│   ├── plots/
│   ├── embeddings_train.npz
│   ├── embeddings_test.npz
│
├── main.py
├── requirements.txt
└── README.md

---

## Features

- TF-IDF feature extraction  
- Transformer-based embeddings using DistilBERT  
- Mean pooling for sentence embeddings  
- Classical machine learning models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - KMeans Clustering  
- Principal Component Analysis (PCA)  
- t-SNE visualization  
- Embedding caching using NPZ format  
- Batch processing with optional GPU support  
- Modular pipeline design  

---

## Dataset

- Dataset: IMDb (via Hugging Face)  
- Task: Binary Sentiment Classification  
- Samples used: 2000 (subset for efficiency)  

---

## Installation

Install dependencies using:

pip install -r requirements.txt

---

## Usage

Run TF-IDF pipeline:

python main.py --mode tfidf

Run embedding pipeline:

python main.py --mode embedding

Generate t-SNE visualization:

python main.py --mode visualize

---

## Results

| Feature    | Model     | Accuracy | F1     |
|------------|----------|----------|--------|
| TF-IDF     | Logistic | 0.8756   | 0.8766 |
| TF-IDF     | SVM      | 0.8618   | 0.8617 |
| TF-IDF     | KNN      | 0.6775   | 0.6800 |
| TF-IDF     | KMeans   | 0.5762   | -      |
| Embeddings | Logistic | 0.8105   | 0.8125 |
| Embeddings | SVM      | 0.7895   | 0.7927 |
| Embeddings | KNN      | 0.7250   | 0.6998 |
| Embeddings | KMeans   | 0.5410   | -      |

---

## Key Observations

- TF-IDF features outperform embedding-based features for this sentiment classification task  
- Embeddings capture semantic relationships but require fine-tuning for optimal performance  
- KNN performs poorly with TF-IDF due to high dimensionality  
- KMeans performs worse as it is an unsupervised algorithm  
- PCA reduces dimensionality and noise, but excessive reduction reduces performance  
- t-SNE visualization shows improved clustering for embedding-based features  

---

## Notes

- KMeans is an unsupervised algorithm; F1-score is not directly applicable  
- Embeddings are generated without fine-tuning  
- GPU significantly improves embedding extraction speed  

---

## Learnings

- Importance of feature representation in NLP  
- Trade-offs between sparse and dense features  
- Implementation of end-to-end ML pipelines  
- Use of dimensionality reduction and visualization techniques  

---

## Author

Om Tripathi  
Roll Number: BA2025027  
Batch: BT DSAI  

---

## License

This project is intended for academic use only.
