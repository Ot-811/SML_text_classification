from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# load model once
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_embeddings(texts, max_samples=2000):
    embeddings = []

    for text in texts[:max_samples]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state

        # MEAN POOLING
        mean_embedding = last_hidden.mean(dim=1).squeeze().numpy()

        embeddings.append(mean_embedding)

    return np.array(embeddings)