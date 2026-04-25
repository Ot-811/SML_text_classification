import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Embedding device: {device}")

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings(texts, batch_size=32, max_length=128, mode="mean"):
    embeddings = []

    print("[INFO] Generating embeddings...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state
        if mode == "mean":
            pooled = mean_pooling(last_hidden, inputs["attention_mask"])
        elif mode == "cls":
            pooled = last_hidden[:, 0, :]

        embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(embeddings)

    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    return embeddings