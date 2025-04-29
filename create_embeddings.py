#create_embeddings.py

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os


model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def create_embeddings(texts):
    """
    Convert a list of texts into embeddings using a transformer model.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

def save_embeddings(embeddings, filename="embeddings.npy"):
    """
    Save embeddings to a .npy file.
    """
    np.save(filename, embeddings)
    print(f"Embeddings saved to {filename}")

if __name__ == "__main__":
   
    texts = ["This is a sample document.", "FAISS is an efficient search library.", "Embeddings are great for similarity search."]
    
 
    embeddings = create_embeddings(texts)

    save_embeddings(embeddings)
