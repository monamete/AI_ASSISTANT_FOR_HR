# search_faiss.py

import numpy as np
import faiss
from create_embeddings import create_embeddings  # <- plain import, works fine
import os

def load_embeddings(filename=r"embeddings.npy"):
    """
    Load saved embeddings from a .npy file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    return np.load(filename)

def create_faiss_index(embeddings):
    """
    Create and return a FAISS index from embeddings.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_document(query, index, all_texts, top_k=3):
    """
    Search for top_k most similar documents to the query.
    """
    query_embedding = create_embeddings([query])
    _, indices = index.search(query_embedding, top_k)
    results = [all_texts[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    # Load original texts and embeddings
    texts = [
        "This is a sample document.",
        "FAISS is an efficient similarity search library.",
        "Embeddings are important for semantic search."
    ]
    
    embeddings = load_embeddings()
    
    # Create a FAISS index
    index = create_faiss_index(embeddings)
    
    # Example query
    query = "What is FAISS?"
    results = search_document(query, index, texts, top_k=2)
    
    print("Top search results:")
    for res in results:
        print("-", res)



