# build_faiss_index.py

import faiss
import numpy as np
import pickle
from create_embeddings import create_embeddings  # your function
import os

# Step 1: Load your HR Policy documents (text data)
# For demo, let's assume it's a list. Replace this with your real data loading.
all_texts = [
    "Leave policy for employees",
    "Work from home guidelines",
    "Performance appraisal process",
    "Health insurance benefits",
    "Employee grievance redressal procedure"
]

# Step 2: Create embeddings
embeddings = create_embeddings(all_texts)  # Should be (num_texts x embedding_dim)

# Step 3: Create a FAISS index
dimension = embeddings.shape[1]  # Embedding size (e.g., 384 for MiniLM)
index = faiss.IndexFlatL2(dimension)  # Use L2 (Euclidean) similarity

# Step 4: Add embeddings to the index
index.add(embeddings)

# Step 5: Save the FAISS index
faiss.write_index(index, "faiss_index.idx")
print("✅ FAISS index saved as faiss_index.idx")

# Step 6: Save the list of texts
with open("all_texts.pkl", "wb") as f:
    pickle.dump(all_texts, f)
print("✅ All texts saved as all_texts.pkl")
