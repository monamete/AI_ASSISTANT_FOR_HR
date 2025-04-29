# build_faiss_index.py

import faiss
import numpy as np
import pickle
from create_embeddings import create_embeddings  
import os


all_texts = [
    "Leave policy for employees",
    "Work from home guidelines",
    "Performance appraisal process",
    "Health insurance benefits",
    "Employee grievance redressal procedure"
]


embeddings = create_embeddings(all_texts)  

dimension = embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)  


index.add(embeddings)

faiss.write_index(index, "faiss_index.idx")
print("FAISS index saved as faiss_index.idx")


with open("all_texts.pkl", "wb") as f:
    pickle.dump(all_texts, f)
print("All texts saved as all_texts.pkl")