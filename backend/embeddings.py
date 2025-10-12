from sentence_transformers import SentenceTransformer
import numpy as np

 
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

def generate_embeddings(chunks: list) -> np.ndarray:
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

