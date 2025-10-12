import faiss
import numpy as np
import os
import pickle

INDEX_PATH = "backend/faiss_index"
META_PATH = "backend/index_meta.pkl"


class Retriever:
    def __init__(self, dim: int = 384):
        """
        Initialize FAISS retriever.
        dim = embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # store (filename, chunk_text)

    def add_embeddings(self, embeddings: np.ndarray, metadatas: list):
        """Add embeddings and metadata to FAISS index."""
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 3):
        """Search similar documents for a given query embedding."""
        distances, indices = self.index.search(query_vector.astype("float32"), top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def save_index(self):
        """Persist index and metadata."""
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        """Load index and metadata if exists."""
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False
