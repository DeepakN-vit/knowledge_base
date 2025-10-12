from utils import load_documents, chunk_text
from embeddings import generate_embeddings
from retriever import Retriever
from llm import generate_answer
import numpy as np

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever(dim=384)
         
        if not self.retriever.load_index():
            print("No existing index found. Building new index from documents...")
            self.build_index()

    def build_index(self, doc_folder: str = "documents"):
        """Load documents, create embeddings, and store in FAISS index."""
        docs = load_documents(doc_folder)
        all_chunks = []
        metadata = []

        for filename, text in docs:
            chunks = chunk_text(text, chunk_size=500, overlap=100)
            all_chunks.extend(chunks)
            metadata.extend([(filename, chunk) for chunk in chunks])

        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = generate_embeddings(all_chunks)
        self.retriever.add_embeddings(embeddings, metadata)
        self.retriever.save_index()
        print("Index built and saved successfully.")

    def query(self, user_query: str, top_k: int = 3) -> str:
        """Retrieve relevant chunks and generate a detailed answer."""
        query_embedding = generate_embeddings([user_query])
        retrieved_chunks = self.retriever.search(query_embedding, top_k=top_k)

        if not retrieved_chunks:
            return "No relevant information found in the knowledge base."

      
        context = "\n\n".join([chunk[1] for chunk in retrieved_chunks])
 
        answer = generate_answer(context=context, query=user_query)
        return answer
