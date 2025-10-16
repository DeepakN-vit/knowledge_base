
---

#  Knowledge-Base Search Engine (RAG + LLM)

A **Retrieval-Augmented Generation (RAG)**-based **Knowledge-Base Search Engine** that lets you **upload multiple documents (PDF/TXT)** and **ask natural-language questions**.
The system retrieves relevant document chunks using **semantic embeddings** and generates **concise, context-aware answers** with a **Local/Free LLM** — all with a simple **Streamlit UI**.

---

##  Features

*  Upload multiple text/PDF documents
*  Search and retrieve semantically similar text chunks
*  Generate synthesized answers using a free local LLM (e.g., *google/flan-t5-base* or *distilbart-cnn-12-6*)
*  RAG (Retrieval-Augmented Generation) pipeline for high-quality responses
*  FAISS-based vector store for efficient retrieval
*  Interactive frontend built with **Streamlit**
*  Modular backend using **FastAPI**

---

##  Project Structure

```
knowledge_base_rag/
│
├── backend/
│   ├── main.py                 # FastAPI backend
│   ├── rag_pipeline.py         # Core RAG pipeline
│   ├── retriever.py            # FAISS retriever
│   ├── embeddings.py           # Embedding generation
│   ├── llm.py                  # LLM answer generation
│   ├── utils.py                # PDF/text loader & chunking
│   ├── requirements.txt        # Python dependencies
│   ├── __init__.py
│   └── documents/              # Folder for uploaded docs
│
├── frontend/
│   ├── app.py                  # Streamlit UI
│   └── __init__.py
│
└── README.md                   # Project documentation
```

---

##  Installation & Setup

### 1️ Clone the repository

```bash
git clone https://github.com/yourusername/knowledge_base_rag.git
cd knowledge_base_rag
```

### 2️ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # (Linux/Mac)
venv\Scripts\activate           # (Windows)
```

### 3️ Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4️ Run the backend (FastAPI)

```bash
cd backend
uvicorn main:app --reload
```

Backend will start at → `http://127.0.0.1:8000`

### 5️ Run the frontend (Streamlit)

```bash
cd ../frontend
streamlit run app.py
```

Frontend will start at → `http://localhost:8501`

---

##  How It Works

1. **Document Upload:**
   Upload multiple `.txt` or `.pdf` files through the Streamlit UI.
2. **Document Indexing:**
   Documents are split into small chunks, converted into embeddings, and stored in a **FAISS index**.
3. **Query:**
   You enter a natural language question.
4. **Retrieval:**
   The retriever searches for the most relevant chunks across all documents.
5. **Answer Generation:**
   The retrieved context is passed to the **LLM** (e.g., Flan-T5) which generates a synthesized answer.

---

##  Example Queries

| Example Question                        | Answer Type      |
| --------------------------------------- | ---------------- |
| What is Artificial Intelligence?        | Definition       |
| How is ML different from AI?            | Comparison       |
| Who is mentioned in person.txt?         | Entity retrieval |
| What are the types of Machine Learning? | Enumerated list  |

---

##  Technologies Used

* **FastAPI** — Backend API
* **Streamlit** — Frontend interface
* **SentenceTransformers** — For embeddings
* **FAISS** — Vector similarity search
* **Transformers (Hugging Face)** — Free LLMs (e.g., Flan-T5, DistilBart)
* **PyPDF2** — For PDF text extraction
* **NumPy / Pandas** — Data processing

---

##  Deliverables

*  GitHub repository with clean, modular code
*  README documentation (this file)
*  Demo video showing end-to-end working

---

##  Evaluation Focus

* 🔹 Retrieval accuracy (semantic relevance of chunks)
* 🔹 Quality of synthesized answers
* 🔹 Code structure & readability
* 🔹 LLM integration efficiency

---

##  Example LLM Prompt

```
"Using the following document context, answer the user’s question succinctly."
```

---

##  Author

**Deepak N**
Python & Machine Learning Developer | VIT-AP
