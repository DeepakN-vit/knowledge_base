import os
import PyPDF2

def read_pdf(file_path: str) -> str:
    """Extract text content from a PDF file."""
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def read_text(file_path: str) -> str:
    """Read content from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents(folder_path: str = "documents") -> list:
    """
    Load and return text from all PDF and TXT files in the given folder.
    Returns a list of tuples: (filename, text_content)
    """
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            text = read_pdf(file_path)
        elif filename.lower().endswith(".txt"):
            text = read_text(file_path)
        else:
            continue  # Skip unsupported files
        if text.strip():
            docs.append((filename, clean_text(text)))
    return docs


def clean_text(text: str) -> str:
    """Basic cleaning of extracted text."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # remove extra spaces
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Split text into chunks for embedding.
    Example: chunk_size=500 tokens, overlap=100 to preserve context.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
