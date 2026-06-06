"""
document.py
-----------
Document tool for the LLM agent.
Loads and queries documents using RAG.
"""

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


class DocumentTool:
    """
    RAG-based document tool.
    Load a document and answer questions from it.
    """

    name = "document"
    description = (
        "Load and query documents. "
        "To load: 'load: /path/to/file'. "
        "To query: 'query: your question'."
    )

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.loaded_file = None

    def _load_txt(self, path: str) -> str:
        return Path(path).read_text(encoding='utf-8')

    def _load_pdf(self, path: str) -> str:
        try:
            import PyPDF2
            text = ""
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text
        except ImportError:
            raise ImportError("pip install pypdf2 to load PDF files")

    def _chunk(self, text: str, sentences_per_chunk: int = 3) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            if len(chunk.strip()) > 30:
                chunks.append(chunk.strip())
        return chunks

    def _build_index(self, chunks: list) -> None:
        self.chunks = chunks
        embeddings = self.model.encode(chunks).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def run(self, command: str) -> str:
        """
        Execute a document command.

        Args:
            command: 'load: path' or 'query: question'

        Returns:
            Result as string
        """
        command = command.strip()

        if command.lower().startswith("load:"):
            path = command[5:].strip()
            try:
                if path.endswith('.pdf'):
                    text = self._load_pdf(path)
                else:
                    text = self._load_txt(path)
                chunks = self._chunk(text)
                self._build_index(chunks)
                self.loaded_file = path
                return f"Loaded: {path} ({len(chunks)} chunks indexed)"
            except Exception as e:
                return f"Error loading document: {str(e)}"

        elif command.lower().startswith("query:"):
            if self.index is None:
                return "No document loaded. Use 'load: path' first."
            question = command[6:].strip()
            query_embedding = self.model.encode([question]).astype(np.float32)
            _, indices = self.index.search(query_embedding, 3)
            context = '\n'.join([self.chunks[i] for i in indices[0]])
            return f"Relevant context:\n{context}"

        else:
            return "Unknown command. Use: 'load: path' or 'query: question'"