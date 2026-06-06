"""
search.py
---------
Search tool for the LLM agent.
Queries a local knowledge base using semantic search.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SearchTool:
    """
    Semantic search over a local knowledge base.
    Add facts to the knowledge base and query them semantically.
    """

    name = "search"
    description = (
        "Search a local knowledge base. "
        "To add: 'add: fact or document text'. "
        "To search: 'search: your query'."
    )

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self._embeddings = []

    def _rebuild_index(self) -> None:
        if not self._embeddings:
            return
        matrix = np.array(self._embeddings).astype(np.float32)
        self.index = faiss.IndexFlatL2(matrix.shape[1])
        self.index.add(matrix)

    def run(self, command: str) -> str:
        """
        Execute a search command.

        Args:
            command: 'add: text' or 'search: query'

        Returns:
            Result as string
        """
        command = command.strip()

        if command.lower().startswith("add:"):
            text = command[4:].strip()
            if not text:
                return "No text provided."
            embedding = self.model.encode([text])[0]
            self.documents.append(text)
            self._embeddings.append(embedding)
            self._rebuild_index()
            return f"Added to knowledge base. Total documents: {len(self.documents)}"

        elif command.lower().startswith("search:"):
            if self.index is None or len(self.documents) == 0:
                return "Knowledge base is empty. Use 'add: text' first."
            query = command[7:].strip()
            query_embedding = self.model.encode([query]).astype(np.float32)
            top_k = min(3, len(self.documents))
            distances, indices = self.index.search(query_embedding, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                score = float(1 / (1 + dist))
                results.append(f"[{score:.3f}] {self.documents[idx]}")

            return "Search results:\n" + "\n".join(results)

        elif command.lower() == "list":
            if not self.documents:
                return "Knowledge base is empty."
            return "\n".join(f"{i+1}. {doc[:80]}" for i, doc in enumerate(self.documents))

        else:
            return "Unknown command. Use: 'add: text', 'search: query', or 'list'"

    @property
    def size(self) -> int:
        return len(self.documents)


if __name__ == "__main__":
    tool = SearchTool()
    tool.run("add: Python is a high-level programming language")
    tool.run("add: Machine learning uses algorithms to learn from data")
    tool.run("add: FAISS is a library for efficient similarity search")
    print(tool.run("search: how to search vectors efficiently"))