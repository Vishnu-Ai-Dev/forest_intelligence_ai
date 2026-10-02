from typing import List

class EmbeddingInterface:
    """
    Interface for semantic embeddings.
    Kept lightweight for the hackathon prototype.
    In the future, this can be swapped with sentence-transformers or OpenAI embeddings.
    """
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement embed_text")

class SimpleMockEmbedder(EmbeddingInterface):
    """
    A placeholder embedder that uses a very simple statistical/mock approach
    to avoid downloading heavy ML models unnecessarily during the hackathon.
    """
    def embed_text(self, text: str) -> List[float]:
        # Generate a deterministic mock vector based on string length and basic hashing
        # This is purely to satisfy the interface requirements without heavy dependencies.
        val = float(len(text) % 10) / 10.0
        return [val, val * 0.5, 1.0 - val]

# Singleton instance for the pipeline to use
embedder = SimpleMockEmbedder()
