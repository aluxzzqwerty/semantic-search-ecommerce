import faiss
import torch

class SemanticSearch:
    def __init__(self, embeddings: torch.Tensor):
        """
        embeddings: torch tensor of shape (n_samples, dim)
        """
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings.cpu().numpy())

    def search(self, query_embedding: torch.Tensor, top_k: int = 5):
        """
        Perform vector search.
        Returns indices and distances.
        """
        query_np = query_embedding.cpu().numpy()
        distances, indices = self.index.search(query_np, top_k)

        return distances[0], indices[0]
