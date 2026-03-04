from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name: str = "fine_tuned_model", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_texts(self, texts, batch_size: int = 32):
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        return embeddings

    def encode_query(self, query: str):
        return self.model.encode([query], convert_to_tensor=True)
