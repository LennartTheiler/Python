from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Vortrainiertes Modell
_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def embed(texts: List[str]) -> np.ndarray:
    """
    Wandelt Liste von Texten in embedding-Vektoren um.
    Gibt ein Array der Form (n_samples, n_features).
    """
    embeddings = _MODEL.encode(texts, show_progress_bar=True)
    return embeddings
