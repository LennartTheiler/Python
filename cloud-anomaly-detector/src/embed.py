from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Vortrainiertes Modell
_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def embed(texts: List[str]) -> np.ndarray:
    """
    Wandelt Liste von Texten in embedding-Vektoren um. Diese Embeddings erfassen die 
    semantische Bedeutung der Eingabetexte, sodass ähnliche Bedeutungen in ähnlichen 
    Vektoren resultieren.

    Verwendet wird ein vortrainiertes Transformer-Modell ('all-MiniLM-L6-v2'), das auf
    der Transformer-Architektur basiert. Transformer sind wichtig, da sie:
    - Kontextinformationen effizient durch Selbstaufmerksamkeit (Self-Attention) 
      modellieren,
    - lange Abhängigkeiten in Texten erfassen können.

    Args:
        texts (List[str]): Liste von Eingabetexten.
    
    Returns:
        np.ndarray: 2D-Array mit Embeddings der Form (n_samples, n_features),
                    wobei jeder Zeile einem Texteingang entspricht.
    """
    embeddings = _MODEL.encode(texts, show_progress_bar=True)
    return embeddings
