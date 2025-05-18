from pyod.models.iforest import IForest
import numpy as np
from typing import Tuple

def train_detector(embeddings: np.ndarray,
                   contamination: float = 0.01) -> IForest:
    """
    Trainiert einen IsolationForest-Detektor.
    contamination: erwarteter Anteil der Ausreißer.
    """
    clf = IForest(contamination=contamination)
    clf.fit(embeddings)
    return clf

def predict(clf: IForest,
            embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Liefert (AnomalyScore, Label), Label=1 für Outlier.
    """
    scores = clf.decision_function(embeddings)
    labels = clf.predict(embeddings)
    return scores, labels
