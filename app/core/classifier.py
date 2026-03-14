"""
classifier.py — Laufzeit-Klassifizierer für Commit-Messages

Lädt das trainierte Sentence-Transformer + sklearn-Modell einmalig beim Start und klassifiziert Commit-Messages in Batches. 
Fallback: Einfache RegEx-Heuristik, wenn kein Modell gefunden oder geladen werden kann
"""

import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Feature-Extraktion aus dem gemeinsamen Modul
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.features import build_features

# Standardpfade für das trainierte Modell und den Label-Encoder
MODEL_PATH   = "models/bert_classifier"
ENCODER_PATH = "models/label_encoder.pkl"

# Alle möglichen Ausgabe-Label des Klassifkators (muss mit den CATEGORIES in ml/utils.py übereinstimmen)
CATEGORIES = ["feature", "bugfix", "documentation", "refactor", "test", "chore"]

class ModelNotTrainedError(Exception):
    """Wird geworfen, wenn kein trainiertes Modell gefunden wurde."""
    pass

class Classifier:
    """
    Kapselt das trainierte NLP-Modell und stellt eine einheitliche Klassifizierungs-API bereit.
    Vor der ersten Verwendung muss load() aufgerufen werden, ist kein Modell vorhanden, wird eine ModelNotTrainedError geworfen.
    """
    def __init__(self):
        self._loaded: bool = False # True nach erfolgreichem load()
        self._st_model = None # SentenceTransformer-Modell für die Feature-Extraktion
        self._clf      = None # Trainierter sklearn-Klassifikator
        self._le       = None # Label-Encoder für Kategorie-Indizes <-> Namen

    def load(self, model_path: str = MODEL_PATH, encoder_path: str = ENCODER_PATH) -> None:
        """
        Lädt das trainierte Modell und den zugehörigen Label-Encoder.

        Raises:
            ModelNotTrainedError: Wenn kein trainiertes Modell gefunden wird.
            RuntimeError: Bei anderen Fehlern (z.B. beschädigte Dateien, Inkompatibilitäten).
        """
        if not Path(encoder_path).exists():
            raise ModelNotTrainedError(f"Kein trainiertes Modell gefunden unter '{encoder_path}'. Bitte ml/train_bert.py ausführen.")

        with open(encoder_path, "rb") as f:
            meta = pickle.load(f)

        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(model_path)
            self._clf      = meta["sklearn_classifier"]
            self._le       = meta["label_encoder"]
            self.is_loaded = True

            clf_name = meta.get("classifier_name", "sklearn")
            print(f"[Classifier] Geladen: {clf_name}")

        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden des Modells: {e}")

    @property
    def is_loaded(self) -> bool:
        """Gibt True zurück, wenn das Modell erfolgreich geladen wurde."""
        return self._loaded

    def _check_loaded(self):
        """Wirft ModelNotTrainedError, wenn das Modell nicht geladen ist."""
        if not self.is_loaded:
            raise ModelNotTrainedError("Der Klassifikator wurde noch nicht geladen. Bitte load() aufrufen oder ml/train_bert.py ausführen.")

    def classify_batch(self, messages: list[str]) -> list[tuple[str, float]]:
        """
        Klassifiziert eine Liste von Commit-Messages in einem Batch.

        Args:
            messages: Liste von Commit-Messages

        Returns:
            Liste von (Label, Konfidenz)-Tupeln in derselben Reihenfolge wie die Eingabe (messages)
        
        Raises:
            ModelNotTrainedError: Wenn kein Modell geladen ist 
        """
        self._check_loaded()

        if not messages:
            return []

        # BERT-Embeddings + Feature-Vektor berechnen, dann Klassifikator und Wahrscheinlichkeiten abrufen
        X      = build_features(messages, self._st_model)
        preds  = self._clf.predict(X)
        proba  = self._clf.predict_proba(X)
        labels = list(self._le.inverse_transform(preds))

        # Höchste Wahrscheinlichkeit als Konfidenz zurückgeben
        confs  = proba.max(axis=1).tolist()
        return list(zip(labels, confs))

    def classify_one(self, message: str) -> tuple[str, float]:
        """
        Klassifiziert eine einzelne Commit-Message. Gibt (Label, Konfidenz) zurück.
        """
        results = self.classify_batch([message])
        return results[0] if results else ("chore", 0.0)
