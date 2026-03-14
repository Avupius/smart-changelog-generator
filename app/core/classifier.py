"""
classifier.py — Runtime commit classifier.

Laedt das trainierte SentenceTransformer + sklearn Modell einmalig beim Start
und klassifiziert Commit-Messages in Batches.
Fallback: einfache Regex-Heuristik wenn kein Modell vorhanden.
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

MODEL_PATH   = "models/bert_classifier"
ENCODER_PATH = "models/label_encoder.pkl"

CATEGORIES = ["feature", "bugfix", "documentation", "refactor", "test", "chore"]

# ── Regex-Fallback ─────────────────────────────────────────────────────────────
_REGEX_RULES: list[tuple[str, re.Pattern]] = [
    ("feature",       re.compile(r"^(feat|feature|add|new|implement|introduce|support)", re.I)),
    ("bugfix",        re.compile(r"^(fix|bug|hotfix|patch|resolve|correct|repair)", re.I)),
    ("documentation", re.compile(r"^(doc|docs|documentation|readme|changelog|comment|typo)", re.I)),
    ("refactor",      re.compile(r"^(refactor|refact|clean|cleanup|rename|restructure|simplify|extract|move)", re.I)),
    ("test",          re.compile(r"^(test|tests|spec|coverage|pytest|unittest)", re.I)),
    ("chore",         re.compile(r"^(chore|build|ci|style|format|lint|deps|dependency|bump|version|release|perf)", re.I)),
]

def _regex_classify(message: str) -> tuple[str, float]:
    stripped = re.sub(r"^[a-z]+(\([^)]+\))?!?:\s*", "", message, flags=re.IGNORECASE)
    subject = (stripped or message).strip()
    for label, pattern in _REGEX_RULES:
        if pattern.match(subject):
            return label, 0.7
    return "chore", 0.3


class Classifier:
    def __init__(self):
        self._mode: Optional[str] = None
        self._st_model = None
        self._clf      = None
        self._le       = None

    def load(self, model_path: str = MODEL_PATH, encoder_path: str = ENCODER_PATH) -> None:
        if not Path(encoder_path).exists():
            print(f"[Classifier] Kein Modell gefunden ({encoder_path}). Nutze Regex-Fallback.")
            self._mode = "regex"
            return

        with open(encoder_path, "rb") as f:
            meta = pickle.load(f)

        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_path)
            self._clf      = meta["sklearn_classifier"]
            self._le       = meta["label_encoder"]
            self._mode     = "sklearn"
            clf_name = meta.get("classifier_name", "sklearn")
            print(f"[Classifier] Geladen: {clf_name}")
        except Exception as e:
            print(f"[Classifier] Ladefehler: {e}. Nutze Regex-Fallback.")
            self._mode = "regex"

    @property
    def is_loaded(self) -> bool:
        return self._mode is not None

    @property
    def mode(self) -> str:
        return self._mode or "not loaded"

    def classify_batch(self, messages: list[str]) -> list[tuple[str, float]]:
        if not messages:
            return []
        if self._mode == "regex":
            return [_regex_classify(m) for m in messages]
        # sklearn — verwendet ml.features.build_features
        X      = build_features(messages, self._st_model)
        preds  = self._clf.predict(X)
        proba  = self._clf.predict_proba(X)
        labels = list(self._le.inverse_transform(preds))
        confs  = proba.max(axis=1).tolist()
        return list(zip(labels, confs))

    def classify_one(self, message: str) -> tuple[str, float]:
        results = self.classify_batch([message])
        return results[0] if results else ("chore", 0.0)
