"""
features.py — Gemeinsame Feature-Extraktion fuer Training, Inference und Evaluation.

Wird importiert von:
  - ml/train_bert.py
  - ml/evaluate.py
  - app/core/classifier.py

Alle drei muessen identische Features erzeugen!
"""

import re

import numpy as np

from ml.utils import CATEGORIES

# ── Conventional-Commit-Prefix Mapping ───────────────────────────────────────
PREFIX_MAP = {
    "feat": "feature", "feature": "feature",
    "fix": "bugfix", "bugfix": "bugfix", "hotfix": "bugfix", "bug": "bugfix",
    "docs": "documentation", "doc": "documentation", "documentation": "documentation",
    "refactor": "refactor", "refact": "refactor",
    "test": "test", "tests": "test",
    "chore": "chore", "build": "chore", "ci": "chore",
    "style": "chore", "perf": "chore",
}

_PREFIX_RE = re.compile(r"^([a-z]+)(\([^)]+\))?!?\s*:\s*", re.IGNORECASE)
_CAT2IDX   = {c: i for i, c in enumerate(CATEGORIES)}

# ── Keyword-Features: diskriminative Woerter pro Kategorie ───────────────────
KEYWORD_GROUPS = {
    "feature":       ["add", "new", "implement", "create", "introduce", "support",
                      "enable", "allow", "build", "integrate", "initial", "init",
                      "expose", "provide", "include", "generate"],
    "bugfix":        ["fix", "bug", "resolve", "patch", "correct", "repair",
                      "handle", "prevent", "avoid", "regression", "crash",
                      "error", "issue", "problem", "wrong", "broken"],
    "documentation": ["doc", "docs", "readme", "comment", "typo", "spelling",
                      "example", "guide", "tutorial", "wiki", "changelog",
                      "clarify", "describe", "explain", "note"],
    "refactor":      ["refactor", "clean", "cleanup", "rename", "restructure",
                      "reorganize", "simplify", "extract", "split", "move",
                      "migrate", "rewrite", "decouple", "reduce", "consolidate",
                      "separate", "improve", "optimize", "replace", "reuse"],
    "test":          ["test", "spec", "coverage", "assert", "mock", "fixture",
                      "unit", "integration", "e2e", "scenario", "case"],
    "chore":         ["bump", "upgrade", "version", "release", "deploy", "ci",
                      "lint", "format", "style", "dependency", "deps", "build",
                      "config", "setup", "install", "pipeline", "workflow"],
}

ALL_KEYWORDS = [kw for kws in KEYWORD_GROUPS.values() for kw in kws]
_KW_INDEX    = {kw: i for i, kw in enumerate(ALL_KEYWORDS)}
_KW_RE       = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ALL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Gewichtungen — muessen in Training UND Inference gleich sein
PREFIX_WEIGHT  = 10.0
KEYWORD_WEIGHT = 3.0


def strip_prefix(msg: str) -> str:
    """Entfernt den Conventional-Commit-Prefix damit BERT nur den Inhalt sieht."""
    stripped = _PREFIX_RE.sub("", msg.strip())
    return stripped.strip() or msg.strip()


def extract_prefix_features(messages: list[str]) -> np.ndarray:
    """One-Hot fuer Conventional-Commit-Prefix. Dim: 6 + 1 ('kein Prefix')."""
    n = len(CATEGORIES)
    out = np.zeros((len(messages), n + 1), dtype=np.float32)
    for i, msg in enumerate(messages):
        m = _PREFIX_RE.match(msg.strip())
        if m:
            prefix = m.group(1).lower()
            cat    = PREFIX_MAP.get(prefix)
            if cat and cat in _CAT2IDX:
                out[i, _CAT2IDX[cat]] = 1.0
            else:
                out[i, n] = 1.0
        else:
            out[i, n] = 1.0
    return out


def extract_keyword_features(messages: list[str]) -> np.ndarray:
    """Binaerer Keyword-Vektor — welche diskriminativen Woerter kommen vor?"""
    out = np.zeros((len(messages), len(ALL_KEYWORDS)), dtype=np.float32)
    for i, msg in enumerate(messages):
        text = strip_prefix(msg).lower()
        for match in _KW_RE.finditer(text):
            kw = match.group(1).lower()
            if kw in _KW_INDEX:
                out[i, _KW_INDEX[kw]] = 1.0
    return out


def build_features(messages: list[str], st_model, batch_size: int = 64) -> np.ndarray:
    """
    Kombinierte Features:
      BERT-Embedding (ohne Prefix)  +  Prefix-One-Hot (x10)  +  Keyword-Vektor (x3)
    """
    stripped   = [strip_prefix(m) for m in messages]
    embeddings = st_model.encode(stripped, batch_size=batch_size, show_progress_bar=True)
    prefix_f   = extract_prefix_features(messages)
    keyword_f  = extract_keyword_features(messages)
    return np.hstack([embeddings, prefix_f * PREFIX_WEIGHT, keyword_f * KEYWORD_WEIGHT])
