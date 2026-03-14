import argparse
import pickle
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

"""
train_bert.py — Phase 2 of the NLP pipeline.

SentenceTransformer + optimierter sklearn Klassifikator.

Verbesserungen gegenüber Basisversion:
  1. Conventional-Commit-Prefix als explizites One-Hot-Feature (stärkster Hebel)
  2. class_weight='balanced' - gegen Klassenungleichgewicht (test << chore)
  3. Mehrere Klassifikatoren werden verglichen (LR, LinearSVC, SGD)
  4. GridSearchCV für optimale Hyperparameter
  5. Feature-Kombination: BERT-Embedding + Prefix-Features

Usage:
    python ml/train_bert.py
    python ml/train_bert.py --model all-mpnet-base-v2   # größeres Modell
    python ml/train_bert.py --no-grid                    # kein GridSearch (schneller)
"""


sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.features import build_features, ALL_KEYWORDS, CATEGORIES as FEAT_CATEGORIES
from ml.utils import CATEGORIES, load_jsonl, save_jsonl, stratified_split

# DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_DATA = "data/labeled/commits_labeled.jsonl"
DEFAULT_OUT = "models/bert_classifier"
ENCODER_OUT = "models/label_encoder.pkl"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_distribution(name: str, data: list[dict]) -> None:
    """Gibt die Klassenverteilung eines Datensatzes als ASCII-Balkendiagramm aus."""
    dist = Counter(d["label"] for d in data)
    print(f"\n{name} ({len(data)} samples):")
    for cat in CATEGORIES:
        count = dist.get(cat, 0)
        bar = "#" * (count // max(len(data) // 40, 1))
        print(f"  {cat:<15} {count:>5}  {bar}")


def print_val_metrics(y_true, y_pred, name: str) -> dict:
    """
    Berechnet und gibt Klassifikationsmetriken aus.

    Returns:
        dict mit 'accuracy', 'f1_macro', 'f1_weighted'
    """
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n  [{name}] Accuracy={acc:.4f}  F1-macro={f1m:.4f}  F1-weighted={f1w:.4f}  "
          f"Precision={prec:.4f}  Recall={rec:.4f}")
    return {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w}


# ─── Classifier Comparison ────────────────────────────────────────────────────

def compare_classifiers(X_train, y_train, X_val, y_val, le, use_grid: bool) -> object:
    """
    Trainiert mehrere Klassifikatoren, vergleicht Val-F1 und gibt den besten zurück.
    """

    val_labels = le.inverse_transform(y_val)
    candidates = {}

    # ── 1. LogisticRegression ──────────────────────────────────────────────────
    print("\n  Trainiere LogisticRegression...")
    if use_grid:
        grid = GridSearchCV(
            LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
            param_grid={"C": [0.1, 0.5, 1.0, 5.0, 10.0]},
            cv=3, scoring="f1_macro", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        clf_lr = grid.best_estimator_
        print(f"    Bestes C={grid.best_params_['C']}  (CV F1={grid.best_score_:.4f})")
    else:
        clf_lr = LogisticRegression(C=5.0, max_iter=2000, random_state=42, class_weight="balanced")
        clf_lr.fit(X_train, y_train)

    preds = le.inverse_transform(clf_lr.predict(X_val))
    m = print_val_metrics(val_labels, preds, "LogisticRegression")
    candidates["LogisticRegression"] = (clf_lr, m["f1_macro"])

    # ── 2. LinearSVC (calibriert für Wahrscheinlichkeiten) ────────────────────
    print("\n  Trainiere LinearSVC...")
    if use_grid:
        base_svc = LinearSVC(max_iter=3000, random_state=42, class_weight="balanced")
        grid = GridSearchCV(
            CalibratedClassifierCV(base_svc, cv=3),
            param_grid={"estimator__C": [0.01, 0.1, 0.5, 1.0]},
            cv=3, scoring="f1_macro", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        clf_svc = grid.best_estimator_
        print(f"    Bestes C={grid.best_params_['estimator__C']}  (CV F1={grid.best_score_:.4f})")
    else:
        svc = LinearSVC(C=0.5, max_iter=3000, random_state=42, class_weight="balanced")
        clf_svc = CalibratedClassifierCV(svc, cv=3)
        clf_svc.fit(X_train, y_train)

    preds = le.inverse_transform(clf_svc.predict(X_val))
    m = print_val_metrics(val_labels, preds, "LinearSVC (calibrated)")
    candidates["LinearSVC"] = (clf_svc, m["f1_macro"])

    # ── 3. SGDClassifier (schnell, gut bei großen Datensätzen) ────────────────
    print("\n  Trainiere SGDClassifier (SVM-loss)...")
    if use_grid:
        base_sgd = SGDClassifier(loss="hinge", max_iter=200, random_state=42, class_weight="balanced")
        grid = GridSearchCV(
            CalibratedClassifierCV(base_sgd, cv=3),
            param_grid={"estimator__alpha": [1e-5, 1e-4, 1e-3]},
            cv=3, scoring="f1_macro", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        clf_sgd = grid.best_estimator_
        print(f"    Bestes alpha={grid.best_params_['estimator__alpha']}  (CV F1={grid.best_score_:.4f})")
    else:
        sgd = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=200, random_state=42, class_weight="balanced")
        clf_sgd = CalibratedClassifierCV(sgd, cv=3)
        clf_sgd.fit(X_train, y_train)

    preds = le.inverse_transform(clf_sgd.predict(X_val))
    m = print_val_metrics(val_labels, preds, "SGDClassifier")
    candidates["SGDClassifier"] = (clf_sgd, m["f1_macro"])

    # ── Besten wählen ──────────────────────────────────────────────────────────
    best_name = max(candidates, key=lambda k: candidates[k][1])
    best_clf, best_f1 = candidates[best_name]
    print(f"\n  Bester Klassifikator: {best_name} (Val F1-macro={best_f1:.4f})")
    return best_clf, best_name


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    """
    Hauptpipeline: Daten laden → Embeddings erzeugen → Klassifikatoren vergleichen → Modell speichern.

    Ablauf:
      1. JSONL-Daten laden und stratifiziert splitten (train/val/test)
      2. SentenceTransformer-Embeddings + Prefix-Features berechnen
      3. Drei Klassifikatoren trainieren und auf Val-Set vergleichen
      4. Bestes Modell + LabelEncoder unter `models/` speichern
      5. Automatische Evaluation auf dem Test-Set starten
    """
    
    parser = argparse.ArgumentParser(description="Train SentenceTransformer + sklearn commit classifier")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformer Basismodell")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-grid", action="store_true", help="Kein GridSearch (schneller, ~3 Min)")
    args = parser.parse_args()

    use_grid = not args.no_grid

    # ── Daten laden ───────────────────────────────────────────────────────────
    if not Path(args.data).exists():
        print(f"ERROR: Trainingsdaten nicht gefunden: {args.data}")
        print("Zuerst ml/label_commits.py ausführen.")
        sys.exit(1)

    print(f"Lade Daten aus {args.data}...")
    data = load_jsonl(args.data)
    data = [d for d in data if d.get("label") in CATEGORIES]
    print(f"Gültige Samples: {len(data)}")

    train_data, val_data, test_data = stratified_split(data)
    print_distribution("Train", train_data)
    print_distribution("Val", val_data)
    print_distribution("Test", test_data)

    save_jsonl(train_data, "data/splits/train.jsonl")
    save_jsonl(val_data, "data/splits/val.jsonl")
    save_jsonl(test_data, "data/splits/test.jsonl")
    print("\nSplits gespeichert: data/splits/")

    # ── BERT-Embeddings erzeugen ───────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import LabelEncoder

    print(f"\nLade SentenceTransformer: {args.model}")
    st_model = SentenceTransformer(args.model)

    train_texts = [d["message"] for d in train_data]
    val_texts = [d["message"] for d in val_data]
    train_labels = [d["label"] for d in train_data]
    val_labels = [d["label"] for d in val_data]

    print("Encoding Train...")
    X_train = build_features(train_texts, st_model)
    print("Encoding Val...")
    X_val = build_features(val_texts, st_model)

    le = LabelEncoder()
    le.fit(CATEGORIES)
    y_train = le.transform(train_labels)
    y_val = le.transform(val_labels)

    print(f"\nFeature-Dimension: {X_train.shape[1]} "
          f"(BERT={st_model.get_sentence_embedding_dimension()} + Prefix={len(CATEGORIES)+1}×5)")

    # ── Klassifikatoren vergleichen ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KLASSIFIKATOR-VERGLEICH (Val-Set)")
    print("=" * 60)
    best_clf, best_name = compare_classifiers(X_train, y_train, X_val, y_val, le, use_grid)

    # ── Modell speichern ───────────────────────────────────────────────────────
    Path(args.out).mkdir(parents=True, exist_ok=True)
    st_model.save(args.out)

    label2id = {cat: int(le.transform([cat])[0]) for cat in CATEGORIES}
    id2label = {v: k for k, v in label2id.items()}

    with open(ENCODER_OUT, "wb") as f:
        pickle.dump({
            "label2id": label2id,
            "id2label": id2label,
            "categories": CATEGORIES,
            "sklearn_classifier": best_clf,
            "classifier_name": best_name,
            "label_encoder": le,
            "mode": "sklearn",
            "use_prefix_features": True,
        }, f)

    print(f"\nModell gespeichert: {args.out}")
    print(f"Encoder gespeichert: {ENCODER_OUT}")
    print(f"Klassifikator: {best_name}")

    # ── Evaluation automatisch starten ────────────────────────────────────────
    test_path = "data/splits/test.jsonl"
    if Path(test_path).exists():
        print("\n" + "=" * 60)
        print("STARTE EVALUATION AUF TEST-SET...")
        print("=" * 60)
        from ml.evaluate import evaluate
        evaluate(test_path, args.out, ENCODER_OUT)


if __name__ == "__main__":
    main()
