"""
evaluate.py — Phase 3 des NLP-Pipelines (Evaluation).

Evaluiert den trainierten SentenceTransformer + sklearn Klassifikator
auf dem held-out Test-Set.

Metriken:
  - Accuracy
  - Precision (macro, micro, weighted)
  - Recall    (macro, micro, weighted)
  - F1-Score  (macro, micro, weighted)
  - Per-Class: Precision, Recall, F1, Support
  - Confusion Matrix (6×6)

Usage:
    python ml/evaluate.py
    python ml/evaluate.py --test data/splits/test.jsonl --model models/bert_classifier
    python ml/evaluate.py --plot   # speichert Confusion Matrix als PNG
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.features import build_features
from ml.utils import CATEGORIES, load_jsonl

DEFAULT_TEST    = "data/splits/test.jsonl"
DEFAULT_MODEL   = "models/bert_classifier"
ENCODER_PATH    = "models/label_encoder.pkl"
OUTPUT_JSON     = "data/evaluation_results.json"


# ─── Classifier loader ────────────────────────────────────────────────────────

def load_classifier(model_path: str, encoder_path: str):
    """Lädt SentenceTransformer + sklearn Klassifikator, gibt predict-Funktion zurück."""
    if not Path(encoder_path).exists():
        raise FileNotFoundError(
            f"Encoder nicht gefunden: {encoder_path}\n"
            "Zuerst ml/train_bert.py ausführen."
        )

    with open(encoder_path, "rb") as f:
        meta = pickle.load(f)

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(model_path)
    clf      = meta["sklearn_classifier"]
    le       = meta["label_encoder"]
    clf_name = meta.get("classifier_name", "sklearn")

    print(f"  Klassifikator: {clf_name}")

    def predict(messages: list[str]) -> list[str]:
        X     = build_features(messages, st_model)
        preds = clf.predict(X)
        return list(le.inverse_transform(preds))

    return predict


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix — Commit Classifier", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\nConfusion Matrix gespeichert → {output_path}")
    except ImportError:
        print("\n[INFO] matplotlib/seaborn nicht installiert — Plot übersprungen.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(test_path: str, model_path: str, encoder_path: str, plot: bool = False) -> dict:
    if not Path(test_path).exists():
        raise FileNotFoundError(
            f"Test-Daten nicht gefunden: {test_path}\n"
            "Zuerst ml/train_bert.py ausführen."
        )

    print(f"Lade Test-Daten aus {test_path}...")
    test_data = load_jsonl(test_path)
    test_data = [d for d in test_data if d.get("label") in CATEGORIES]
    print(f"Test-Samples: {len(test_data)}")

    if not test_data:
        raise ValueError("Keine gültigen Test-Samples gefunden.")

    messages = [d["message"] for d in test_data]
    y_true   = [d["label"]   for d in test_data]

    print(f"\nLade Klassifikator aus {model_path}...")
    predict_fn = load_classifier(model_path, encoder_path)

    print("Inferenz läuft...")
    y_pred = predict_fn(messages)

    # ── Metriken ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)

    prec_macro    = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    prec_micro    = precision_score(y_true, y_pred, average="micro",    zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    rec_macro     = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_micro     = recall_score(y_true, y_pred, average="micro",    zero_division=0)
    rec_weighted  = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    f1_macro      = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_micro      = f1_score(y_true, y_pred, average="micro",    zero_division=0)
    f1_weighted   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = classification_report(
        y_true, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)

    # ── Ausgabe ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — Commit Classifier")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Macro':>10} {'Micro':>10} {'Weighted':>10}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {'':>10} {acc:>10.4f} {'':>10}")
    print(f"{'Precision':<25} {prec_macro:>10.4f} {prec_micro:>10.4f} {prec_weighted:>10.4f}")
    print(f"{'Recall':<25} {rec_macro:>10.4f} {rec_micro:>10.4f} {rec_weighted:>10.4f}")
    print(f"{'F1-Score':<25} {f1_macro:>10.4f} {f1_micro:>10.4f} {f1_weighted:>10.4f}")
    print("\n" + "-" * 60)
    print("Per-Class Metrics:")
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 50)
    for cat in CATEGORIES:
        if cat in per_class:
            r = per_class[cat]
            print(f"  {cat:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                  f"{r['f1-score']:>10.4f} {int(r['support']):>10}")
    print("\n" + "-" * 60)
    print("Confusion Matrix (Zeilen=True, Spalten=Predicted):")
    header = "  " + "".join(f"{c[:6]:>8}" for c in CATEGORIES)
    print(header)
    for i, cat in enumerate(CATEGORIES):
        print("  " + f"{cat[:6]:<6}" + "".join(f"{cm[i][j]:>8}" for j in range(len(CATEGORIES))))
    print("=" * 60)

    # ── Speichern ──────────────────────────────────────────────────────────────
    results = {
        "total_samples":     len(test_data),
        "accuracy":          acc,
        "precision_macro":   prec_macro,
        "precision_micro":   prec_micro,
        "precision_weighted":prec_weighted,
        "recall_macro":      rec_macro,
        "recall_micro":      rec_micro,
        "recall_weighted":   rec_weighted,
        "f1_macro":          f1_macro,
        "f1_micro":          f1_micro,
        "f1_weighted":       f1_weighted,
        "per_class": {
            cat: {
                "precision": per_class[cat]["precision"],
                "recall":    per_class[cat]["recall"],
                "f1":        per_class[cat]["f1-score"],
                "support":   int(per_class[cat]["support"]),
            }
            for cat in CATEGORIES if cat in per_class
        },
        "confusion_matrix": cm.tolist(),
        "categories":        CATEGORIES,
    }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse gespeichert: {OUTPUT_JSON}")

    if plot:
        plot_confusion_matrix(cm, CATEGORIES, "data/confusion_matrix.png")

    return results


def main():
    parser = argparse.ArgumentParser(description="Commit-Klassifikator evaluieren")
    parser.add_argument("--test",    default=DEFAULT_TEST)
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--encoder", default=ENCODER_PATH)
    parser.add_argument("--plot",    action="store_true", help="Confusion Matrix als PNG speichern")
    args = parser.parse_args()

    try:
        evaluate(args.test, args.model, args.encoder, plot=args.plot)
    except FileNotFoundError as e:
        print(f"\nFEHLER: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
