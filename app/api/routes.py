"""
routes.py — Alle API-Endpunkte für den Smart Changelog Generator.

Definiert die REST-API die vom Frontend aufgerufen wird. 
Orchestriert die Pipeline-Schritte:
    1. Commits laden (github_client)
    2. Commit-Klassifikation (classifier)
    3. Kategorien zusammenfassen (summarizer)
    4. Changelog generieren (changelog_gen)
"""

import asyncio
import time
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.core import changelog_gen, github_client, summarizer
from app.models.schemas import ChangelogRequest, ChangelogResponse, EvaluationResponse

router = APIRouter()

# Alle vom Frontend unterstützten Sprachen (muss mit Frontend übereinstimmen)
SUPPORTED_LANGUAGES = [
    "English",
    "German",
    "Romanian",
    "French",
    "Spanish",
    "Italian",
    "Portuguese",
    "Japanese",
    "Chinese",
]


@router.get("/health")
async def health(request: Request):
    """
    Health-Check Endpoint, um den Status des Servers und des geladenen Modells zu überprüfen.
    (Aktuell nicht genutzt)
    """
    classifier = getattr(request.app.state, "classifier", None)
    loaded = classifier is not None and classifier.is_loaded
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_mode": classifier.mode if loaded else "not loaded",
    }


@router.get("/languages")
async def get_languages():
    """Gibt die Liste der unterstützten Sprachen zurück (muss mit Frontend übereinstimmen)."""
    return SUPPORTED_LANGUAGES


@router.post("/changelog", response_model=ChangelogResponse)
async def generate_changelog(req: ChangelogRequest, request: Request):
    """
    Hauptendpunkt: Orchestriert die gesamte Pipeline von der Commit-Abfrage bis zur Changelog-Generierung.
    Nimmt eine Changelog-Anfrage entgegen und durchläuft die Schritte:
    1. Commits laden (github_client)
    2. Commit-Klassifikation (classifier)
    3. Kategorien zusammenfassen (summarizer)
    4. Changelog generieren (changelog_gen)
    Gibt die generierte Markdown-Changelog und Metadaten zurück.

    Die Gesamtlaufzeit wird gemessen und im Response zurückgegeben, damit das Frontend sie anzeigen kann. 
    """
    t_start = time.perf_counter()

    # Klassifikator aus dem App-State laden, wird einmalig beim Serverstart geladen und dann für alle Anfragen wiederverwendet
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Classifier not loaded. Check server logs.")

    # 1. Commits laden 
    # GitHub API aufrufen und Commits für die angegebene Repo und Zeitspanne / Anzahl abrufen
    # ValueError wird bei ungültiger URL, falschem Token oder nicht gefundenen Repo geworfen und als HTTP 400 zurückgegeben
    try:
        commits = github_client.fetch_commits(
            repo_url=req.repo_url,
            github_token=req.github_token,
            date_from=req.date_from,
            date_to=req.date_to,
            last_n=req.last_n_commits,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not commits:
        raise HTTPException(
            status_code=404,
            detail="No commits found for the selected repository and time range.",
        )


    # 2. Commit-Klassifikation
    # Alle Commit-Nachrichten in einem einzigen Batch klassifizieren
    messages = [c.message for c in commits]
    classifications = classifier.classify_batch(messages)

    # Commits nach Kategorie gruppieren
    categorized: dict[str, list[str]] = defaultdict(list)
    for commit, (label, _confidence) in zip(commits, classifications):
        categorized[label].append(commit.message)

    # Liste der tatsächlich gefundenen Kategorien (für die Response-Metadaten)
    categories_found = [cat for cat in categorized if categorized[cat]]

    # 3. Kategorien zusammenfassen
    # Für jede Kategorie die zugehörigen Commit-Nachrichten an den Summarizer übergeben, um eine kurze Zusammenfassung zu erhalten.
    # ValueError -> fehlender API-Key (500)
    # Alle anderen Fehler -> OpenAI API nicht erreichbar oder Fehler bei der Anfrage (502)
    try:
        summaries = await summarizer.summarize_all(
            categorized=dict(categorized),
            language=req.output_language,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {e}")

    # 4. Changelog generieren
    # Baut aus den Zusammenfassungen und den Metadaten das finale Markdown-Dokument
    markdown = changelog_gen.generate_changelog(
        summaries=summaries,
        repo_url=req.repo_url,
        date_from=req.date_from,
        date_to=req.date_to,
        last_n=req.last_n_commits,
        commit_count=len(commits),
        output_language=req.output_language,
    )

    elapsed = time.perf_counter() - t_start

    return ChangelogResponse(
        markdown=markdown,
        commit_count=len(commits),
        categories_found=categories_found,
        generation_time_seconds=round(elapsed, 2),
    )


@router.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(request: Request):
    """
    Startet die Modell-Evaluation auf dem Test-Split. Lädt die Test-Daten, führt die Vorhersagen durch und berechnet die Metriken.
    Läuft in einem separaten Thread, damit der Server nicht blockiert wird. Gibt die Ergebnisse zurück, sobald sie verfügbar sind.
    """
    
    test_path = "data/splits/test.jsonl"
    model_path = "models/bert_classifier"
    encoder_path = "models/label_encoder.pkl"

    # Prüfe, ob die benötigten Dateien vorhanden sind, bevor die Evaluation gestartet wird.
    if not Path(test_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Test split not found. Run ml/train_bert.py first.",
        )
    if not Path(encoder_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Run ml/train_bert.py first.",
        )

    try:
        import sys
        sys.path.insert(0, ".")
        from ml.evaluate import evaluate
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: evaluate(test_path, model_path, encoder_path),
        )
        return EvaluationResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
