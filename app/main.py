"""
main.py — Einstiegspunkt der FastAPI-Anwendung.

Lädt das Klassifikator-Modell einmalig beim Start über den Lifespan-Context-Manager
und stellt anschließend die API sowie die statischen Frontend-Dateien bereit.

Starten mit:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.classifier import Classifier

# Umgebungsvariablen aus der .env-Datei laden (z.B. OPENAI_API_KEY)
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Klassifikator laden 
    # ModelNotTrainedError wird abgefangen damit die App auch ohne trainiertes
    # Modell startet -> API-Aufrufe geben dann einen 503-Fehler mit Hinweis zurück.
    from app.core.classifier import ModelNotTrainedError
    print("[Startup] Lade Commit-Klassifikator...")
    classifier = Classifier()
    try:
        classifier.load()
        print(f"[Startup] Klassifikator bereit.")
    except ModelNotTrainedError as e:
        print(f"[Startup] WARNUNG: {e}")
        print("[Startup] App startet ohne Modell — bitte ml/train_bert.py ausführen.")
    except RuntimeError as e:
        print(f"[Startup] FEHLER beim Laden: {e}")

    # Klassifikator im App-State speichern, damit alle Request-Handler übner request.app.state.classifier darauf zugreifen können
    app.state.classifier = classifier
    yield

    # Shutdown 
    print("[Shutdown] Cleaning up...")


# FastAPI-Instanz mit Metadaten für die automatische Dokumentation (OpenAPI/Swagger)
app = FastAPI(
    title="Smart Changelog Generator",
    description="NLP pipeline that generates structured changelogs from GitHub commits.",
    version="1.0.0",
    lifespan=lifespan,
)

# Alle API-Routen unter dem Präfix /api einbinden (z.B. /api/changelog, /api/evaluate)
app.include_router(router, prefix="/api")

# Statische Dateien (frontend)
static_dir = Path("app/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Liefert die SPA-Einstiegsseite. Fallback auf JSON wenn die Datei fehlt."""
    index = Path("app/static/index.html")
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Smart Changelog Generator API", "docs": "/docs"}

# Beide Routen zeigen auf dieselbe index.html -> die SPA übernimmt das
# clientseitige Routing selbst (via app.js). So funktionieren auch direkte
# Aufrufe von /evaluation ohne 404-Fehler.
@app.get("/index.html", include_in_schema=False)
@app.get("/evaluation", include_in_schema=False)
async def spa_pages():
    return FileResponse(str(Path("app/static/index.html")))