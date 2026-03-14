from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, model_validator, field_validator


class ChangelogRequest(BaseModel):
    """
    Eingehende Anfrage für die Changelog-Generierung.
    Validiert durch Pydantic, FastAPI deserialisiert den JSON-Request-Body automatisch in dieses Modell und wirft einen 422-Fehler wenn die
    Validierung fehlschlägt.
    """
    repo_url: str
    github_token: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    last_n_commits: Optional[int] = None
    output_language: str = "English"

    @model_validator(mode="after")
    def validate_date_or_commits(self):
        """
        Stellt sicher das genau einer der beiden Modi angegeben wurde: 
        entweder ein Datenbereich (date_from/date_to) oder eine Anzahl letzter Commits (last_n_commits).
        """
        has_dates = self.date_from is not None or self.date_to is not None
        has_n = self.last_n_commits is not None
        if not has_dates and not has_n:
            raise ValueError("Provide either a date range or last_n_commits.")
        if has_dates and has_n:
            raise ValueError("Provide either a date range or last_n_commits, not both.")
        if self.last_n_commits is not None and self.last_n_commits < 1:
            raise ValueError("last_n_commits must be >= 1.")
        return self

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        """
        Bereinigt die Repository-URL und stellt sicher, dass sie auf ein GitHub-Repository zeigt.
        """
        v = v.strip().rstrip("/")
        if "github.com" not in v:
            raise ValueError("URL must point to a GitHub repository.")
        return v


class CommitData(BaseModel):
    """
    Repräsentiert einen einzelnen Commit nach dem Laden via GitHub API. 
    """
    sha: str
    message: str
    author: str
    date: datetime
    url: str


# class ClassifiedCommit(BaseModel):
#     """
#     Commit nach der Klassifizierung durch GPT, mit Kategorie und Konfidenzwert des Klassifikators
#     """
#     sha: str
#     message: str
#     author: str
#     date: datetime
#     url: str
#     category: str
#     confidence: float


class ChangelogResponse(BaseModel):
    """
    Reponse des /api/changelog Endpunkts, enthält das fertige Markdown-Dokument und Metadaten für das Frontend.
    """
    markdown: str
    commit_count: int
    categories_found: list[str]
    generation_time_seconds: float


class EvaluationResponse(BaseModel):
    """
    Response des /api/evaluate Endpunkts, enthält alle Klassifikationsmetriken für die Evaluations-Ansicht im Frontend.
    Metriken werden in drei Varianten geliefert:
    - macro: Gleichgewichtung aller Klassen
    - micro: Gewichtung nach Häufigkeit (entspricht der Accuracy)
    - weighted: Gewicht nach Support (Anzahl Samples pro Klasse)
    """
    accuracy: float
    precision_macro: float
    precision_micro: float
    precision_weighted: float
    recall_macro: float
    recall_micro: float
    recall_weighted: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    per_class: dict # Metriken pro Kategorie {label: {precision, recall, f1, support}}
    confusion_matrix: list[list[int]] # 6×6 Matrix: Zeilen = True Label, Spalten = Predicted Label
    categories: list[str] # Reihenfolge der Kategorien für die Confusion Matrix
    total_samples: int # Gesamtanzahl der Test-Samples