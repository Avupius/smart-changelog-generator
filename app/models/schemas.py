from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, model_validator, field_validator


class ChangelogRequest(BaseModel):
    repo_url: str
    github_token: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    last_n_commits: Optional[int] = None
    output_language: str = "English"

    @model_validator(mode="after")
    def validate_date_or_commits(self):
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
        v = v.strip().rstrip("/")
        if "github.com" not in v:
            raise ValueError("URL must point to a GitHub repository.")
        return v


class CommitData(BaseModel):
    sha: str
    message: str
    author: str
    date: datetime
    url: str


class ClassifiedCommit(BaseModel):
    sha: str
    message: str
    author: str
    date: datetime
    url: str
    category: str
    confidence: float


class ChangelogResponse(BaseModel):
    markdown: str
    commit_count: int
    categories_found: list[str]
    generation_time_seconds: float


class EvaluationResponse(BaseModel):
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
    per_class: dict
    confusion_matrix: list[list[int]]
    total_samples: int
