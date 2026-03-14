from datetime import date, datetime, timezone
from typing import Optional
import re

from github import Github, GithubException
from app.models.schemas import CommitData


def _parse_repo_path(repo_url: str) -> str:
    """
    Extrahier den Pfad "owner/repo" aus einer GitHub-URL
    """
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", repo_url)
    if not match:
        raise ValueError(f"Cannot parse repository path from URL: {repo_url}")
    return match.group(1)


def _is_merge_commit(message: str) -> bool:
    """Erkennt Merge-Commits und filtert sie heraus, Merge-Commits enthalten keine neuen Informationen für die Changelog-Generierung."""
    subject = message.splitlines()[0].strip()
    return (
        subject.startswith("Merge ")
        or subject.startswith("merge ")
        or re.match(r"^Merge (pull request|branch|remote)", subject, re.IGNORECASE) is not None
    )


def _subject_line(message: str) -> str:
    """Gibt nur die erste Zeile der Commit-Message zurück <-> Commit-Titel"""
    return message.splitlines()[0].strip()


def fetch_commits(repo_url: str, github_token: Optional[str], date_from: Optional[date], 
                  date_to: Optional[date], last_n: Optional[int],) -> list[CommitData]:
    """
    Lädt Commits aus einem GitHub-Repository via PyGithub.
    Unterstützt zwei Filtermodi: Datumsbereich (date_from, date_to) oder die letzten N Commits (last_n).
    Merge-Commits werden automatisch herausgefiltert, da sie keine neuen Informationen für die Changelog-Generierung enthalten.

    Args: 
            Args:
        repo_url:     Vollständige GitHub-URL des Repositories
        github_token: Optionaler Personal Access Token für private Repos oder höhere API Rate-Limits
        date_from:    Commits ab diesem Datum (inklusiv, UTC Mitternacht)
        date_to:      Commits bis zu diesem Datum (inklusiv, UTC 23:59:59)
        last_n:       Maximale Anzahl zurückgegebener Commits

    Returns: 
        Liste von CommitData-Objekten, chronologisch absteigend sortiert (neueste zuerst)

    Raises: 
        ValueError: Bei ungültigen Eingaben (z.B. ungültige URL, ungültige Datumsangaben, API-Fehler)
    """

    # Ohne Token ist die Rate auf 60 Anfragen pro Stunde begrenzt, mit Token 5000 pro Stunde
    g = Github(github_token) if github_token else Github()

    try:
        repo_path = _parse_repo_path(repo_url)
        repo = g.get_repo(repo_path)
    except GithubException as e:

        # HTTP-Statuscodes abfangen und verständliche Fehlermeldungen zurückgeben
        if e.status == 404:
            raise ValueError(f"Repository not found: {repo_url}. Check the URL and token.")
        if e.status == 401:
            raise ValueError("Invalid GitHub token.")
        raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}")

    # GitHub API erwartet timezone-aware datetime-Objekte.
    # date_from/date_to sind date-Objekte ohne Zeit — wir konvertieren auf UTC.
    # date_to bekommt 23:59:59 damit der gesamte letzte Tag eingeschlossen ist.
    kwargs: dict = {}
    if date_from:
        kwargs["since"] = datetime(date_from.year, date_from.month, date_from.day, tzinfo=timezone.utc)
    if date_to:
        kwargs["until"] = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=timezone.utc)

    try:
        paginated = repo.get_commits(**kwargs)
    except GithubException as e:
        raise ValueError(f"Failed to fetch commits: {e.data.get('message', str(e))}")

    results: list[CommitData] = []

    # Ohne last_n alle Commits laden (bis zu 10_000 als Sicherheitsgrenze)
    limit = last_n if last_n else 10_000

    for commit in paginated:
        if len(results) >= limit:
            break

        # Merge-Commits und Commits ohne Nachricht überspringen
        msg = commit.commit.message or ""
        if _is_merge_commit(msg):
            continue
        subject = _subject_line(msg)
        if not subject:
            continue
        
        # Author-Name und Commit-Datum extrahieren, mit Fallbacks für fehlende Daten
        author = (
            commit.commit.author.name
            if commit.commit.author and commit.commit.author.name
            else "Unknown"
        )
        committed_date = commit.commit.author.date if commit.commit.author else datetime.now(timezone.utc)

        results.append(
            CommitData(
                sha=commit.sha[:7], # Nur die ersten 7 Zeichen des Hashes
                message=subject,
                author=author,
                date=committed_date,
                url=commit.html_url,
            )
        )

    return results
