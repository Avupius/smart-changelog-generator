from datetime import date, datetime, timezone
from typing import Optional
import re

from github import Github, GithubException
from app.models.schemas import CommitData


def _parse_repo_path(repo_url: str) -> str:
    """Extract 'owner/repo' from a GitHub URL."""
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", repo_url)
    if not match:
        raise ValueError(f"Cannot parse repository path from URL: {repo_url}")
    return match.group(1)


def _is_merge_commit(message: str) -> bool:
    """Filter out merge commits — they carry no semantic changelog value."""
    subject = message.splitlines()[0].strip()
    return (
        subject.startswith("Merge ")
        or subject.startswith("merge ")
        or re.match(r"^Merge (pull request|branch|remote)", subject, re.IGNORECASE) is not None
    )


def _subject_line(message: str) -> str:
    """Return only the first line of a commit message."""
    return message.splitlines()[0].strip()


def fetch_commits(
    repo_url: str,
    github_token: Optional[str],
    date_from: Optional[date],
    date_to: Optional[date],
    last_n: Optional[int],
) -> list[CommitData]:
    """Fetch commits from a GitHub repository."""
    g = Github(github_token) if github_token else Github()

    try:
        repo_path = _parse_repo_path(repo_url)
        repo = g.get_repo(repo_path)
    except GithubException as e:
        if e.status == 404:
            raise ValueError(f"Repository not found: {repo_url}. Check the URL and token.")
        if e.status == 401:
            raise ValueError("Invalid GitHub token.")
        raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}")

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
    limit = last_n if last_n else 10_000

    for commit in paginated:
        if len(results) >= limit:
            break
        msg = commit.commit.message or ""
        if _is_merge_commit(msg):
            continue
        subject = _subject_line(msg)
        if not subject:
            continue

        author = (
            commit.commit.author.name
            if commit.commit.author and commit.commit.author.name
            else "Unknown"
        )
        committed_date = commit.commit.author.date if commit.commit.author else datetime.now(timezone.utc)

        results.append(
            CommitData(
                sha=commit.sha[:7],
                message=subject,
                author=author,
                date=committed_date,
                url=commit.html_url,
            )
        )

    return results
