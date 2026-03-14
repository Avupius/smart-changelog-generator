import argparse
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from github import Github, GithubException
from openai import OpenAI

"""
Phase 1 der NLP-Pipeline.

Ruft Commits aus vielen GitHub-Repositories ab und klassifiziert jeden Commit
mit einer von 6 Kategorien mittels GPT-4o-mini. Speichert Ergebnisse als JSONL
für die Verwendung als Trainingsdaten in train_bert.py.

Verwendung:
    python ml/label_commits.py [--token GITHUB_TOKEN] [--target 10000]
"""

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.utils import CATEGORIES, normalize_label, normalize_message, save_jsonl

load_dotenv()

# ─── Konfiguration ────────────────────────────────────────────────────────────

TARGET_REPOS = [
    # Python / Backend
    "pallets/flask", # https://github.com/pallets/flask
    "django/django", # https://github.com/django/django
    "psf/requests", # https://github.com/psf/requests
    "fastapi/fastapi", # https://github.com/fastapi/fastapi
    "celery/celery", # https://github.com/celery/celery

    # Data Science / Machine Learning
    "numpy/numpy", # https://github.com/numpy/numpy
    "pandas-dev/pandas", # https://github.com/pandas-dev/pandas
    "scikit-learn/scikit-learn", # https://github.com/scikit-learn/scikit-learn
    "tensorflow/tensorflow", # https://github.com/tensorflow/tensorflow
    "pytorch/pytorch", # https://github.com/pytorch/pytorch
    "huggingface/transformers", # https://github.com/huggingface/transformers

    # DevOps / Infrastructure
    "docker/docker-ce", # https://github.com/docker/docker-ce
    "kubernetes/kubernetes", # https://github.com/kubernetes/kubernetes
    "ansible/ansible", # https://github.com/ansible/ansible
    "hashicorp/terraform", # https://github.com/hashicorp/terraform
    "prometheus/prometheus", # https://github.com/prometheus/prometheus

    # Frontend / Web
    "facebook/react", # https://github.com/facebook/react
    "vuejs/vue", # https://github.com/vuejs/vue
    "angular/angular", # https://github.com/angular/angular
    "vercel/next.js", # https://github.com/vercel/next.js

    # Programming Languages / Compilers
    "rust-lang/rust", # https://github.com/rust-lang/rust
    "golang/go", # https://github.com/golang/go

    # CLI / Developer Tools
    "sharkdp/bat", # https://github.com/sharkdp/bat
    "sharkdp/fd", # https://github.com/sharkdp/fd
    "BurntSushi/ripgrep", # https://github.com/BurntSushi/ripgrep
    "cli/cli", # https://github.com/cli/cli

    # Large systems
    "torvalds/linux", # https://github.com/torvalds/linux
    "freebsd/freebsd-src", # https://github.com/freebsd/freebsd-src

    # Misc large projects
    "home-assistant/core", # https://github.com/home-assistant/core
    "neovim/neovim", # https://github.com/neovim/neovim
]

# Größe der Batch für die parallele Verarbeitung durch GPT
BATCH_SIZE = 20
OUTPUT_PATH = Path("data/labeled/commits_labeled.jsonl")

LABEL_SYSTEM_PROMPT = """You are an expert software engineer classifying Git commit messages.
Classify each commit into exactly one of these 6 categories:
- feature: new functionality, new capability, new endpoint, new UI element
- bugfix: fixing a bug, error, crash, incorrect behavior, or regression
- documentation: changes to docs, README, comments, docstrings, wikis
- refactor: restructuring code without changing behavior (cleanup, rename, extract)
- test: adding, fixing, or updating tests
- chore: build system, dependencies, CI/CD, tooling, version bumps, formatting

Rules:
- Return ONLY a JSON array of labels (strings), one per message, in the same order.
- Each label must be exactly one of: feature, bugfix, documentation, refactor, test, chore
- No explanations, no extra text, no markdown — just the JSON array.
"""


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _is_merge_commit(msg: str) -> bool:
    """Prüfe, ob eine Nachricht ein Merge-Commit ist."""
    first = msg.splitlines()[0].strip()
    return first.lower().startswith("merge")


def fetch_repo_commits(repo_name: str, g: Github, limit: int) -> list[dict]:
    """Rufe bis zu `limit` nicht-Merge Commits aus einem Repo ab."""
    try:
        repo = g.get_repo(repo_name)
        commits = []
        for c in repo.get_commits():
            if len(commits) >= limit:
                break
            msg = c.commit.message or ""
            subject = msg.splitlines()[0].strip()
            if not subject or _is_merge_commit(subject):
                continue
            commits.append({
                "sha": c.sha[:7],
                "message": subject,
                "repo": repo_name,
            })
        print(f"  [{repo_name}] fetched {len(commits)} commits")
        return commits
    except GithubException as e:
        print(f"  [{repo_name}] SKIP — {e.status}: {e.data.get('message', '')}")
        return []


def label_batch(messages: list[str], client: OpenAI) -> list[str | None]:
    """Rufe GPT-4o-mini auf, um einen Batch von Commit-Nachrichten zu klassifizieren."""
    messages_json = json.dumps(messages, ensure_ascii=False)
    user_prompt = f"Classify these {len(messages)} commit messages:\n{messages_json}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        labels_raw = json.loads(raw)
        if not isinstance(labels_raw, list) or len(labels_raw) != len(messages):
            print(f"  WARNING: GPT returned {len(labels_raw)} labels for {len(messages)} messages")
            return [None] * len(messages)
        return [normalize_label(lbl) for lbl in labels_raw]
    except (json.JSONDecodeError, Exception) as e:
        print(f"  WARNING: GPT labeling error — {e}")
        return [None] * len(messages)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Label GitHub commits with GPT-4o-mini")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"), help="GitHub token")
    parser.add_argument("--target", type=int, default=10_000, help="Target number of labeled commits")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSONL path")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set in environment / .env file")
        sys.exit(1)

    client = OpenAI(api_key=openai_key)
    g = Github(args.token) if args.token else Github()

    per_repo = max(args.target // len(TARGET_REPOS), 200)
    print(f"Target: {args.target} commits | ~{per_repo} per repo | {len(TARGET_REPOS)} repos")
    print()

    # ── Schritt 1: Rufe rohe Commits ab ───────────────────────────────────────────
    all_commits: list[dict] = []
    seen_normalized: set[str] = set()

    for repo_name in TARGET_REPOS:
        raw = fetch_repo_commits(repo_name, g, per_repo * 2)  # Rufe Extra-Commits für deduplizierung
        for c in raw:
            norm = normalize_message(c["message"])
            if norm not in seen_normalized:
                seen_normalized.add(norm)
                all_commits.append(c)

    print(f"\nTotal unique commits after dedup: {len(all_commits)}")

# ── Schritt 2: Klassifiziere in Batches ───────────────────────────────────────
    labeled: list[dict] = []
    failed = 0
    messages = [c["message"] for c in all_commits]

    print(f"\nLabeling {len(messages)} commits in batches of {BATCH_SIZE}...")
    for i in range(0, len(messages), BATCH_SIZE):
        batch_msgs = messages[i : i + BATCH_SIZE]
        batch_commits = all_commits[i : i + BATCH_SIZE]

        labels = label_batch(batch_msgs, client)

        for commit, label in zip(batch_commits, labels):
            if label is None:
                failed += 1
                continue
            labeled.append({
                "sha": commit["sha"],
                "message": commit["message"],
                "repo": commit["repo"],
                "label": label,
            })

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Progress: {len(labeled)} labeled, {failed} failed (batch {i // BATCH_SIZE + 1})")

        # Kurze Pause zur einhaltung der API-Limits
        time.sleep(0.1)

# ── Schritt 3: Speichere und berichte ──────────────────────────────────────────
    save_jsonl(labeled, args.output)
    print(f"\nSaved {len(labeled)} labeled commits to {args.output}")
    print(f"Failed/skipped: {failed}")

    # Label distribution
    from collections import Counter
    dist = Counter(c["label"] for c in labeled)
    print("\nLabel distribution:")
    for cat in CATEGORIES:
        count = dist.get(cat, 0)
        pct = count / len(labeled) * 100 if labeled else 0
        bar = "#" * (count // max(len(labeled) // 50, 1))
        print(f"  {cat:<15} {count:>5} ({pct:5.1f}%)  {bar}")


if __name__ == "__main__":
    main()
