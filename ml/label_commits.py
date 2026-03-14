"""
label_commits.py — Phase 1 of the NLP pipeline.

Fetches commits from diverse GitHub repositories and labels each commit
with one of 6 categories using GPT-4o-mini. Saves results as JSONL for
use as training data in train_bert.py.

Usage:
    python ml/label_commits.py [--token GITHUB_TOKEN] [--target 10000]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from github import Github, GithubException
from openai import OpenAI

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.utils import CATEGORIES, normalize_label, normalize_message, save_jsonl

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

TARGET_REPOS = [
    "microsoft/vscode",
    "pytorch/pytorch",
    "django/django",
    "facebook/react",
    "kubernetes/kubernetes",
    "rust-lang/rust",
    "golang/go",
    "tensorflow/tensorflow",
    "rails/rails",
    "laravel/laravel",
    "vuejs/vue",
    "angular/angular",
    "expressjs/express",
    "fastapi/fastapi",
    "scikit-learn/scikit-learn",
    "huggingface/transformers",
    "numpy/numpy",
    "pandas-dev/pandas",
    "home-assistant/core",
    "grafana/grafana",
]

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_merge_commit(msg: str) -> bool:
    first = msg.splitlines()[0].strip()
    return first.lower().startswith("merge")


def fetch_repo_commits(repo_name: str, g: Github, limit: int) -> list[dict]:
    """Fetch up to `limit` non-merge commits from a repo."""
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
    """Call GPT-4o-mini to label a batch of commit messages."""
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

    # ── Step 1: Fetch raw commits ──────────────────────────────────────────────
    all_commits: list[dict] = []
    seen_normalized: set[str] = set()

    for repo_name in TARGET_REPOS:
        raw = fetch_repo_commits(repo_name, g, per_repo * 2)  # fetch extra for dedup
        for c in raw:
            norm = normalize_message(c["message"])
            if norm not in seen_normalized:
                seen_normalized.add(norm)
                all_commits.append(c)

    print(f"\nTotal unique commits after dedup: {len(all_commits)}")

    # ── Step 2: Label in batches ───────────────────────────────────────────────
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

        # Small sleep to respect rate limits
        time.sleep(0.1)

    # ── Step 3: Save and report ────────────────────────────────────────────────
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
