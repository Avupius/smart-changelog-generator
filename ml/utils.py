import json
import random
import re
from pathlib import Path
from collections import defaultdict

CATEGORIES = ["feature", "bugfix", "documentation", "refactor", "test", "chore"]

CATEGORY_ALIASES = {
    "feat": "feature",
    "feature": "feature",
    "fix": "bugfix",
    "bugfix": "bugfix",
    "bug": "bugfix",
    "hotfix": "bugfix",
    "docs": "documentation",
    "doc": "documentation",
    "documentation": "documentation",
    "refactor": "refactor",
    "refact": "refactor",
    "test": "test",
    "tests": "test",
    "testing": "test",
    "chore": "chore",
    "build": "chore",
    "ci": "chore",
    "style": "chore",
    "perf": "chore",
    "performance": "chore",
}


def normalize_label(label: str) -> str | None:
    """Normalize a raw label string to one of the 6 canonical categories."""
    label = label.strip().lower()
    return CATEGORY_ALIASES.get(label)


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(data: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stratified_split(
    data: list[dict],
    label_key: str = "label",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split preserving label distribution."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1."
    random.seed(seed)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_label[item[label_key]].append(item)

    train, val, test = [], [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def normalize_message(msg: str) -> str:
    """Strip conventional commit prefix and normalize for deduplication."""
    msg = msg.strip()
    # Remove conventional commit prefix like "feat: ", "fix(scope): "
    msg = re.sub(r"^[a-z]+(\([^)]+\))?!?:\s*", "", msg, flags=re.IGNORECASE)
    return msg.lower().strip()
