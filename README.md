# Smart Changelog Generator

An NLP pipeline that automatically generates structured, categorized, and summarized changelogs from GitHub commits — without manual configuration, for any repository.

> Bachelor module project for **Natural Language Processing**

---

## Architecture

```
GitHub API  →  Commit Classifier (Sentence-BERT)  →  GPT-4o-mini Summarizer  →  Markdown Changelog
                       ↑
              Trained on 10k labeled commits
              (labeled via GPT-4o-mini)
```

### 6 Commit Categories
| Category | Description |
|---|---|
| `feature` | New functionality or capabilities |
| `bugfix` | Bug fixes and error corrections |
| `documentation` | Docs, README, comments |
| `refactor` | Code restructuring without behavior change |
| `test` | Test additions and fixes |
| `chore` | Build, CI/CD, dependencies, tooling |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and optionally GITHUB_TOKEN
```

---

## NLP Pipeline (run in order)

### Step 1 — Label training data

Fetches ~10,000 commits from 20 popular GitHub repositories and labels each with GPT-4o-mini:

```bash
python ml/label_commits.py
# Output: data/labeled/commits_labeled.jsonl
```

### Step 2 — Train the classifier

Fine-tunes a Sentence-BERT model (`all-MiniLM-L6-v2`) on the labeled data:

```bash
python ml/train_bert.py
# Output: models/bert_classifier/ + models/label_encoder.pkl
# Also creates: data/splits/{train,val,test}.jsonl
```

Options:
```bash
python ml/train_bert.py --epochs 5 --force-sklearn  # use LogisticRegression fallback
```

### Step 3 — Evaluate the classifier

Computes full NLP classification metrics on the held-out test set:

```bash
python ml/evaluate.py
python ml/evaluate.py --plot  # also saves confusion matrix as PNG
```

**Metrics reported:**
- Accuracy
- Precision (macro / micro / weighted)
- Recall (macro / micro / weighted)
- F1-Score (macro / micro / weighted)
- Per-class: Precision, Recall, F1, Support
- Confusion matrix (6×6)

Output saved to `data/evaluation_results.json`.

---

## Web Application

### Start the server

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

### Usage

1. Enter a GitHub repository URL (public or private)
2. Optionally enter a GitHub token (required for private repos)
3. Select a date range **or** a number of recent commits
4. Choose an output language
5. Click **Generate Changelog**

The changelog is rendered as Markdown in the browser and can be downloaded as `.md`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/changelog` | Generate changelog |
| `GET` | `/api/languages` | List supported output languages |
| `GET` | `/api/health` | Health check + model status |
| `POST` | `/api/evaluate` | Run classifier evaluation |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Project Structure

```
smart-changelog-generator/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/routes.py        # API endpoints
│   ├── core/
│   │   ├── classifier.py    # Sentence-BERT commit classifier
│   │   ├── summarizer.py    # Async GPT-4o-mini summarizer
│   │   ├── changelog_gen.py # Markdown generator
│   │   └── github_client.py # GitHub API client
│   ├── models/schemas.py    # Pydantic models
│   └── static/              # Frontend (HTML/CSS/JS)
├── ml/
│   ├── utils.py             # Shared utilities
│   ├── label_commits.py     # GPT-4o-mini labeling script
│   ├── train_bert.py        # Sentence-BERT training
│   └── evaluate.py          # Full evaluation metrics
├── data/                    # Training data + splits (gitignored)
├── models/                  # Trained model artifacts (gitignored)
├── requirements.txt
└── .env.example
```

---

## Notes

- The classifier falls back to a regex-based heuristic if no trained model is found, so the web app works end-to-end before training.
- The labeling script costs approximately **$0.10** in OpenAI API credits for 10,000 commits.
- GPU is not required for inference. Training is faster with a CUDA GPU but works on CPU.
