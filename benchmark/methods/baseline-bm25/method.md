# Baseline: BM25

Sanity-floor baseline. SQLite FTS5 with the BM25 ranking function over the 50-note corpus.

## What this measures

Pure lexical overlap between query and note body. No semantics, no learning, no embeddings. If geodesic retrieval can't beat BM25 on cross-domain bridge queries (where the *whole point* is finding notes with no shared keywords), the entire premise of the project is wrong.

## Inputs

- `../../corpus/notes.jsonl` — 50 notes
- `../../queries.public.jsonl` — 30 cross-domain bridge queries

## Outputs

- `../../results/bm25-{date}.jsonl` — for each query, top-5 retrieved note IDs ranked by BM25 score

## Run

```bash
pip install -r requirements.txt
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.public.jsonl --out ../../results/bm25-$(date +%Y-%m-%d).jsonl
```

## Implementation

Uses Python's `rank_bm25` package with default Okapi BM25 (k1=1.5, b=0.75). Tokenization: simple lowercase + whitespace split. No stemming — keeps the baseline as raw as possible.
