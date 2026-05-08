# Baseline: Voyage-3 + Voyage native reranker

Strong embedding + reranker baseline with **no LLM in the loop**. Isolates "embedding + reranker quality" from "LLM-as-reranker effect" and from "LLM-explanation effect."

## What this measures

How far we can push retrieval quality on cross-domain bridge queries using only commercial embedding + commercial reranker — no Claude or GPT involvement, no prompt engineering, no rationale generation. This is the *cleanest* embedding baseline; if Method 6 (activation-derived geodesic) doesn't beat this, the gain isn't from manifold geometry, it's from something further up the stack.

## Inputs

- `../../corpus/notes.jsonl`
- `../../queries.jsonl`

## Outputs

- `../../results/voyage-native-rerank-{date}.jsonl` — top-5 reranked note IDs per query
- `../../results/traces/voyage-native-rerank-{date}/q<NN>.json` — full request/response trace per query

## Run

```bash
export VOYAGE_API_KEY=...
pip install -r requirements.txt
python run.py --corpus ../../corpus/notes.jsonl --queries ../../queries.jsonl --out ../../results/voyage-native-rerank-$(date +%Y-%m-%d).jsonl
```

## Implementation

1. Embed each note with `voyage-3` (input_type=document).
2. Embed each query with `voyage-3` (input_type=query).
3. Cosine-rank → take all 50 candidates per query (full corpus at v0.1 scale).
4. Rerank via Voyage's native `rerank-2` model — sends query + candidate texts, returns reranked scores.
5. Take top-5.

Caching: embeddings cached by note ID + model version under `.cache/`.

## Why "full corpus" reranking at v0.1

50 notes is small enough to rerank exhaustively. At v0.2 scale (500 notes) the candidate pool will be top-100 from cosine; size is pre-registered before any results run.

## What this baseline doesn't test

- Whether prompted rationale-generation helps. That's Method 4.
- Whether Claude/GPT-class LLMs as rerankers outperform Voyage's native reranker. That's Method 3 vs this.
- Whether activation-space geometry surfaces things that no embedding sees. That's Method 6.
