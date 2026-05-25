# IS-RAG — Image-Schematic Retrieval-Augmented Generation

Pilot study augmenting dense retrieval with **image-schema cognitive boosting** (Lakoff & Johnson 1980, Talmy 1988) for Brazilian parliamentary discourse in Portuguese.

Companion code for the paper *"IS-RAG: Image-Schematic Retrieval-Augmented Generation for Cognitive Search in Political Discourse"*.

---

## Requirements

- Python 3.11+
- Docker & Docker Compose
- Anthropic API key (for the Cognitive Parser — `claude-haiku-4-5`)

---

## Setup

### API key

IS-RAG uses `claude-haiku-4-5` (Anthropic) as the Cognitive Parser — both during corpus ingestion (schema annotation) and at query time (query analysis). An Anthropic API key is required.

Create a `.env` file at the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

This file is already listed in `.gitignore`. Every script that calls the parser loads it automatically via `python-dotenv`.

### Install and start the database

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d          # PostgreSQL 16 + pgvector on port 5433
psql "host=localhost port=5433 dbname=is_rag_db user=is_rag_user password=is_rag_password" \
  -f schema.sql
```

---

## Reproducing the Experiment

### 1 — Corpus

The collected corpus (`src/data/corpus_camara_piloto.jsonl`, 121 speeches, 13 parties) is included in the repository. Skip to step 2 to use it directly.

To recollect from scratch:

```bash
cd src
python collect_corpus.py          # ~165 deputies across the ideological spectrum
python collect_corpus.py --dry-run  # preview collection plan without fetching
```

### 2 — Ingestion

Index the corpus into PostgreSQL:

```bash
cd src
python ingestion.py               # indexes all speeches in corpus_camara_piloto.jsonl
python ingestion.py --limit 20    # index first 20 speeches only (quick test)
```

### 3 — Search

Run IS-RAG hybrid search on a query:

```bash
cd src
python search.py "A inflação empurra as famílias para a crise" --mode hibrido
python search.py "barriers blocking constitutional reform" --mode hibrido --top 10
python search.py "query text" --mode texto      # text-only (no cognitive embedding)
python search.py "query text" --mode cognitivo  # cognitive embedding only
python search.py "query text" --baseline        # dense retrieval without boosting
```

### 4 — Evaluation (NDCG@5)

The annotated ground truth (`src/data/ground_truth_real.json`, 9 valid queries) is included.

```bash
cd src
python eval_real.py               # IS-RAG vs. baseline, NDCG@5, all 9 queries
python eval_real.py --top-k 10    # NDCG@10
python eval_real.py --workers 8   # increase parallelism
```

Expected output (hybrid mode, NDCG@5):

| System   | Mean NDCG@5 | Δ      |
|----------|-------------|--------|
| IS-RAG   | 0.7073      | +0.013 |
| Baseline | 0.6947      | —      |

### 5 — Robustness

```bash
cd src
python robust_eval.py             # query perturbation sensitivity
python robust_eval.py --query-id 3 --json-out report.json
```

---

## Repository Layout

```
src/
  data/
    corpus_camara_piloto.jsonl  # 121 speeches, 13 parties (2025)
    ground_truth_real.json      # 9 annotated queries (TREC-style pool, 0–3 relevance)
  coletor.py          # Chamber of Deputies API collector
  collect_corpus.py   # multi-party batch collection
  ingestion.py        # chunking → dual embeddings → PostgreSQL
  search.py           # IS-RAG query interface
  eval_real.py        # NDCG@K evaluation against ground truth
  robust_eval.py      # robustness to query surface variations
  annotate_pool.py    # LLM annotation of retrieval pools (ground truth construction)
  viewer.py           # chunk inspector (terminal, with schema filters)
paper/
  main.tex            # arXiv preprint (English)
  main-ptbr.tex       # preprint (Portuguese)
  references.bib
schema.sql            # PostgreSQL DDL (document_chunks table + HNSW/GIN indexes)
docker-compose.yml    # PostgreSQL 16 + pgvector
```

---

## Key Design Decisions

| Component | Choice | Rationale |
|---|---|---|
| Embedding model | `paraphrase-multilingual-mpnet-base-v2` (local) | Reproducible without API; validated Portuguese coverage |
| Cognitive Parser | `claude-haiku-4-5` | Fast, low-cost; shared prompt for indexing and query analysis |
| Boosting formula | $\hat{s} = s \cdot (1 + 0.4\,\sigma + 0.3\,\delta)$ | Schema match primary signal; domain match secondary |
| Vector store | PostgreSQL 16 + pgvector (HNSW) | Single dependency; GIN indexes for JSONB metadata filtering |
| Evaluation | TREC-style pooling + NDCG@5 | Handles graded relevance; standard IR benchmark protocol |
