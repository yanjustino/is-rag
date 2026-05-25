# IS-RAG — Image-Schematic Retrieval-Augmented Generation

A technique for hybrid RAG that extends standard dense retrieval with **Image Schemas** (Lakoff's Cognitive Linguistics), enabling the retrieval of documents that share deep cognitive structures — even when their vocabularies are entirely different.

The pilot corpus uses speeches from Brazilian federal deputies collected via the Open Data API of the Chamber of Deputies.

---

## Concept

Standard RAG fails when the user and the document express the same abstract concept through different physical metaphors (e.g., the user uses *boundary* language — CONTAINER — while the document uses *blockage* language — FORCE). IS-RAG addresses this by indexing, alongside the raw semantic embedding, a cognitive representation extracted by an LLM from three image-schema macroschemas:

| Schema | Typical concepts |
|---|---|
| **CONTAINER** | inside/outside, limits, containment, intrusion, enclosure |
| **PATH** | trajectory, direction, destination, progress, steps, stages |
| **FORCE** | barrier, impulse, resistance, attraction, blockage, shielding |

When the user's query and a retrieved chunk share the same schema, the similarity score receives a **Cognitive Boost** (×1.2 partial or ×1.4 full match).

### Storage structure

Each row in the `document_chunks` table combines three independent layers:

| Layer | Column | Content |
|---|---|---|
| **Semantic** | `embedding VECTOR(768)` | Raw text vector — topic/subject search |
| **Cognitive** | `cognitive_embedding VECTOR(768)` | Schema representation vector — structural search |
| **Cognitive** | `cognitive_metadata JSONB` | JSON with schemas, subtypes, anchor words, and target domain |
| **Contextual** | `source_metadata JSONB` | Source data: speaker, party, state, date, chunk index |

Separating the layers enables cross-dimensional queries that standard RAG cannot perform: filtering by target domain (`"Economy"`) and schema subtype (`"COMPULSION"`) within the same JSONB, combined with semantic vector search.

---

## Prerequisites

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`) available on PATH
- Docker and Docker Compose
- `venv` available in local Python

## Setup with `venv`

Create and activate a virtual environment in the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After that, use `python` to run scripts, or `./file.py` if you prefer to use the shebangs.

---

## Configuration

### 1. Database

Start PostgreSQL with pgvector via Docker:

```bash
docker compose up -d
```

Create the table and indexes:

```bash
psql "host=localhost port=5433 dbname=is_rag_db user=is_rag_user password=is_rag_password" \
  -f schema.sql
```

### 2. Environment variables

Optionally, create a `.env` file at the project root to centralise local configuration. This file is already excluded via `.gitignore`.

---

## Files

### `coletor.py` — Speech collection

Queries the Brazilian Chamber of Deputies Open Data API and saves speeches to `corpus_camara_piloto.jsonl`. Filters out texts shorter than 300 characters.

```bash
# Default collection: 20 deputies in alphabetical order
python coletor.py

# Filter by party
python coletor.py --partido PT

# Filter by state
python coletor.py --uf SP

# Skip the first 20 deputies (page 2)
python coletor.py --pagina 2

# Limit to 10 deputies from PL
python coletor.py --max 10 --partido PL

# Append to existing corpus without overwriting
python coletor.py --append
```

Output: `corpus_camara_piloto.jsonl` (one JSON line per valid speech).

---

### `ingestion.py` — Ingestion pipeline

Reads the JSONL file produced by the collector and runs the full pipeline for each speech:

1. **Semantic chunking** — splits text into ~150-word blocks respecting sentence boundaries
2. **Text embedding** — generates a 768d `embedding` with `paraphrase-multilingual-mpnet-base-v2` (local)
3. **Cognitive extraction** — calls the LLM to identify Image Schemas in the chunk and returns structured `cognitive_metadata` JSON
4. **Cognitive embedding** — generates a 768d `cognitive_embedding` from the textual serialisation of detected schemas
5. **Persistence** — inserts the three layers (semantic, cognitive, contextual) into PostgreSQL

The `source_metadata` field is built per speech and includes `doc_id`, `orador_nome`, `orador_partido`, `orador_uf`, `data_coleta`, and `chunk_index`.

```bash
# Process the first 20 speeches (default)
python ingestion.py

# Process 50 speeches
python ingestion.py --limit 50

# Skip the first 10 speeches
python ingestion.py --skip 10

# Append to the database without truncating the table
python ingestion.py --append

# Combining flags
python ingestion.py --limit 30 --skip 20 --append
```

> Requires `corpus_camara_piloto.jsonl` to exist. Run `coletor.py` first.

---

### `search.py` — IS-RAG hybrid search

Receives a natural-language query, detects Image Schemas via the LLM, generates the corresponding embeddings, and runs the search with Cognitive Boosting.

Three search modes are available:

| Mode | Description |
|---|---|
| `texto` | Raw text embedding + schema boosting (default) |
| `cognitivo` | Cognitive representation embedding + schema boosting |
| `hibrido` | 50/50 average of both embeddings + schema boosting |

```bash
# Basic search (text mode, top 5)
python search.py "bill approval process"

# Search with top 10 results
python search.py "bill approval process" --top 10

# Search with cognitive embedding
python search.py "barrier to reform progress" --mode cognitivo

# Hybrid search
python search.py "path to sustainable development" --mode hibrido
```

Each result shows: speaker, party/state, similarity score, final boosted score, and the document's cognitive schemas.

---

### `viewer.py` — Chunk inspector

Inspects chunks stored in the database with colour-formatted terminal output.

```bash
# List the first 10 chunks
python viewer.py

# List 20 chunks
python viewer.py --limit 20

# Filter by party
python viewer.py --partido PT

# Filter by cognitive schema
python viewer.py --schema FORCE
python viewer.py --schema PATH
python viewer.py --schema CONTAINER

# Show summary statistics only (total, by party, by schema)
python viewer.py --stats

# Combinations
python viewer.py --partido PL --schema CONTAINER --limit 5
```

---

### `eval_real.py` — Offline evaluation on the real corpus

Compares IS-RAG and the baseline using the annotated `ground_truth_real.json`, computing `NDCG@K` per query and overall mean.

```bash
# Standard evaluation
python eval_real.py

# Evaluation at top 10
python eval_real.py --top-k 10

# Force a single mode for all queries
python eval_real.py --force-mode hibrido
```

---

### `robust_eval.py` — Robustness evaluation via query perturbation

Measures sensitivity of the technique to small surface-level variations of the same search intent. The script generates deterministic variants of the queries in `ground_truth_real.json` and reports:

- `NDCG@K` drop relative to the original query
- stability of `top-1` and `top-k` composition
- stability of detected schema and expected subtype
- rate at which IS-RAG continues to outperform the baseline under perturbation

```bash
# Full robustness evaluation
python robust_eval.py

# Single query only
python robust_eval.py --query-id 3

# Save detailed report
python robust_eval.py --json-out robust_report.json
```

---

### `schema.sql` — Database schema

DDL to create the `document_chunks` table with pgvector support. Structures the three storage layers and creates four indexes:

| Index | Type | Column | Purpose |
|---|---|---|---|
| `idx_chunks_embedding` | HNSW | `embedding` | Semantic search by topic |
| `idx_chunks_cognitive_embedding` | HNSW | `cognitive_embedding` | Search by cognitive structure |
| `idx_chunks_cognitive_meta` | GIN | `cognitive_metadata` | Filter/boosting by image schema |
| `idx_chunks_source_meta` | GIN | `source_metadata` | Contextual filters (party, state, date) |

Apply manually before the first ingestion (see Configuration section).

---

### `docker-compose.yml` — Infrastructure

Starts a PostgreSQL 16 container with the `pgvector` extension pre-installed on port **5433**.

```bash
docker compose up -d    # start in background
docker compose down     # stop and remove the container
```

Data is persisted in a Docker volume (`postgres_data`) — survives `down` without `-v`.

---

### `corpus_camara_piloto.jsonl` — Collected corpus

JSON Lines file generated by `coletor.py`. Each line contains:

```json
{
  "id_interno": "CAMARA_<id>",
  "data_hora": "2025-03-15T14:00:00",
  "orador_nome": "Deputy Name",
  "orador_partido": "PT",
  "orador_uf": "SP",
  "texto": "Cleaned speech text..."
}
```

---

### `is_rag_development_plan.md` — Development plan

Conceptual specification document and project roadmap. Details the theoretical foundations (Lakoff, image schemas), data modelling, code examples, and experiment phases through arXiv pre-print submission.

---

## Full workflow

```
coletor.py  →  corpus_camara_piloto.jsonl  →  ingestion.py  →  PostgreSQL
                                                                      ↓
                                                               search.py / viewer.py
```

1. `python coletor.py` — collect speeches
2. `python ingestion.py` — process and index into the database
3. `python search.py "your query"` — IS-RAG hybrid search
4. `python viewer.py --stats` — inspect what was indexed
5. `python eval_real.py` — measure ranking gain on the annotated ground truth
6. `python robust_eval.py` — measure robustness to query variations
