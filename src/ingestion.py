#!/usr/bin/env python3
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
import anthropic
import psycopg
from psycopg import sql
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuração do banco de dados (conecta à porta 5433 do Docker)
DB_CONN = "host=localhost port=5433 dbname=is_rag_db user=is_rag_user password=is_rag_password"

try:
    from dotenv import load_dotenv

    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path, override=True)
except ImportError:
    pass

DB_TABLE = os.getenv("IS_RAG_TABLE", "document_chunks")

print("[*] Carregando modelo de embeddings...")
_embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
print("[√] Modelo de embeddings carregado.")


# 1. Definição das Estruturas Pydantic
class ImageSchemaDetail(BaseModel):
    schema_name: str = Field(description="Must be CONTAINER, PATH, or FORCE")
    sub_type: str = Field(
        description="Ex: BARRIER, COMPULSION, ENABLEMENT, INSIDE, OUTSIDE, SOURCE, TARGET"
    )
    anchor_word_pt: str = Field(
        description="A palavra ou expressão em português que disparou o esquema"
    )
    target_domain_pt: str = Field(
        description="O conceito abstrato ou domínio alvo em português sendo discutido"
    )


class CognitiveAnalysis(BaseModel):
    schemas: List[str] = Field(
        description="List of unique schema names present in the text (CONTAINER, PATH, FORCE)"
    )
    details: List[ImageSchemaDetail] = Field(
        description="Detailed image schematic mappings found in the text"
    )


# 2. Chunking Semântico baseado em Sentenças
def semantic_chunking(text: str, target_words: int = 150) -> List[str]:
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        if not sentence.strip():
            continue
        # Restaura o ponto final perdido no split
        sentence_clean = sentence.strip()
        if not sentence_clean.endswith("."):
            sentence_clean += "."

        words = len(sentence_clean.split())

        if current_word_count + words > target_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence_clean]
            current_word_count = words
        else:
            current_chunk.append(sentence_clean)
            current_word_count += words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


_QUERY_PROMPT_TEMPLATE = (
    Path(__file__).parent / "prompts" / "cognitive_analysis.md"
).read_text(encoding="utf-8")

_COGNITIVE_MODEL  = os.getenv("COGNITIVE_MODEL", "claude-haiku-4-5-20251001")
_MAX_WORKERS      = int(os.getenv("COGNITIVE_MAX_WORKERS", "5"))
_anthropic_client = anthropic.Anthropic()

_EMPTY = {"schemas": [], "details": []}


# 3. Extração cognitiva via Anthropic API (concorrente, 1 chunk por chamada)
def _extract_single(text: str) -> dict:
    """Envia um único chunk à API e retorna os metadados cognitivos."""
    prompt = _QUERY_PROMPT_TEMPLATE.replace("{text}", text)
    try:
        msg = _anthropic_client.messages.create(
            model=_COGNITIVE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        output = msg.content[0].text.strip()
        match  = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            print(f"\n[X] JSON não encontrado: {output[:200]}")
            return _EMPTY.copy()
        return CognitiveAnalysis(**json.loads(match.group())).model_dump()
    except Exception as e:
        print(f"\n[X] Erro na API cognitiva: {e}")
        return _EMPTY.copy()


def extract_cognitive_metadata_concurrent(texts: List[str]) -> List[dict]:
    """Processa chunks em paralelo preservando a ordem original."""
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(_extract_single, t): i for i, t in enumerate(texts)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"\n[X] Erro no chunk {idx}: {e}")
                results[idx] = _EMPTY.copy()
    return results


def get_embedding(text: str) -> List[float]:
    """Gera embedding vetorial 768d usando sentence-transformers (local)."""
    try:
        return _embedding_model.encode(text, convert_to_numpy=True).tolist()
    except Exception as e:
        print(f"\n[X] Erro ao gerar embedding: {e}")
        return []


def cognitive_text(metadata: dict) -> str:
    """Serializa os detalhes cognitivos em texto para embedding."""
    parts = []
    for d in metadata.get("details", []):
        parts.append(
            f"{d['schema_name']}: {d['anchor_word_pt']} → {d['target_domain_pt']}"
        )
    return " | ".join(parts) if parts else "sem esquema imagético"


# 4. Gravação no Postgres
def salvar_chunk(
    conn,
    content: str,
    embedding: List[float],
    cognitive_emb: List[float],
    cognitive_meta: dict,
    source_meta: dict,
):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
            INSERT INTO {}
                (content, embedding, cognitive_embedding, cognitive_metadata, source_metadata)
            VALUES (%s, %s, %s, %s, %s);
            """
            ).format(sql.Identifier(DB_TABLE)),
            (
                content,
                embedding,
                cognitive_emb,
                json.dumps(cognitive_meta),
                json.dumps(source_meta),
            ),
        )


def resolve_input_file(input_file: str | None = None) -> Path:
    if input_file:
        candidate = Path(input_file).expanduser()
        if candidate.is_absolute():
            return candidate

        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            return cwd_candidate

        return (Path(__file__).parent / candidate).resolve()

    return Path(__file__).parent / "data" / "corpus_camara_piloto.jsonl"


# 5. Execução Principal do Pipeline
def run_pipeline(
    limit_speeches: int = 20,
    skip: int = 0,
    append: bool = False,
    input_file: str | None = None,
    sample: int = 0,
):
    input_path = resolve_input_file(input_file)
    if not input_path.exists():
        print(
            f"[X] Arquivo {input_path} não encontrado! Execute o coletor.py primeiro."
        )
        return

    print(f"[*] Carregando discursos de {input_path}...")
    speeches = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            speeches.append(json.loads(line))

    print(f"[√] Carregados {len(speeches)} discursos no total.")

    if sample > 0:
        speeches = random.sample(speeches, min(sample, len(speeches)))
        print(f"[*] Amostra aleatória de {len(speeches)} discursos (seed não fixada — diversidade máxima).")
    else:
        if skip > 0:
            speeches = speeches[skip:]
            print(f"[*] Pulando os primeiros {skip} discursos (skip={skip}).")
        if limit_speeches > 0:
            speeches = speeches[:limit_speeches]

    print(f"[*] Processando {len(speeches)} discursos.")

    # Conectar ao banco de dados
    print("[*] Conectando ao PostgreSQL...")
    try:
        with psycopg.connect(DB_CONN) as conn:
            print("[√] Conexão com o banco estabelecida com sucesso.")

            if not append:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("TRUNCATE TABLE {};").format(sql.Identifier(DB_TABLE))
                    )
                    conn.commit()
                    print(f"[*] Tabela {DB_TABLE} truncada para nova ingestão.")
            else:
                print("[*] Modo append: mantendo chunks existentes.")

            total_chunks = 0
            for idx, speech in enumerate(speeches):
                orador = speech.get("orador_nome")
                partido = speech.get("orador_partido")
                uf = speech.get("orador_uf")
                texto = speech.get("texto")
                doc_id = speech.get("id_interno")
                data_hora = speech.get("data_hora", "")

                source_meta = {
                    "doc_id": doc_id,
                    "orador_nome": orador,
                    "orador_partido": partido,
                    "orador_uf": uf,
                    "data_coleta": data_hora,
                }

                # Chunking
                chunks = semantic_chunking(texto)
                print(f"\n[*] [{idx + 1}/{len(speeches)}] {orador} ({partido}-{uf}) → {len(chunks)} chunks")

                # Embeddings de texto (local, rápido)
                print("    embedding...", end=" ", flush=True)
                embeddings = [get_embedding(c) for c in chunks]
                print("ok")

                # Análise cognitiva concorrente (1 chunk por chamada, _MAX_WORKERS em paralelo)
                print(f"    cognitivo ({len(chunks)} chunks, workers={_MAX_WORKERS})...", end=" ", flush=True)
                cognitive_metas = extract_cognitive_metadata_concurrent(chunks)
                print("ok")

                # Persistência
                for chunk_idx, (chunk, embedding, cognitive_meta) in enumerate(
                    zip(chunks, embeddings, cognitive_metas)
                ):
                    if not embedding:
                        continue
                    cog_emb = get_embedding(cognitive_text(cognitive_meta))
                    salvar_chunk(
                        conn,
                        chunk,
                        embedding,
                        cog_emb,
                        cognitive_meta,
                        {**source_meta, "chunk_index": chunk_idx},
                    )
                    total_chunks += 1

                conn.commit()
                schemas_vistos = {s for m in cognitive_metas for s in m.get("schemas", [])}
                print(f"    [√] {len(chunks)} chunks salvos | schemas: {schemas_vistos} | total: {total_chunks}")

            print(
                f"\n[√] Pipeline concluído com sucesso. {total_chunks} chunks inseridos no PostgreSQL."
            )

    except Exception as e:
        print(f"[X] Erro no banco de dados: {e}")


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = 20
    skip = 0
    sample = 0
    append = "--append" in args
    input_file = None

    try:
        if "--sample" in args:
            sample = int(args[args.index("--sample") + 1])
        if "--limit" in args:
            limit = int(args[args.index("--limit") + 1])
        elif args and args[0].lstrip("-").isdigit():
            limit = int(args[0])
        if "--skip" in args:
            skip = int(args[args.index("--skip") + 1])
        if "--input-file" in args:
            input_file = args[args.index("--input-file") + 1]
    except (IndexError, ValueError):
        print("Uso: python ingestion.py [--limit N] [--skip N] [--sample N] [--append] [--input-file arquivo.jsonl]")
        sys.exit(1)

    run_pipeline(
        limit_speeches=limit,
        skip=skip,
        append=append,
        input_file=input_file,
        sample=sample,
    )
