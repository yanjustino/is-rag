#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
import anthropic
import psycopg
from psycopg import sql
from sentence_transformers import SentenceTransformer

# Configuração do banco de dados (conecta à porta 5433 do Docker)
DB_CONN = "host=localhost port=5433 dbname=is_rag_db user=is_rag_user password=is_rag_password"

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path, override=True)
except ImportError:
    pass

DB_TABLE         = os.getenv("IS_RAG_TABLE",     "document_chunks")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

print(f"[*] Carregando modelo de embeddings: {_EMBEDDING_MODEL}...", file=sys.stderr)
_embedding_model = SentenceTransformer(_EMBEDDING_MODEL)
print("[√] Modelo de embeddings carregado.", file=sys.stderr)

# 1. Definição das Estruturas Pydantic
class ImageSchemaDetail(BaseModel):
    schema_name: str = Field(description="Must be CONTAINER, PATH, or FORCE")
    sub_type: str = Field(description="Ex: BARRIER, COMPULSION, ENABLEMENT, INSIDE, OUTSIDE, SOURCE, TARGET")
    anchor_word_pt: str = Field(description="A palavra ou expressão em português que disparou o esquema")
    target_domain_pt: str = Field(description="O conceito abstrato ou domínio alvo em português sendo discutido")

class CognitiveAnalysis(BaseModel):
    schemas: List[str] = Field(description="List of unique schema names present in the text (CONTAINER, PATH, FORCE)")
    details: List[ImageSchemaDetail] = Field(description="Detailed image schematic mappings found in the text")

_QUERY_PROMPT_TEMPLATE = (
    Path(__file__).parent / "prompts" / "cognitive_analysis.md"
).read_text(encoding="utf-8")

_COGNITIVE_MODEL  = os.getenv("COGNITIVE_MODEL", "claude-haiku-4-5-20251001")
_anthropic_client = anthropic.Anthropic()

# 2. Análise da Query
def analyze_query(query_text: str) -> dict:
    """Detecta esquemas e sub_types da query via Anthropic API."""
    prompt = _QUERY_PROMPT_TEMPLATE.replace("{text}", query_text)
    empty = {"schemas": [], "sub_types": [], "domains": [], "details": []}
    try:
        msg = _anthropic_client.messages.create(
            model=_COGNITIVE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        output = msg.content[0].text.strip()
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            data = json.loads(match.group())
            analysis = CognitiveAnalysis(**data)
            return {
                "schemas":   analysis.schemas,
                "sub_types": [d.sub_type for d in analysis.details],
                "domains":   list({d.target_domain_pt for d in analysis.details}),
                "details":   [d.model_dump() for d in analysis.details],
            }
        else:
            print(f"[X] JSON não encontrado na resposta: {output[:200]}", file=sys.stderr)
            return empty
    except Exception as e:
        print(f"[X] Erro ao analisar query: {e}", file=sys.stderr)
        return empty

def get_embedding(text: str) -> List[float]:
    """Gera embedding vetorial para a query usando sentence-transformers (local)."""
    try:
        return _embedding_model.encode(text, convert_to_numpy=True).tolist()
    except Exception as e:
        print(f"[X] Erro ao gerar embedding: {e}", file=sys.stderr)
        return []

def cognitive_text_from_query(details: list) -> str:
    """Serializa detalhes cognitivos da query no mesmo formato usado na ingestão:
    'SCHEMA: anchor_word → domain | SCHEMA: anchor_word → domain'
    """
    parts = [
        f"{d['schema_name']}: {d['anchor_word_pt']} → {d['target_domain_pt']}"
        for d in details
    ]
    return " | ".join(parts) if parts else "sem esquema imagético"


def normalize_results(rows) -> list[dict]:
    normalized = []
    for rank, res in enumerate(rows, start=1):
        cid, source_meta, content, metadata, sim_score, final_score = res
        ratio = final_score / sim_score if sim_score > 0 else 1.0
        normalized.append(
            {
                "rank": rank,
                "chunk_id": cid,
                "doc_id": source_meta.get("doc_id"),
                "source_metadata": source_meta,
                "content": content,
                "cognitive_metadata": metadata,
                "vector_similarity": sim_score,
                "final_score": final_score,
                "boost_ratio": ratio,
                "boost_applied": ratio > 1.01,
            }
        )
    return normalized


def print_search_report(report: dict):
    results = report.get("results", [])
    print(f"\n[√] Resultados da Busca (Top {len(results)}):")
    print("=" * 80)
    for item in results:
        source_meta = item["source_metadata"]
        metadata = item["cognitive_metadata"]
        orador = source_meta.get("orador_nome")
        partido = source_meta.get("orador_partido")
        uf = source_meta.get("orador_uf")
        if item["boost_applied"]:
            boost_label = f"×{item['boost_ratio']:.2f}"
        else:
            boost_label = "NÃO"

        print(f"Rank {item['rank']}: {orador} ({partido}-{uf}) - ID: {item['chunk_id']}")
        print(f" -> Similaridade Vetorial: {item['vector_similarity']:.4f}")
        print(f" -> Score Final Boosted:  {item['final_score']:.4f} [Boost: {boost_label}]")
        print(f" -> Esquemas no Documento: {metadata.get('schemas', [])}")
        print(f" -> Trecho:\n    \"{item['content']}\"")
        print("-" * 80)

# 3. Busca
def metaphorical_search(
    query_text: str,
    top_k: int = 5,
    mode: str = "texto",
    verbose: bool = True,
    baseline: bool = False,
    precomputed_analysis: dict = None,
) -> dict:
    """
    Modos:
      texto    — embedding do texto bruto (comportamento original)
      cognitivo — embedding dos detalhes cognitivos
      hibrido  — média dos dois scores

    baseline=True pula a análise cognitiva e executa busca vetorial pura (RAG tradicional).
    """
    if baseline:
        detected_schemas   = []
        detected_sub_types = []
        detected_domains   = []
        detected_details   = []
        if verbose:
            print(f"[*] Busca baseline (vetorial pura, sem análise cognitiva): '{query_text}'")
    elif precomputed_analysis is not None:
        detected_schemas   = precomputed_analysis["schemas"]
        detected_sub_types = precomputed_analysis["sub_types"]
        detected_domains   = precomputed_analysis["domains"]
        detected_details   = precomputed_analysis["details"]
        if verbose:
            print(f"[*] Usando análise cognitiva pré-computada: '{query_text}'")
            print(f"    [+] Esquemas detectados  : {detected_schemas}")
            print(f"    [+] Sub-tipos detectados : {detected_sub_types}")
            print(f"    [+] Domínios detectados  : {detected_domains}")
    else:
        if verbose:
            print(f"[*] Analisando estrutura cognitiva da busca: '{query_text}'")
        query_analysis     = analyze_query(query_text)
        detected_schemas   = query_analysis["schemas"]
        detected_sub_types = query_analysis["sub_types"]
        detected_domains   = query_analysis["domains"]
        detected_details   = query_analysis["details"]
        if verbose:
            print(f"    [+] Esquemas detectados  : {detected_schemas}")
            print(f"    [+] Sub-tipos detectados : {detected_sub_types}")
            print(f"    [+] Domínios detectados  : {detected_domains}")
    if verbose:
        print(f"    [+] Modo de busca        : {'BASELINE' if baseline else mode.upper()}")

    query_embedding = get_embedding(query_text)
    if not query_embedding:
        error = "[X] Não foi possível gerar o embedding da busca."
        if verbose:
            print(error)
        return {
            "query": query_text,
            "mode": mode,
            "detected_schemas": detected_schemas,
            "detected_sub_types": detected_sub_types,
            "detected_domains": detected_domains,
            "results": [],
            "error": error,
        }

    cog_query_text = cognitive_text_from_query(detected_details)
    cog_query_embedding = get_embedding(cog_query_text)

    try:
        with psycopg.connect(DB_CONN) as conn:
            with conn.cursor() as cur:
                if not detected_schemas:
                    if verbose:
                        print("[*] Nenhum esquema cognitivo detectado. Busca vetorial tradicional...")
                    _threshold = 0.2 if mode in ("cognitivo", "hibrido") else 0.3
                    query_sql = sql.SQL(
                        """
                    SELECT
                        id, source_metadata, content, cognitive_metadata,
                        (1 - (embedding <=> %s::vector)) AS vector_similarity,
                        (1 - (embedding <=> %s::vector)) AS final_score
                    FROM {}
                    WHERE (1 - (embedding <=> %s::vector)) > {}
                    ORDER BY final_score DESC
                    LIMIT %s;
                    """
                    ).format(sql.Identifier(DB_TABLE), sql.Literal(_threshold))
                    cur.execute(query_sql, (query_embedding, query_embedding, query_embedding, top_k))
                else:
                    if mode == "cognitivo":
                        if verbose:
                            print(f"[*] Busca por COGNITIVE_EMBEDDING com boosting para schemas: {detected_schemas}...")
                        query_sql = sql.SQL(
                            """
                        WITH cog_search AS (
                            SELECT
                                id, source_metadata, content, cognitive_metadata,
                                (1 - (cognitive_embedding <=> %s::vector)) AS vector_similarity
                            FROM {}
                            WHERE cognitive_embedding IS NOT NULL
                        ),
                        schema_match AS (
                            SELECT *,
                                (SELECT COUNT(*) FROM jsonb_array_elements_text(cognitive_metadata->'schemas') s
                                 WHERE s = ANY(%s::text[])) AS matched_schemas,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'sub_type' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS subtype_proportion,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'schema_name' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS schema_proportion,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'target_domain_pt' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS domain_proportion
                            FROM cog_search
                        )
                        SELECT id, source_metadata, content, cognitive_metadata,
                               vector_similarity,
                               CASE
                                   WHEN matched_schemas > 0
                                       THEN vector_similarity * (1.0 + 0.4 * schema_proportion
                                                                      + 0.3 * COALESCE(domain_proportion, 0))
                                   ELSE vector_similarity
                               END AS final_score
                        FROM schema_match
                        WHERE vector_similarity > 0.2
                        ORDER BY final_score DESC
                        LIMIT %s;
                        """
                        ).format(sql.Identifier(DB_TABLE))
                        cur.execute(query_sql, (cog_query_embedding, detected_schemas, detected_sub_types, detected_schemas, detected_domains, top_k))

                    elif mode == "hibrido":
                        if verbose:
                            print(f"[*] Busca HÍBRIDA (texto + cognitivo) com boosting para schemas: {detected_schemas}...")
                        query_sql = sql.SQL(
                            """
                        WITH base AS (
                            SELECT
                                id, source_metadata, content, cognitive_metadata,
                                (1 - (embedding           <=> %s::vector)) AS text_sim,
                                CASE WHEN cognitive_embedding IS NOT NULL
                                     THEN (1 - (cognitive_embedding <=> %s::vector))
                                     ELSE 0 END                             AS cog_sim
                            FROM {}
                        ),
                        combined AS (
                            SELECT *,
                                (text_sim * 0.5 + cog_sim * 0.5) AS vector_similarity,
                                (SELECT COUNT(*) FROM jsonb_array_elements_text(cognitive_metadata->'schemas') s
                                 WHERE s = ANY(%s::text[])) AS matched_schemas,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'sub_type' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS subtype_proportion,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'schema_name' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS schema_proportion,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'target_domain_pt' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS domain_proportion
                            FROM base
                        )
                        SELECT id, source_metadata, content, cognitive_metadata,
                               vector_similarity,
                               CASE
                                   WHEN matched_schemas > 0
                                       THEN vector_similarity * (1.0 + 0.4 * schema_proportion
                                                                      + 0.3 * COALESCE(domain_proportion, 0))
                                   ELSE vector_similarity
                               END AS final_score
                        FROM combined
                        WHERE vector_similarity > 0.2
                        ORDER BY final_score DESC
                        LIMIT %s;
                        """
                        ).format(sql.Identifier(DB_TABLE))
                        cur.execute(query_sql, (query_embedding, cog_query_embedding,
                                                detected_schemas, detected_sub_types, detected_schemas, detected_domains, top_k))

                    else:  # modo "texto" (padrão original)
                        if verbose:
                            print(f"[*] Busca por TEXTO com boosting para schemas: {detected_schemas}...")
                        query_sql = sql.SQL(
                            """
                        WITH vector_search AS (
                            SELECT
                                id, source_metadata, content, cognitive_metadata,
                                (1 - (embedding <=> %s::vector)) AS vector_similarity
                            FROM {}
                        ),
                        schema_match AS (
                            SELECT *,
                                (SELECT COUNT(*) FROM jsonb_array_elements_text(cognitive_metadata->'schemas') s
                                 WHERE s = ANY(%s::text[])) AS matched_schemas,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'sub_type' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS subtype_proportion,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'schema_name' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS schema_proportion,
                                (SELECT COUNT(*) FROM jsonb_array_elements(cognitive_metadata->'details') d
                                 WHERE d->>'target_domain_pt' = ANY(%s::text[]))::float
                                 / NULLIF(jsonb_array_length(cognitive_metadata->'details'), 0) AS domain_proportion
                            FROM vector_search
                        )
                        SELECT id, source_metadata, content, cognitive_metadata,
                               vector_similarity,
                               CASE
                                   WHEN matched_schemas > 0
                                       THEN vector_similarity * (1.0 + 0.4 * schema_proportion
                                                                      + 0.3 * COALESCE(domain_proportion, 0))
                                   ELSE vector_similarity
                               END AS final_score
                        FROM schema_match
                        WHERE vector_similarity > 0.3
                        ORDER BY final_score DESC
                        LIMIT %s;
                        """
                        ).format(sql.Identifier(DB_TABLE))
                        cur.execute(query_sql, (query_embedding, detected_schemas, detected_sub_types, detected_schemas, detected_domains, top_k))

                results = cur.fetchall()
                report = {
                    "query": query_text,
                    "mode": mode,
                    "detected_schemas": detected_schemas,
                    "detected_sub_types": detected_sub_types,
                    "detected_domains": detected_domains,
                    "detected_details": detected_details,
                    "results": normalize_results(results),
                }
                if verbose:
                    print_search_report(report)
                return report

    except Exception as e:
        error = f"[X] Erro na busca do banco: {e}"
        if verbose:
            print(error)
        return {
            "query": query_text,
            "mode": mode,
            "detected_schemas": detected_schemas,
            "detected_sub_types": detected_sub_types,
            "detected_domains": detected_domains,
            "results": [],
            "error": error,
        }

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "--help" in args:
        print("Uso: python search.py \"query\" [--top N] [--mode texto|cognitivo|hibrido] [--json]")
        sys.exit(0 if "--help" in args else 1)

    query = args[0]
    k    = 5
    mode = "texto"
    json_output         = "--json" in args
    baseline            = "--baseline" in args
    precomputed_analysis = None

    try:
        if "--top" in args:
            k = int(args[args.index("--top") + 1])
        if "--mode" in args:
            mode = args[args.index("--mode") + 1]
        if "--precomputed-analysis" in args:
            precomputed_analysis = json.loads(args[args.index("--precomputed-analysis") + 1])
    except (IndexError, ValueError) as e:
        print(f"Uso: python search.py \"query\" [--top N] [--mode texto|cognitivo|hibrido] [--baseline] [--precomputed-analysis JSON] [--json]: {e}")
        sys.exit(1)

    report = metaphorical_search(query, top_k=k, mode=mode, verbose=not json_output, baseline=baseline, precomputed_analysis=precomputed_analysis)
    if json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
