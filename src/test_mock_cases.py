#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import psycopg
from psycopg import sql


DB_CONN = "host=localhost port=5433 dbname=is_rag_db user=is_rag_user password=is_rag_password"
DEFAULT_TABLE = "document_chunks_test"

# Relevância anotada: 3=altamente relevante, 2=relevante, 1=parcialmente, 0=irrelevante.
# Scores 0 podem ser omitidos (default). IDCG é calculado sobre todos os valores aqui anotados.
SCENARIOS = [
    {
        "name": "PATH / Economia como jornada",
        "mode": "hibrido",
        "expected_doc_id": "CAMARA_300101",
        "expected_schema": "PATH",
        "query": "Quais políticos descrevem processos econômicos ou reformas como uma jornada física ou movimento no espaço?",
        "relevance": {
            "CAMARA_300101": 3,  # PATH + economia → alvo exato
            "CAMARA_300106": 2,  # PATH + educação → mesmo esquema, domínio diferente
            "CAMARA_300104": 1,  # FORCE + economia → mesmo domínio, esquema diferente
        },
    },
    {
        "name": "FORCE / Obstrução legislativa",
        "mode": "texto",
        "expected_doc_id": "CAMARA_300102",
        "expected_schema": "FORCE",
        "query": "Quais discursos enquadram a atividade legislativa do parlamento sob a ótica de combate, obstrução física ou colisão de forças?",
        "relevance": {
            "CAMARA_300102": 3,  # FORCE + legislativo → alvo exato
            "CAMARA_300104": 2,  # FORCE + social/econômico → mesmo esquema
            "CAMARA_300107": 2,  # FORCE + ambiental → mesmo esquema, domínio diferente
            "CAMARA_300101": 1,  # PATH + economia → contexto político próximo
        },
    },
    {
        "name": "CONTAINER / Estado como estrutura fechada",
        "mode": "cognitivo",
        "expected_doc_id": "CAMARA_300103",
        "expected_schema": "CONTAINER",
        "query": "Busque metáforas onde o Estado ou as instituições públicas são representados como caixas, caçambas ou estruturas fechadas de aprisionamento.",
        "relevance": {
            "CAMARA_300103": 3,  # CONTAINER + Estado/burocracia → alvo exato
            "CAMARA_300105": 2,  # CONTAINER + segurança/fronteiras → mesmo esquema
            "CAMARA_300108": 2,  # CONTAINER + saúde → mesmo esquema, domínio diferente
            "CAMARA_300102": 1,  # FORCE + legislativo → contexto institucional próximo
        },
    },
    {
        # Cenário cross-domain: query usa vocabulário de dominação política/econômica
        # (sem mencionar força/pressão/bloqueio explicitamente nem vocabulário ambiental).
        # O documento esperado (300107) usa FORCE em domínio ambiental — vocabulário
        # completamente distinto da query. O baseline de texto falha por ausência de
        # sobreposição lexical; o modo cognitivo encontra via esquema FORCE compartilhado.
        "name": "FORCE cross-domain / Meio Ambiente via busca cognitiva",
        "mode": "cognitivo",
        "expected_doc_id": "CAMARA_300107",
        "expected_schema": "FORCE",
        "query": "Quais parlamentares descrevem a relação entre interesses privados e o interesse público como uma disputa em que um lado inevitavelmente cede ou é subjugado?",
        "relevance": {
            "CAMARA_300107": 3,  # FORCE + ambiental (agronegócio vs proteção) → alvo cross-domain
            "CAMARA_300102": 2,  # FORCE + legislativo (oposição vs governo) → mesmo esquema
            "CAMARA_300104": 2,  # FORCE + social (capital vs trabalhadores) → mesmo esquema, domínio próximo
            "CAMARA_300105": 1,  # CONTAINER + segurança → disputa/conflito presente mas esquema errado
        },
    },
]


# ── Métricas de Ranking ────────────────────────────────────────────────────────

def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain truncado em k."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def compute_ndcg(doc_ids: list[str], relevance_map: dict, k: int) -> float:
    """
    NDCG@K: normaliza pelo IDCG calculado sobre todos os scores anotados.
    doc_ids deve estar na ordem retornada pelo sistema (rank 1 primeiro).
    """
    rels = [relevance_map.get(did, 0) for did in doc_ids[:k]]
    ideal = sorted(relevance_map.values(), reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(rels, k) / idcg if idcg > 0 else 0.0


def compute_metrics(israg_results: list[dict], baseline_results: list[dict], relevance_map: dict, k: int) -> dict:
    """
    Computa NDCG@K comparando duas buscas independentes:
      - IS-RAG   : busca com análise cognitiva + boosting (modo do cenário)
      - Baseline : busca vetorial pura de texto, sem análise cognitiva
    """
    israg_ids    = [r.get("doc_id") for r in israg_results]
    baseline_ids = [r.get("doc_id") for r in baseline_results]

    return {
        "ndcg_israg":    compute_ndcg(israg_ids,    relevance_map, k),
        "ndcg_baseline": compute_ndcg(baseline_ids, relevance_map, k),
    }


# ── Infraestrutura ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roda a massa mock do IS-RAG em uma tabela isolada e valida cenários cognitivos."
    )
    parser.add_argument("--dataset", default="massa.jsonl", help="Arquivo JSONL com a massa de testes.")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Nome da tabela usada para o teste.")
    parser.add_argument("--top-k", type=int, default=5, help="Resultados recuperados por consulta (afeta NDCG@K).")
    parser.add_argument("--skip-ingestion", action="store_true", help="Pula a ingestão e reutiliza a tabela informada.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).parent / path).resolve()


def ensure_test_table(table_name: str):
    with psycopg.connect(DB_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding VECTOR(768) NOT NULL,
                        cognitive_embedding VECTOR(768),
                        cognitive_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        source_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    );
                    """
                ).format(sql.Identifier(table_name))
            )
        conn.commit()


def run_command(command: list[str], env: dict[str, str], cwd: Path, timeout: int = 600):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Comando falhou ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def ingest_dataset(dataset_path: Path, table_name: str):
    env = os.environ.copy()
    env["IS_RAG_TABLE"] = table_name
    run_command(
        [sys.executable, "ingestion.py", "--limit", "0", "--input-file", str(dataset_path)],
        env=env,
        cwd=Path(__file__).parent,
    )


def run_search(query: str, mode: str, top_k: int, table_name: str, baseline: bool = False) -> dict:
    env = os.environ.copy()
    env["IS_RAG_TABLE"] = table_name
    cmd = [sys.executable, "search.py", query, "--mode", mode, "--top", str(top_k), "--json"]
    if baseline:
        cmd.append("--baseline")
    result = run_command(cmd, env=env, cwd=Path(__file__).parent)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Falha ao interpretar JSON da busca.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from exc


# ── Avaliação ──────────────────────────────────────────────────────────────────

def evaluate_case(case: dict, top_k: int, table_name: str) -> dict:
    israg_report    = run_search(case["query"], case["mode"], top_k, table_name, baseline=False)
    baseline_report = run_search(case["query"], "texto",     top_k, table_name, baseline=True)

    israg_results    = israg_report.get("results", [])
    baseline_results = baseline_report.get("results", [])
    rel_map          = case.get("relevance", {})

    found = next((r for r in israg_results if r.get("doc_id") == case["expected_doc_id"]), None)
    top_doc_id    = israg_results[0].get("doc_id") if israg_results else None
    found_rank    = found.get("rank") if found else None
    found_schemas = found.get("cognitive_metadata", {}).get("schemas", []) if found else []

    baseline_found = next((r for r in baseline_results if r.get("doc_id") == case["expected_doc_id"]), None)
    baseline_rank  = baseline_found.get("rank") if baseline_found else None

    query_schema_ok = case["expected_schema"] in israg_report.get("detected_schemas", [])
    doc_schema_ok   = case["expected_schema"] in found_schemas
    found_in_top_k  = found is not None
    boost_ok        = found.get("boost_applied", False) if found else False
    passed          = query_schema_ok and doc_schema_ok and found_in_top_k and boost_ok

    metrics = compute_metrics(israg_results, baseline_results, rel_map, top_k)

    return {
        "name":            case["name"],
        "mode":            case["mode"],
        "expected_doc_id": case["expected_doc_id"],
        "expected_schema": case["expected_schema"],
        "top_doc_id":      top_doc_id,
        "found_rank":      found_rank,
        "baseline_rank":   baseline_rank,
        "query_schemas":   israg_report.get("detected_schemas", []),
        "found_schemas":   found_schemas,
        "boost_applied":   found.get("boost_applied") if found else None,
        "boost_ratio":     found.get("boost_ratio") if found else None,
        "passed":          passed,
        "ndcg_israg":      metrics["ndcg_israg"],
        "ndcg_baseline":   metrics["ndcg_baseline"],
        "report":          israg_report,
    }


def print_case_result(result: dict, k: int):
    status      = "PASSOU" if result["passed"] else "FALHOU"
    found_rank  = result["found_rank"]    if result["found_rank"]    is not None else "fora"
    base_rank   = result["baseline_rank"] if result["baseline_rank"] is not None else "fora"
    boost_ratio = result.get("boost_ratio")
    boost_label = f"×{boost_ratio:.2f}" if boost_ratio and boost_ratio > 1.01 else "NÃO"
    ndcg_gain   = result["ndcg_israg"] - result["ndcg_baseline"]

    print(f"[{status}] {result['name']}")
    print(f"  modo={result['mode']} | esperado={result['expected_doc_id']} | topo={result['top_doc_id']}")
    print(f"  rank IS-RAG={found_rank} | rank Baseline={base_rank} | boost={boost_label}")
    print(f"  schema_query={result['query_schemas']} | schema_doc={result['found_schemas']}")
    print(f"  NDCG@{k}: IS-RAG={result['ndcg_israg']:.4f}  Baseline={result['ndcg_baseline']:.4f}  Δ={ndcg_gain:+.4f}")


def print_metrics_summary(results: list[dict], k: int):
    n = len(results)
    mean_ndcg_israg    = sum(r["ndcg_israg"]    for r in results) / n
    mean_ndcg_baseline = sum(r["ndcg_baseline"] for r in results) / n
    ndcg_delta         = mean_ndcg_israg - mean_ndcg_baseline

    w = 42
    print("─" * w)
    print(f"{'Métrica':<14} {'IS-RAG':>8} {'Baseline':>10} {'Δ':>8}")
    print("─" * w)
    print(f"{'NDCG@'+str(k):<14} {mean_ndcg_israg:>8.4f} {mean_ndcg_baseline:>10.4f} {ndcg_delta:>+8.4f}")
    print("─" * w)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    if not dataset_path.exists():
        print(f"[X] Dataset não encontrado: {dataset_path}", file=sys.stderr)
        return 1

    ensure_test_table(args.table)

    if not args.skip_ingestion:
        print(f"[*] Ingerindo massa de testes em {args.table}...")
        ingest_dataset(dataset_path, args.table)

    print(f"[*] Rodando validações sobre {args.table} (top-k={args.top_k})...\n")
    results = [evaluate_case(case, args.top_k, args.table) for case in SCENARIOS]

    failures = 0
    for result in results:
        print_case_result(result, args.top_k)
        print()
        if not result["passed"]:
            failures += 1

    passed_count = len(results) - failures
    status_icon = "[√]" if failures == 0 else "[X]"
    print(f"{status_icon} Cenários aprovados: {passed_count}/{len(results)}\n")

    print_metrics_summary(results, args.top_k)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
