#!/usr/bin/env python3
"""
eval_real.py — Avaliação IS-RAG no corpus real usando ground_truth_real.json.

Carrega as queries anotadas pelo annotate_pool.py, exclui as que têm IDCG=0
(zero documentos com score ≥ 2), executa IS-RAG e baseline em paralelo para
todas as queries válidas e computa NDCG@K comparativo.

Uso:
  python eval_real.py
  python eval_real.py --top-k 10
  python eval_real.py --workers 4
  python eval_real.py --ground-truth outro.json
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

BASE_DIR      = Path(__file__).parent
DEFAULT_GT    = BASE_DIR / "data" / "ground_truth_real.json"
DEFAULT_TABLE = "document_chunks"

# Importa search diretamente — modelo carrega uma única vez aqui.
sys.path.insert(0, str(BASE_DIR))
from search import metaphorical_search


# ---------------------------------------------------------------------------
# Métricas (idênticas ao test_mock_cases.py)
# ---------------------------------------------------------------------------

def _dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def compute_ndcg(doc_ids: list[str], relevance_map: dict, k: int) -> float:
    rels  = [relevance_map.get(did, 0) for did in doc_ids[:k]]
    ideal = sorted(relevance_map.values(), reverse=True)
    idcg  = _dcg(ideal, k)
    return _dcg(rels, k) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Carregamento do ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> list[dict]:
    """
    Lê ground_truth_real.json e devolve lista de cenários prontos para avaliação.
    Cada cenário tem: id, text, schema, domain, mode, relevance_map.
    Exclui queries com IDCG=0 (nenhum chunk com score ≥ 2).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = []
    excluded  = []

    for qid, q in data["queries"].items():
        # Usa max() por doc_id: múltiplos chunks do mesmo discurso no pool
        # não devem sobrescrever relevância maior com relevância menor.
        rel_map: dict[str, int] = {}
        for item in q["pool"]:
            if item.get("relevance") and item["relevance"] > 0 and item.get("doc_id"):
                doc_id = item["doc_id"]
                rel_map[doc_id] = max(rel_map.get(doc_id, 0), item["relevance"])
        has_relevant = any(v >= 2 for v in rel_map.values())

        if not has_relevant:
            excluded.append((qid, q["schema"]))
            continue

        scenarios.append({
            "id":                q["id"],
            "name":              f"Q{q['id']:>2} — {q['schema']} / {q['domain']}",
            "schema":            q["schema"],
            "domain":            q["domain"],
            "mode":              q["mode"],
            "text":              q["text"],
            "relevance_map":     rel_map,
            "cognitive_analysis": q.get("cognitive_analysis"),
        })

    if excluded:
        print(f"[!] {len(excluded)} queries excluídas (IDCG=0, sem documentos relevantes):")
        for qid, schema in excluded:
            print(f"    Q{qid}: {schema}")
        print()

    return scenarios


# ---------------------------------------------------------------------------
# Busca via import direto (modelo carregado uma vez, sem overhead de subprocess)
# ---------------------------------------------------------------------------

def run_search(
    query_text: str,
    mode: str,
    top_k: int,
    table: str,
    baseline: bool = False,
    precomputed_analysis: dict = None,
) -> dict:
    return metaphorical_search(
        query_text=query_text,
        top_k=top_k,
        mode=mode,
        verbose=False,
        baseline=baseline,
        precomputed_analysis=precomputed_analysis,
    )


# ---------------------------------------------------------------------------
# Avaliação concorrente
# ---------------------------------------------------------------------------

def debug_schemas(scenarios: list[dict], table: str):
    """Roda cada query em modo IS-RAG e imprime apenas os schemas detectados."""
    print("[debug] Schemas detectados por query (sem boosting aplicado):\n")
    for sc in scenarios:
        report = run_search(sc["text"], sc["mode"], 1, table, baseline=False)
        detected = report.get("detected_schemas", [])
        sub_types = report.get("detected_sub_types", [])
        print(f"  Q{sc['id']:>2} [{sc['schema']:30}] → schemas={detected}  sub_types={sub_types}")
    print()


def evaluate_all(
    scenarios: list[dict],
    top_k: int,
    table: str,
    max_workers: int,
) -> list[dict]:
    """
    Submete IS-RAG + baseline de todas as queries ao ThreadPoolExecutor.
    Cada future é identificada por (scenario_id, is_baseline).
    """
    futures = {}
    results_raw = {}  # scenario_id → {"israg": report, "baseline": report}

    print(f"[*] Executando {len(scenarios)} queries × 2 buscas com {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sc in scenarios:
            sid = sc["id"]
            results_raw[sid] = {}
            futures[ex.submit(run_search, sc["text"], sc["mode"], top_k, table, False, sc.get("cognitive_analysis"))] = (sid, False)
            futures[ex.submit(run_search, sc["text"], sc["mode"], top_k, table, True)]  = (sid, True)

        done = 0
        total = len(futures)
        for future in as_completed(futures):
            sid, is_baseline = futures[future]
            label = "baseline" if is_baseline else "IS-RAG  "
            try:
                results_raw[sid]["baseline" if is_baseline else "israg"] = future.result()
                done += 1
                print(f"  [{done:>2}/{total}] Q{sid:>2} {label} — ok")
            except Exception as e:
                results_raw[sid]["baseline" if is_baseline else "israg"] = {"results": []}
                print(f"  [{done:>2}/{total}] Q{sid:>2} {label} — ERRO: {e}")

    print()

    # Monta resultados finais na ordem dos cenários
    evaluated = []
    for sc in scenarios:
        sid       = sc["id"]
        raw       = results_raw[sid]
        israg_r   = raw.get("israg",    {})
        base_r    = raw.get("baseline", {})
        rel_map   = sc["relevance_map"]

        # Deduplica por doc_id mantendo a primeira ocorrência (maior score).
        # Sem isso, dois chunks do mesmo discurso somam DCG além do IDCG → NDCG > 1.
        def _dedup(results: list[dict]) -> list[str]:
            seen: set[str] = set()
            ids: list[str] = []
            for r in results:
                did = r.get("doc_id")
                if did and did not in seen:
                    seen.add(did)
                    ids.append(did)
            return ids

        israg_ids    = _dedup(israg_r.get("results", []))
        baseline_ids = _dedup(base_r.get("results",  []))

        ndcg_israg    = compute_ndcg(israg_ids,    rel_map, top_k)
        ndcg_baseline = compute_ndcg(baseline_ids, rel_map, top_k)

        evaluated.append({
            "id":               sid,
            "name":             sc["name"],
            "mode":             sc["mode"],
            "schema":           sc["schema"],
            "detected_schemas": israg_r.get("detected_schemas", []),
            "ndcg_israg":       ndcg_israg,
            "ndcg_baseline":    ndcg_baseline,
            "delta":            ndcg_israg - ndcg_baseline,
            "n_relevant":       sum(1 for v in rel_map.values() if v >= 2),
        })

    return evaluated


# ---------------------------------------------------------------------------
# Apresentação
# ---------------------------------------------------------------------------

def print_results(results: list[dict], k: int):
    print("=" * 76)
    print(f"{'Cenário':<38} {'Modo':>8} {'IS-RAG':>8} {'Baseline':>10} {'Δ':>8}")
    print("=" * 76)
    for r in results:
        name = r["name"][:37]
        print(f"{name:<38} {r['mode']:>8} {r['ndcg_israg']:>8.4f} {r['ndcg_baseline']:>10.4f} {r['delta']:>+8.4f}")
    print("─" * 76)

    n = len(results)
    mean_israg    = sum(r["ndcg_israg"]    for r in results) / n
    mean_baseline = sum(r["ndcg_baseline"] for r in results) / n
    delta         = mean_israg - mean_baseline
    print(f"{'Média (' + str(n) + ' queries)':<38} {'—':>8} {mean_israg:>8.4f} {mean_baseline:>10.4f} {delta:>+8.4f}")
    print("=" * 76)
    print(f"\nNDCG@{k}  IS-RAG={mean_israg:.4f}  Baseline={mean_baseline:.4f}  Δ={delta:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliação IS-RAG no corpus real.")
    p.add_argument("--ground-truth",   default=str(DEFAULT_GT), help="Caminho para ground_truth_real.json")
    p.add_argument("--table",          default=DEFAULT_TABLE,    help="Tabela PostgreSQL a usar")
    p.add_argument("--top-k",          type=int, default=5,      help="NDCG@K (default 5)")
    p.add_argument("--workers",        type=int, default=4,      help="Workers concorrentes (default 4)")
    p.add_argument("--force-mode",     default=None,             help="Força um modo para todas as queries (texto|cognitivo|hibrido)")
    p.add_argument("--debug-schemas",  action="store_true",      help="Imprime schemas detectados por query antes de buscar")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gt_path = Path(args.ground_truth)

    if not gt_path.exists():
        print(f"[X] Ground truth não encontrado: {gt_path}", file=sys.stderr)
        return 1

    print(f"[*] Carregando ground truth: {gt_path}")
    scenarios = load_ground_truth(gt_path)

    if not scenarios:
        print("[X] Nenhuma query válida após filtragem.", file=sys.stderr)
        return 1

    if args.force_mode:
        for sc in scenarios:
            sc["mode"] = args.force_mode
        print(f"[!] Modo forçado: {args.force_mode} para todas as queries\n")

    print(f"[*] {len(scenarios)} queries válidas | tabela={args.table} | top-k={args.top_k}\n")

    if args.debug_schemas:
        debug_schemas(scenarios, args.table)

    results = evaluate_all(scenarios, args.top_k, args.table, args.workers)
    print_results(results, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
