#!/usr/bin/env python3
"""
robust_eval.py — Avaliação de robustez do IS-RAG por perturbação de queries.

Objetivo:
  medir o quanto o ranking e a análise cognitiva permanecem estáveis quando a
  mesma intenção de busca é expressa com pequenas variações superficiais.

Estratégia:
  1. Carrega o mesmo ground truth usado em eval_real.py.
  2. Gera variantes determinísticas e leves para cada query.
  3. Executa IS-RAG e baseline para cada variante.
  4. Mede:
     - NDCG@K médio e pior caso
     - queda relativa ao enunciado original
     - taxa de preservação do top-1
     - sobreposição média do top-k (Jaccard)
     - estabilidade da detecção de schema/subtipo esperado

Uso:
  python robust_eval.py
  python robust_eval.py --top-k 10 --workers 4
  python robust_eval.py --query-id 3
  python robust_eval.py --json-out robust_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.eval_real import (
    DEFAULT_GT,
    DEFAULT_TABLE,
    compute_ndcg,
    load_ground_truth,
    run_search,
)

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\sÀ-ÿ-]", re.UNICODE)

SYNONYM_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bbloqueados\b", re.IGNORECASE), "barrados"),
    (re.compile(r"\btravados\b", re.IGNORECASE), "paralisados"),
    (re.compile(r"\bimpedem\b", re.IGNORECASE), "obstruem"),
    (re.compile(r"\bavanço\b", re.IGNORECASE), "progresso"),
    (re.compile(r"\bpressiona\b", re.IGNORECASE), "aperta"),
    (re.compile(r"\bempurra\b", re.IGNORECASE), "conduz"),
    (re.compile(r"\bsufocando\b", re.IGNORECASE), "asfixiando"),
    (re.compile(r"\bembate\b", re.IGNORECASE), "conflito"),
    (re.compile(r"\bse chocam\b", re.IGNORECASE), "colidem"),
    (re.compile(r"\bresiste\b", re.IGNORECASE), "faz resistência"),
    (re.compile(r"\binvade\b", re.IGNORECASE), "ultrapassa"),
    (re.compile(r"\blimites\b", re.IGNORECASE), "fronteiras"),
    (re.compile(r"\bexcluídos\b", re.IGNORECASE), "deixados de fora"),
    (re.compile(r"\bfora\b", re.IGNORECASE), "do lado de fora"),
    (re.compile(r"\bavança\b", re.IGNORECASE), "segue"),
    (re.compile(r"\bcaminho\b", re.IGNORECASE), "trajeto"),
    (re.compile(r"\brumo\b", re.IGNORECASE), "direção"),
    (re.compile(r"\borigens\b", re.IGNORECASE), "raízes"),
    (re.compile(r"\bponto de partida\b", re.IGNORECASE), "marco inicial"),
    (re.compile(r"\bdesviou\b", re.IGNORECASE), "tirou"),
]


def normalize_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def strip_punctuation(text: str) -> str:
    return normalize_spaces(PUNCT_RE.sub(" ", text))


def lowercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def apply_synonym_swaps(text: str) -> str:
    updated = text
    for pattern, replacement in SYNONYM_RULES:
        updated = pattern.sub(replacement, updated, count=1)
    return normalize_spaces(updated)


def build_variants(text: str) -> list[dict[str, str]]:
    candidates = [
        ("original", text),
        ("lowercase", text.lower()),
        ("sem_pontuacao", strip_punctuation(text)),
        (
            "prefixo_contextual",
            f"No debate parlamentar brasileiro, {lowercase_first(text)}",
        ),
        ("sufixo_contextual", f"{text.rstrip('.')} no contexto político atual."),
        ("troca_lexical", apply_synonym_swaps(text)),
    ]

    variants: list[dict[str, str]] = []
    seen: set[str] = set()
    for name, content in candidates:
        normalized = normalize_spaces(content)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        variants.append({"name": name, "text": normalized})
    return variants


def parse_expected_schema(raw_schema: str) -> tuple[set[str], set[str]]:
    schemas: set[str] = set()
    subtypes: set[str] = set()
    for piece in raw_schema.split("+"):
        schema_part, _, subtype_part = piece.strip().partition("/")
        if schema_part.strip():
            schemas.add(schema_part.strip())
        if subtype_part.strip():
            subtypes.add(subtype_part.strip())
    return schemas, subtypes


def dedup_doc_ids(results: list[dict]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in results:
        doc_id = item.get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def topk_jaccard(left: list[str], right: list[str], k: int) -> float:
    set_left = set(left[:k])
    set_right = set(right[:k])
    union = set_left | set_right
    if not union:
        return 1.0
    return len(set_left & set_right) / len(union)


def run_variant(
    scenario: dict, variant: dict[str, str], top_k: int, table: str
) -> dict:
    israg_report = run_search(
        variant["text"], scenario["mode"], top_k, table, baseline=False
    )
    baseline_report = run_search(
        variant["text"], scenario["mode"], top_k, table, baseline=True
    )

    israg_ids = dedup_doc_ids(israg_report.get("results", []))
    baseline_ids = dedup_doc_ids(baseline_report.get("results", []))
    rel_map = scenario["relevance_map"]

    return {
        "scenario_id": scenario["id"],
        "variant_name": variant["name"],
        "variant_text": variant["text"],
        "mode": scenario["mode"],
        "detected_schemas": israg_report.get("detected_schemas", []),
        "detected_sub_types": israg_report.get("detected_sub_types", []),
        "israg_doc_ids": israg_ids,
        "baseline_doc_ids": baseline_ids,
        "ndcg_israg": compute_ndcg(israg_ids, rel_map, top_k),
        "ndcg_baseline": compute_ndcg(baseline_ids, rel_map, top_k),
        "top1_israg": israg_ids[0] if israg_ids else None,
        "top1_baseline": baseline_ids[0] if baseline_ids else None,
    }


def summarize_scenario(scenario: dict, records: list[dict], top_k: int) -> dict:
    ordered = sorted(records, key=lambda item: item["variant_name"] != "original")
    original = next(
        (item for item in ordered if item["variant_name"] == "original"), None
    )
    if original is None:
        raise ValueError(f"Scenario Q{scenario['id']} sem variante original.")

    perturbed = [item for item in ordered if item["variant_name"] != "original"]
    expected_schemas, expected_subtypes = parse_expected_schema(scenario["schema"])

    def _has_expected_schema(item: dict) -> bool:
        detected = set(item.get("detected_schemas", []))
        return expected_schemas.issubset(detected)

    def _has_expected_subtype(item: dict) -> bool:
        if not expected_subtypes:
            return True
        detected = set(item.get("detected_sub_types", []))
        return expected_subtypes.issubset(detected)

    variant_count = len(perturbed)
    comparison_set = perturbed if perturbed else [original]

    israg_drops = [
        item["ndcg_israg"] - original["ndcg_israg"] for item in comparison_set
    ]
    baseline_drops = [
        item["ndcg_baseline"] - original["ndcg_baseline"] for item in comparison_set
    ]
    top1_consistency = sum(
        item["top1_israg"] == original["top1_israg"] for item in comparison_set
    ) / len(comparison_set)
    mean_jaccard = statistics.fmean(
        topk_jaccard(item["israg_doc_ids"], original["israg_doc_ids"], top_k)
        for item in comparison_set
    )
    schema_hit_rate = sum(_has_expected_schema(item) for item in comparison_set) / len(
        comparison_set
    )
    subtype_hit_rate = sum(
        _has_expected_subtype(item) for item in comparison_set
    ) / len(comparison_set)
    win_rate = sum(
        item["ndcg_israg"] >= item["ndcg_baseline"] for item in comparison_set
    ) / len(comparison_set)

    return {
        "id": scenario["id"],
        "name": scenario["name"],
        "mode": scenario["mode"],
        "schema": scenario["schema"],
        "variant_count": variant_count,
        "original_ndcg_israg": original["ndcg_israg"],
        "original_ndcg_baseline": original["ndcg_baseline"],
        "mean_ndcg_israg": statistics.fmean(
            item["ndcg_israg"] for item in comparison_set
        ),
        "mean_ndcg_baseline": statistics.fmean(
            item["ndcg_baseline"] for item in comparison_set
        ),
        "mean_drop_israg": statistics.fmean(israg_drops),
        "mean_drop_baseline": statistics.fmean(baseline_drops),
        "worst_drop_israg": min(israg_drops),
        "worst_drop_baseline": min(baseline_drops),
        "schema_hit_rate": schema_hit_rate,
        "subtype_hit_rate": subtype_hit_rate,
        "top1_consistency": top1_consistency,
        "mean_topk_jaccard": mean_jaccard,
        "win_rate_vs_baseline": win_rate,
        "variants": ordered,
    }


def print_summary(results: list[dict], top_k: int):
    print("=" * 116)
    print(
        f"{'Q':<4} {'Schema':<24} {'Orig Δ':>9} {'Drop IS':>9} {'Drop Base':>11} "
        f"{'Schema%':>9} {'Subtype%':>10} {'Top1%':>8} {'Jaccard':>9} {'Win%':>7}"
    )
    print("=" * 116)
    for row in results:
        original_delta = row["original_ndcg_israg"] - row["original_ndcg_baseline"]
        print(
            f"Q{row['id']:<3} {row['schema'][:24]:<24} {original_delta:>+9.4f} "
            f"{row['worst_drop_israg']:>+9.4f} {row['worst_drop_baseline']:>+11.4f} "
            f"{row['schema_hit_rate'] * 100:>8.1f}% {row['subtype_hit_rate'] * 100:>9.1f}% "
            f"{row['top1_consistency'] * 100:>7.1f}% {row['mean_topk_jaccard']:>9.4f} "
            f"{row['win_rate_vs_baseline'] * 100:>6.1f}%"
        )

    print("─" * 116)
    overall = {
        "mean_original_delta": statistics.fmean(
            row["original_ndcg_israg"] - row["original_ndcg_baseline"]
            for row in results
        ),
        "mean_worst_drop_israg": statistics.fmean(
            row["worst_drop_israg"] for row in results
        ),
        "mean_worst_drop_baseline": statistics.fmean(
            row["worst_drop_baseline"] for row in results
        ),
        "schema_hit_rate": statistics.fmean(row["schema_hit_rate"] for row in results),
        "subtype_hit_rate": statistics.fmean(
            row["subtype_hit_rate"] for row in results
        ),
        "top1_consistency": statistics.fmean(
            row["top1_consistency"] for row in results
        ),
        "mean_topk_jaccard": statistics.fmean(
            row["mean_topk_jaccard"] for row in results
        ),
        "win_rate": statistics.fmean(row["win_rate_vs_baseline"] for row in results),
    }
    print(
        f"{'Média':<29} {overall['mean_original_delta']:>+9.4f} "
        f"{overall['mean_worst_drop_israg']:>+9.4f} {overall['mean_worst_drop_baseline']:>+11.4f} "
        f"{overall['schema_hit_rate'] * 100:>8.1f}% {overall['subtype_hit_rate'] * 100:>9.1f}% "
        f"{overall['top1_consistency'] * 100:>7.1f}% {overall['mean_topk_jaccard']:>9.4f} "
        f"{overall['win_rate'] * 100:>6.1f}%"
    )
    print("=" * 116)
    print(
        f"\nRobustez @ NDCG@{top_k}: quanto menos negativo o 'Drop IS', mais estável o IS-RAG."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliação de robustez do IS-RAG por perturbação de queries."
    )
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GT),
        help="Caminho para ground_truth_real.json",
    )
    parser.add_argument(
        "--table", default=DEFAULT_TABLE, help="Tabela PostgreSQL a usar"
    )
    parser.add_argument("--top-k", type=int, default=5, help="NDCG@K (default 5)")
    parser.add_argument(
        "--workers", type=int, default=4, help="Workers concorrentes (default 4)"
    )
    parser.add_argument(
        "--force-mode",
        default=None,
        help="Força um modo para todas as queries (texto|cognitivo|hibrido)",
    )
    parser.add_argument(
        "--query-id", type=int, default=None, help="Avalia apenas uma query específica"
    )
    parser.add_argument(
        "--json-out", default=None, help="Salva relatório detalhado em JSON"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"[X] Ground truth não encontrado: {gt_path}", file=sys.stderr)
        return 1

    print(f"[*] Carregando ground truth: {gt_path}")
    scenarios = load_ground_truth(gt_path)
    if args.query_id is not None:
        scenarios = [
            scenario for scenario in scenarios if scenario["id"] == args.query_id
        ]

    if not scenarios:
        print("[X] Nenhuma query válida para avaliar.", file=sys.stderr)
        return 1

    if args.force_mode:
        for scenario in scenarios:
            scenario["mode"] = args.force_mode
        print(
            f"[!] Modo forçado: {args.force_mode} para todas as queries selecionadas.\n"
        )

    tasks = []
    for scenario in scenarios:
        variants = build_variants(scenario["text"])
        for variant in variants:
            tasks.append((scenario, variant))

    print(
        f"[*] Robustez em {len(scenarios)} queries | {len(tasks)} execuções IS-RAG+baseline "
        f"| tabela={args.table} | top-k={args.top_k}\n"
    )

    raw_results: dict[int, list[dict]] = {scenario["id"]: [] for scenario in scenarios}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(run_variant, scenario, variant, args.top_k, args.table): (
                scenario["id"],
                variant["name"],
            )
            for scenario, variant in tasks
        }

        done = 0
        total = len(future_map)
        for future in as_completed(future_map):
            scenario_id, variant_name = future_map[future]
            try:
                record = future.result()
                raw_results[scenario_id].append(record)
                done += 1
                print(f"  [{done:>2}/{total}] Q{scenario_id:>2} {variant_name:<18} ok")
            except Exception as exc:
                print(
                    f"  [{done:>2}/{total}] Q{scenario_id:>2} {variant_name:<18} ERRO: {exc}"
                )

    print()
    summarized = []
    for scenario in scenarios:
        scenario_records = raw_results.get(scenario["id"], [])
        if not scenario_records:
            print(f"[X] Q{scenario['id']} sem resultados.", file=sys.stderr)
            return 1
        summarized.append(summarize_scenario(scenario, scenario_records, args.top_k))

    summarized.sort(key=lambda item: item["id"])
    print_summary(summarized, args.top_k)

    if args.json_out:
        output_path = Path(args.json_out).expanduser()
        output_path.write_text(
            json.dumps({"results": summarized}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[√] Relatório salvo em: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
