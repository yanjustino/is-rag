#!/usr/bin/env python3
"""
Coleta discursos de múltiplos partidos para diversificação do corpus IS-RAG.
Executa o coletor sequencialmente por partido, deduplica por id_interno e
salva tudo em corpus_camara_piloto.jsonl.

Uso:
  python collect_corpus.py                  # coleta completa (~240 deputados)
  python collect_corpus.py --dry-run        # mostra o plano sem coletar
  python collect_corpus.py --max 5          # 5 deputados por partido (teste rápido)
  python collect_corpus.py --output outro.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from src.coletor import CamaraDataCollector

# Espectro ideológico cobrindo esquerda, centro e direita.
# max_dep: deputados solicitados por partido (a API retorna em ordem alfabética).
PARTIDOS = [
    # Esquerda
    {"sigla": "PT", "max_dep": 20, "espectro": "esquerda"},
    {"sigla": "PSOL", "max_dep": 10, "espectro": "esquerda"},
    {"sigla": "REDE", "max_dep": 10, "espectro": "esquerda"},
    {"sigla": "PSB", "max_dep": 10, "espectro": "esquerda"},
    # Centro
    {"sigla": "MDB", "max_dep": 15, "espectro": "centro"},
    {"sigla": "UNIÃO", "max_dep": 15, "espectro": "centro"},
    {"sigla": "PSD", "max_dep": 15, "espectro": "centro"},
    {"sigla": "SOLIDARIEDADE", "max_dep": 10, "espectro": "centro"},
    # Direita
    {"sigla": "PL", "max_dep": 20, "espectro": "direita"},
    {"sigla": "NOVO", "max_dep": 10, "espectro": "direita"},
    {"sigla": "REPUBLICANOS", "max_dep": 10, "espectro": "direita"},
    {"sigla": "PP", "max_dep": 10, "espectro": "direita"},
    {"sigla": "AVANTE", "max_dep": 10, "espectro": "direita"},
]

DATA_INICIO = "2025-01-01"
DATA_FIM = "2025-12-31"


def load_existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id_interno"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def collect_all(output_path: Path, max_per_party: int | None, dry_run: bool):
    collector = CamaraDataCollector()
    seen_ids = load_existing_ids(output_path)

    total_new = 0
    total_skipped = 0

    print(f"\n{'DRY-RUN — ' if dry_run else ''}Corpus alvo: {output_path}")
    print(f"IDs já existentes: {len(seen_ids)}\n")
    print(f"{'Partido':<16} {'Espectro':<10} {'Max dep':>7}")
    print("─" * 36)
    for p in PARTIDOS:
        limit = max_per_party if max_per_party else p["max_dep"]
        print(f"{p['sigla']:<16} {p['espectro']:<10} {limit:>7}")
    print()

    if dry_run:
        total_dep = sum(max_per_party or p["max_dep"] for p in PARTIDOS)
        print(f"Total de deputados a coletar: ~{total_dep}")
        print("Rode sem --dry-run para executar.")
        return

    with output_path.open("a", encoding="utf-8") as out_file:
        for p in PARTIDOS:
            limit = max_per_party if max_per_party else p["max_dep"]
            print(f"\n{'=' * 60}")
            print(f"Partido: {p['sigla']} ({p['espectro']}) — {limit} deputados")
            print(f"{'=' * 60}")

            discursos = collector.coletar_discursos(
                data_inicio=DATA_INICIO,
                data_fim=DATA_FIM,
                max_deputados=limit,
                partido=p["sigla"],
            )

            novos = 0
            repetidos = 0
            for d in discursos:
                iid = d["id_interno"]
                if iid in seen_ids:
                    repetidos += 1
                    continue
                seen_ids.add(iid)
                out_file.write(json.dumps(d, ensure_ascii=False) + "\n")
                novos += 1

            out_file.flush()
            total_new += novos
            total_skipped += repetidos
            print(f"  → {novos} novos salvos, {repetidos} duplicatas ignoradas")

    print(f"\n{'=' * 60}")
    print(f"Coleta concluída.")
    print(f"  Novos discursos adicionados : {total_new}")
    print(f"  Duplicatas ignoradas        : {total_skipped}")
    print(f"  Total no corpus             : {len(seen_ids)}")
    print(f"  Arquivo                     : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Coleta diversificada do corpus IS-RAG."
    )
    parser.add_argument(
        "--output",
        default="data/corpus_camara_piloto.jsonl",
        help="Arquivo de saída (default: data/corpus_camara_piloto.jsonl)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        dest="max_per_party",
        help="Limite de deputados por partido (sobrepõe o padrão por partido)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Mostra o plano sem coletar"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path(__file__).parent / output_path).resolve()

    collect_all(output_path, args.max_per_party, args.dry_run)


if __name__ == "__main__":
    main()
