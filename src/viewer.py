#!/usr/bin/env python3
import json
import os
import sys
import psycopg
from psycopg import sql

DB_CONN = "host=localhost port=5433 dbname=is_rag_db user=is_rag_user password=is_rag_password"
DB_TABLE = os.getenv("IS_RAG_TABLE", "document_chunks")

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
GRAY   = "\033[90m"
MAGENTA= "\033[95m"

def print_header():
    print(f"\n{BOLD}{CYAN}{'═' * 80}")
    print("  IS-RAG · Visualizador de Chunks")
    print(f"{'═' * 80}{RESET}\n")

def collect_stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(DB_TABLE)))
        total = cur.fetchone()[0]

        cur.execute(sql.SQL("""
            SELECT source_metadata->>'orador_partido' AS orador_partido, COUNT(*) as n
            FROM {}
            GROUP BY source_metadata->>'orador_partido'
            ORDER BY n DESC
        """).format(sql.Identifier(DB_TABLE)))
        partidos = cur.fetchall()

        cur.execute(sql.SQL("""
            SELECT jsonb_array_elements_text(cognitive_metadata->'schemas') AS schema, COUNT(*) AS n
            FROM {}
            GROUP BY schema
            ORDER BY n DESC
        """).format(sql.Identifier(DB_TABLE)))
        schemas = cur.fetchall()

    return {
        "total": total,
        "by_partido": {p: n for p, n in partidos},
        "by_schema": {s: n for s, n in schemas},
    }

def print_stats(conn):
    stats = collect_stats(conn)
    partidos_str = "  ".join(f"{p}({n})" for p, n in stats["by_partido"].items())
    schemas_str  = "  ".join(f"{MAGENTA}{s}{RESET}({n})" for s, n in stats["by_schema"].items())
    print(f"{BOLD}Resumo da base{RESET}")
    print(f"  Total de chunks   : {YELLOW}{stats['total']}{RESET}")
    print(f"  Por partido       : {partidos_str}")
    print(f"  Schemas cognitivos: {schemas_str}")
    print()

def print_chunk(rank, row):
    cid, source_meta, content, metadata = row
    orador   = source_meta.get("orador_nome")
    partido  = source_meta.get("orador_partido")
    uf       = source_meta.get("orador_uf")
    schemas  = metadata.get("schemas", [])
    details  = metadata.get("details", [])

    schema_str = "  ".join(f"{MAGENTA}[{s}]{RESET}" for s in schemas) or f"{GRAY}(nenhum){RESET}"
    print(f"{BOLD}{CYAN}── Chunk #{rank} ─── ID {cid} ──────────────────────────────────{RESET}")
    print(f"  {BOLD}Orador{RESET}  : {orador} ({partido}-{uf})")
    print(f"  {BOLD}Schemas{RESET} : {schema_str}")

    if details:
        print(f"  {BOLD}Detalhes{RESET}:")
        for d in details:
            print(f"    {GREEN}▸{RESET} [{d['schema_name']}/{d['sub_type']}] âncora: '{d['anchor_word_pt']}' → {d['target_domain_pt']}")

    print(f"  {BOLD}Trecho{RESET}:")
    words = content.split()
    for i in range(0, len(words), 14):
        print(f"    {GRAY}{' '.join(words[i:i+14])}{RESET}")
    print()

def fetch_chunks(conn, limit: int, partido: str | None, schema: str | None) -> list:
    conditions = []
    params: list = []

    if partido:
        conditions.append("source_metadata->>'orador_partido' ILIKE %s")
        params.append(f"%{partido}%")
    if schema:
        conditions.append("cognitive_metadata->'schemas' ? %s")
        params.append(schema.upper())

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            SELECT id, source_metadata, content, cognitive_metadata
            FROM {}
            {where_clause}
            ORDER BY id
            LIMIT %s
        """).format(
            sql.Identifier(DB_TABLE),
            where_clause=sql.SQL(where),
        ), params)
        return cur.fetchall()

def list_chunks(conn, limit: int, partido: str | None, schema: str | None):
    rows = fetch_chunks(conn, limit, partido, schema)
    if not rows:
        print(f"{YELLOW}Nenhum chunk encontrado com os filtros aplicados.{RESET}\n")
        return
    for i, row in enumerate(rows, 1):
        print_chunk(i, row)

def list_chunks_json(conn, limit: int, partido: str | None, schema: str | None):
    rows = fetch_chunks(conn, limit, partido, schema)
    chunks = [
        {
            "id": cid,
            "content": content,
            "source_metadata": source_meta,
            "cognitive_metadata": cognitive_meta,
        }
        for cid, source_meta, content, cognitive_meta in rows
    ]
    print(json.dumps(chunks, ensure_ascii=False, indent=2))

def usage():
    print("Uso:")
    print("  python viewer.py                        # lista 10 chunks")
    print("  python viewer.py --limit 20             # lista 20 chunks")
    print("  python viewer.py --partido PT           # filtra por partido")
    print("  python viewer.py --schema FORCE         # filtra por esquema cognitivo")
    print("  python viewer.py --stats                # apenas resumo")
    print("  python viewer.py --json                 # saída JSON com todos os campos")
    print("  python viewer.py --stats --json         # resumo em JSON")
    sys.exit(0)

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--help" in args:
        usage()

    limit      = 10
    partido    = None
    schema     = None
    only_stats = "--stats" in args
    json_output = "--json" in args

    try:
        if "--limit" in args:
            limit = int(args[args.index("--limit") + 1])
        if "--partido" in args:
            partido = args[args.index("--partido") + 1]
        if "--schema" in args:
            schema = args[args.index("--schema") + 1]
    except (IndexError, ValueError):
        usage()

    with psycopg.connect(DB_CONN) as conn:
        if json_output:
            if only_stats:
                print(json.dumps(collect_stats(conn), ensure_ascii=False, indent=2))
            else:
                list_chunks_json(conn, limit, partido, schema)
        else:
            print_header()
            print_stats(conn)
            if not only_stats:
                list_chunks(conn, limit, partido, schema)
