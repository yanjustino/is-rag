#!/usr/bin/env python3
"""
annotate_pool.py — Anotação de relevância (TREC-style pooling).

Para cada query, combina top-10 IS-RAG + top-10 baseline, deduplica por
chunk_id e anota relevância (0–3). Dois modos disponíveis:

  Manual  — você digita o score para cada chunk
  Auto    — LLM (claude-haiku) anota todos os chunks em paralelo

Uso:
  python annotate_pool.py                        # manual, todas as queries
  python annotate_pool.py --auto                 # LLM anota tudo
  python annotate_pool.py --auto --query 3       # LLM anota só a query 3
  python annotate_pool.py --resume               # continuar de onde parou
  python annotate_pool.py --dry-run              # mostrar pools sem anotar
  python annotate_pool.py --auto-workers 8       # concorrência do anotador (padrão 5)
"""

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from search import metaphorical_search

# ---------------------------------------------------------------------------
# Anotador LLM
# ---------------------------------------------------------------------------
_ANNOTATOR_MODEL = os.getenv("ANNOTATOR_MODEL", "claude-opus-4-7")
_annotator_client = anthropic.Anthropic()

_VALID_SUB_TYPES = (
    "INSIDE, OUTSIDE, BOUNDARY, INTRUSION, "
    "SOURCE, TRAJECTORY, GOAL, DIVERSION, "
    "BLOCKAGE, COMPULSION, RESISTANCE, COUNTER_FORCE"
)

_ANNOTATOR_PROMPT = """\
Você é um juiz de relevância para um experimento de avaliação de sistemas de \
busca cognitiva baseados em Esquemas Imagéticos (IS-RAG), fundamentado na teoria \
de Lakoff & Johnson (1980) e Talmy (1988).

Sua tarefa: dado um TRECHO de discurso parlamentar e uma QUERY com schema \
esperado, atribuir uma nota de relevância de 0 a 3.

=== TAXONOMIA DE REFERÊNCIA ===
Analise o texto seguindo rigorosamente esta taxonomia estrita:

1. MACROESQUEMAS E SUBTIPOS VÁLIDOS (Mantenha as chaves em MAIÚSCULAS e em inglês):
- CONTAINER:
  * INSIDE (Estar contido, protegido, preso, engessado)
  * OUTSIDE (Estar fora, excluído, marginalizado)
  * BOUNDARY (Fronteiras, limites, barreiras de contenção)
  * INTRUSION (Invasão, penetração forçada no recipiente)
- PATH:
  * SOURCE (Ponto de partida, origem, base histórica)
  * TRAJECTORY (Movimento, progresso, passos, rumo, avanço)
  * GOAL (Destino, objetivo final, chegada, conclusão)
  * DIVERSION (Desvio de rota, perda de foco, sabotagem do processo)
- FORCE:
  * BLOCKAGE (Bloqueio completo, barreira física/legal, trancar pauta, barrar)
  * COMPULSION (Força externa que empurra, obriga ou coage a agir)
  * RESISTANCE (Resistência interna, oposição ativa ou estancamento de uma força)
  * COUNTER_FORCE (Duas forças colidindo de frente, embate direto, enfrentamento)

2. REGRAS PARA O DOMÍNIO ALVO (target_domain_pt):
O domínio alvo deve ser a categoria macro do assunto abstrato que está sendo estruturado pela metáfora física. Use OBRIGATORIAMENTE um destes termos padronizados:
- "Economia" (Inflação, Imposto de Renda, arcabouço fiscal, juros, tributação)
- "Política" (Disputas partidárias, eleições, cassação de mandato, anistia, obstrução, oposição)
- "Infraestrutura e Transportes" (Rodovias, portos, asfalto, indústria naval, energia, pontes, aeroportos)
- "Segurança Pública" (Crime organizado, facções, milícias, policiamento, penas, armamento)
- "Justiça" (Decisões do STF, processos judiciais, constitucionalidade, cumprimento de leis, foro)
- "Direitos Humanos e Cultura" (Racismo, pautas indígenas/quilombolas, feminicídio, manifestações culturais, minorias, mulheres)
- "Educação" (Universidades, escolas, institutos federais, professores, financiamento, Pé-de-Meia)
- "Saúde" (SUS, hospitais, médicos peritos, climatério, planos de saúde, doenças)
- "Meio Ambiente" (Crise climática, COP 30, desmatamento, transição energética, sustentabilidade)
- "Relações Internacionais" (Diplomacia, comércio exterior, tratados, geopolítica, tarifas alfandegárias, Trump/EUA)
- "Outros" (Casos excepcionais que fujam completamente do escopo político, como pêsames, homenagens fúnebres ou saudações protocolares)

3. REGRA DE LITERALIDADE:
Se o texto for puramente literal, descritivo, técnico, administrativo ou não contiver nenhuma metáfora conceitual baseada nos esquemas acima, retorne as listas de esquemas e detalhes completamente vazias. Não force classificações em textos literais.

=== CRITÉRIOS DE PONTUAÇÃO ===

  0 — IRRELEVANTE
      O trecho não trata do tema da query E não contém nenhum uso do schema
      esperado. Pode ser protocolar, de saudação ou completamente off-topic.

  1 — MARGINAL
      O trecho trata do mesmo tema da query, mas de forma literal ou descritiva.
      O schema imagético esperado NÃO estrutura a argumentação — não há
      metáfora conceitual identificável, apenas discurso factual sobre o tópico.

  2 — RELEVANTE
      O schema imagético esperado está PRESENTE no trecho. É possível apontar
      uma palavra-âncora (anchor_word_pt) e o subtipo correspondente que
      evidenciam o uso da metáfora conceitual.

  3 — ALTAMENTE RELEVANTE
      O schema é CENTRAL na argumentação do trecho (estrutura o raciocínio
      principal, não é uso periférico) E o domínio coincide com o da query.

REGRA CRÍTICA: Para atribuir score ≥ 2 você DEVE identificar a palavra-âncora
e o subtipo. Se não conseguir apontar evidência textual concreta, o score
máximo é 1. Não force schemas em textos literais.

=== EXEMPLOS CALIBRADOS (Schema: FORCE/COMPULSION | Domínio: Economia) ===

— Score 0 —
Trecho: "Parabenizo o Sr. Presidente e agradeço aos nobres colegas pela presença nesta sessão solene."
Raciocínio: Texto protocolar, sem relação com economia nem com qualquer schema de força.
{{"relevance": 0, "anchor_word": null, "sub_type": null, "justification": "Saudação protocolar sem conteúdo temático ou esquemático relevante."}}

— Score 1 —
Trecho: "O índice de inflação acumulado nos últimos doze meses atingiu 4,83%, conforme divulgado pelo IBGE na última semana."
Raciocínio: Trata do tema econômico mas é puramente factual — nenhuma força age sobre ninguém, nenhuma metáfora conceitual estrutura o argumento.
{{"relevance": 1, "anchor_word": null, "sub_type": null, "justification": "Dado factual sobre inflação sem uso de schema imagético de força."}}

— Score 2 —
Trecho: "A crise fiscal obriga os estados a cortarem investimentos em infraestrutura e serviços essenciais à população."
Raciocínio: Âncora "obriga" evidencia FORCE/COMPULSION — uma força externa coage os estados a agir. Schema presente, mas não domina toda a argumentação.
{{"relevance": 2, "anchor_word": "obriga", "sub_type": "COMPULSION", "justification": "Âncora 'obriga' evidencia força coercitiva da crise fiscal sobre os estados."}}

— Score 3 —
Trecho: "Os juros abusivos sufocam e empurram os pequenos empresários para a falência, compelindo-os a demitir seus funcionários e encerrar as atividades."
Raciocínio: Âncoras "sufocam", "empurram" e "compelindo" evidenciam FORCE/COMPULSION de forma central — o schema estrutura todo o argumento, no domínio econômico exato da query.
{{"relevance": 3, "anchor_word": "empurram", "sub_type": "COMPULSION", "justification": "Schema FORCE/COMPULSION é central: juros como força que empurra empresários à falência no domínio Economia."}}

=== INPUT ===

Query    : "{query_text}"
Schema   : {schema}
Domínio  : {domain}

Trecho:
\"\"\"{content}\"\"\"

=== OUTPUT ===

Primeiro, em uma linha, identifique: qual palavra-âncora você encontra no trecho e qual subtipo válido ela evidencia?
Subtipos válidos: {valid_sub_types}

Depois retorne apenas o JSON (sem markdown):
{{"relevance": <0|1|2|3>, "anchor_word": "<palavra-âncora ou null>", "sub_type": "<subtipo válido ou null>", "justification": "<uma frase explicando a nota>"}}
"""


def _llm_score(query: dict, item: dict, max_retries: int = 5) -> tuple[int, str]:
    """Chama o LLM anotador com retry exponencial em caso de rate limit (429)."""
    prompt = _ANNOTATOR_PROMPT.format(
        query_text=query["text"],
        schema=query["schema"],
        domain=query["domain"],
        content=item["content"],
        valid_sub_types=_VALID_SUB_TYPES,
    )
    for attempt in range(max_retries):
        try:
            msg = _annotator_client.messages.create(
                model=_ANNOTATOR_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            output = msg.content[0].text.strip()
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if not match:
                return 0, "parse error", None, None
            data = json.loads(match.group())
            return (
                int(data["relevance"]),
                data.get("justification", ""),
                data.get("anchor_word"),
                data.get("sub_type"),
            )
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                return 0, "rate limit esgotado", None, None
            wait = (2**attempt) + random.uniform(0, 1)
            print(
                f"\n    [429] rate limit — aguardando {wait:.1f}s (tentativa {attempt + 1}/{max_retries})...",
                flush=True,
            )
            time.sleep(wait)
        except Exception as e:
            return 0, f"erro: {e}", None, None
    return 0, "max retries atingido", None, None


def auto_annotate_pool(
    query: dict,
    pool: list[dict],
    already_annotated: dict,
    max_workers: int = 5,
) -> list[dict]:
    """Anota todo o pool em paralelo usando o LLM anotador."""
    targets = [p for p in pool if str(p["chunk_id"]) not in already_annotated]
    if not targets:
        print("  Todos os chunks já foram anotados.")
        return pool

    print(
        f"  Anotando {len(targets)} chunks via LLM ({max_workers} workers)...",
        flush=True,
    )
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_llm_score, query, item): item for item in targets}
        for future in as_completed(future_map):
            item = future_map[future]
            score, justification, anchor_word, sub_type = future.result()
            item["relevance"] = score
            item["justification"] = justification
            item["anchor_word"] = anchor_word
            item["sub_type"] = sub_type
            already_annotated[str(item["chunk_id"])] = score
            done += 1
            evidence = f"  [{anchor_word} / {sub_type}]" if anchor_word else ""
            print(
                f"    [{done:>2}/{len(targets)}] chunk_id={item['chunk_id']}  "
                f"score={score}{evidence}  {justification[:55]}"
            )

    return pool


# ---------------------------------------------------------------------------
# 10 queries do experimento real
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "id": 1,
        "schema": "CONTAINER/INSIDE",
        "domain": "Saúde",
        "mode": "cognitivo",
        "text": (
            "As políticas de saúde pública estão dentro do espaço de obrigação constitucional "
            "do Estado, mantidas no interior das garantias democráticas que o poder público "
            "não pode abandonar."
        ),
    },
    {
        "id": 2,
        "schema": "FORCE/COMPULSION",
        "domain": "Economia",
        "mode": "hibrido",
        "text": (
            "A carga tributária pressiona e empurra a população para uma situação econômica "
            "pior, sufocando famílias e empresas sob o peso dos impostos."
        ),
    },
    {
        "id": 3,
        "schema": "FORCE/COUNTER_FORCE",
        "domain": "Justiça",
        "mode": "hibrido",
        "text": (
            "O Poder Judiciário e os demais poderes estão em embate direto, como forças "
            "opostas que se chocam e se contrapõem mutuamente no campo institucional."
        ),
    },
    {
        "id": 4,
        "schema": "FORCE/RESISTANCE",
        "domain": "Política",
        "mode": "hibrido",
        "text": (
            "A minoria parlamentar resiste ativamente ao avanço do governo, opondo-se "
            "às pressões da maioria sem conseguir detê-la por completo."
        ),
    },
    {
        "id": 5,
        "schema": "CONTAINER/INTRUSION",
        "domain": "Justiça",
        "mode": "cognitivo",
        "text": (
            "O STF invade e ultrapassa os limites do espaço de atuação do Congresso, "
            "penetrando indevidamente nas fronteiras do Poder Legislativo."
        ),
    },
    {
        "id": 6,
        "schema": "CONTAINER/OUTSIDE",
        "domain": "Direitos Humanos e Cultura",
        "mode": "cognitivo",
        "text": (
            "Grupos sociais historicamente marginalizados estão excluídos e fora das "
            "estruturas institucionais que deveriam protegê-los e acolhê-los."
        ),
    },
    {
        "id": 7,
        "schema": "PATH/TRAJECTORY",
        "domain": "Economia",
        "mode": "hibrido",
        "text": (
            "O Brasil avança passo a passo em direção à estabilidade econômica, "
            "percorrendo um caminho de reformas rumo ao crescimento sustentável."
        ),
    },
    {
        "id": 8,
        "schema": "PATH/SOURCE",
        "domain": "Política",
        "mode": "hibrido",
        "text": (
            "As origens históricas e os fundamentos da democracia brasileira servem "
            "de ponto de partida e alicerce para justificar posições políticas do presente."
        ),
    },
    {
        "id": 9,
        "schema": "PATH/DIVERSION",
        "domain": "Política",
        "mode": "hibrido",
        "text": (
            "O governo atual desviou o Brasil do seu caminho correto, afastando o país "
            "do rumo natural de crescimento e distanciando-o dos seus objetivos democráticos."
        ),
    },
    {
        "id": 10,
        "schema": "FORCE/COMPULSION + PATH/DIVERSION",
        "domain": "Relações Internacionais / Economia",
        "mode": "hibrido",
        "text": (
            "Forças econômicas externas — tarifas, pressão internacional e juros elevados "
            "— empurram e desviam o Brasil para fora do seu trajeto natural de desenvolvimento."
        ),
    },
]

OUTPUT_FILE = Path(__file__).parent / "data" / "ground_truth_real.json"
TOP_K = 10

RELEVANCE_GUIDE = """
  0 — Irrelevante: não trata do tema nem usa o schema
  1 — Marginal: trata do tema mas sem metáfora conceitual clara
  2 — Relevante: schema imagético presente no texto
  3 — Altamente relevante: schema central e domínio exato da query
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if OUTPUT_FILE.exists():
        return json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    return {"queries": {}}


def save_state(state: dict):
    OUTPUT_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_pool(query: dict, dry_run: bool = False) -> tuple[list[dict], dict]:
    """Roda IS-RAG e baseline em paralelo, retorna (pool, cognitive_analysis)."""
    print("\n  [IS-RAG + Baseline] buscando em paralelo...", end=" ", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_israg = ex.submit(
            metaphorical_search, query["text"], TOP_K, query["mode"], False, False
        )
        f_base = ex.submit(
            metaphorical_search, query["text"], TOP_K, query["mode"], False, True
        )
        israg_report = f_israg.result()
        base_report = f_base.result()
    print("ok")

    cognitive_analysis = {
        "schemas": israg_report.get("detected_schemas", []),
        "sub_types": israg_report.get("detected_sub_types", []),
        "domains": israg_report.get("detected_domains", []),
        "details": israg_report.get("detected_details", []),
    }

    israg_ids = {r["chunk_id"] for r in israg_report.get("results", [])}
    base_ids = {r["chunk_id"] for r in base_report.get("results", [])}

    seen = set()
    pool = []
    for result in israg_report.get("results", []) + base_report.get("results", []):
        cid = result["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        pool.append(
            {
                "chunk_id": cid,
                "doc_id": result.get("doc_id"),
                "orador_nome": result["source_metadata"].get("orador_nome"),
                "orador_partido": result["source_metadata"].get("orador_partido"),
                "orador_uf": result["source_metadata"].get("orador_uf"),
                "content": result["content"],
                "cognitive_schemas": result["cognitive_metadata"].get("schemas", []),
                "in_israg": cid in israg_ids,
                "in_baseline": cid in base_ids,
                "israg_score": next(
                    (
                        r["final_score"]
                        for r in israg_report.get("results", [])
                        if r["chunk_id"] == cid
                    ),
                    None,
                ),
                "baseline_score": next(
                    (
                        r["final_score"]
                        for r in base_report.get("results", [])
                        if r["chunk_id"] == cid
                    ),
                    None,
                ),
                "relevance": None,
            }
        )

    if dry_run:
        print(
            f"\n  Pool: {len(pool)} chunks únicos "
            f"({len(israg_ids)} IS-RAG, {len(base_ids)} baseline, "
            f"{len(israg_ids & base_ids)} em comum)"
        )
        for item in pool:
            origem = []
            if item["in_israg"]:
                origem.append("IS-RAG")
            if item["in_baseline"]:
                origem.append("Baseline")
            print(
                f"    [{', '.join(origem):18}] {item['orador_nome']} ({item['orador_partido']}-{item['orador_uf']}) "
                f"schemas={item['cognitive_schemas']}"
            )

    return pool, cognitive_analysis


def annotate_pool(query: dict, pool: list[dict], already_annotated: dict) -> list[dict]:
    """Exibe cada chunk do pool e coleta relevância 0-3 interativamente."""
    total = len(pool)
    unannotated = [p for p in pool if str(p["chunk_id"]) not in already_annotated]

    if not unannotated:
        print("  Todos os chunks já foram anotados.")
        return pool

    print(
        f"\n  {len(unannotated)} chunks para anotar ({total - len(unannotated)} já prontos)."
    )
    print(RELEVANCE_GUIDE)

    for i, item in enumerate(unannotated, 1):
        origem = []
        if item["in_israg"]:
            origem.append("IS-RAG")
        if item["in_baseline"]:
            origem.append("Baseline")

        print("\n" + "═" * 80)
        print(f"  Chunk {i}/{len(unannotated)}  │  chunk_id={item['chunk_id']}")
        print(f"  Recuperado por : {', '.join(origem)}")
        print(
            f"  Orador         : {item['orador_nome']} ({item['orador_partido']}-{item['orador_uf']})"
        )
        print(f"  Schemas        : {item['cognitive_schemas'] or '(nenhum detectado)'}")
        print("─" * 80)

        # Exibe o texto em linhas de ~80 chars
        text = item["content"]
        words = text.split()
        line, lines = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 76:
                lines.append("  " + " ".join(line))
                line = []
        if line:
            lines.append("  " + " ".join(line))
        print("\n".join(lines))
        print("─" * 80)

        while True:
            try:
                raw = (
                    input("  Relevância [0/1/2/3] (s=skip, q=salvar e sair): ")
                    .strip()
                    .lower()
                )
            except (EOFError, KeyboardInterrupt):
                print("\n[!] Interrompido — salvando progresso.")
                return pool

            if raw == "q":
                print("[!] Saindo — progresso salvo.")
                return pool
            if raw == "s":
                print("  (pulado)")
                break
            if raw in ("0", "1", "2", "3"):
                item["relevance"] = int(raw)
                already_annotated[str(item["chunk_id"])] = int(raw)
                break
            print("  Entrada inválida. Digite 0, 1, 2, 3, s ou q.")

    return pool


def print_summary(state: dict):
    print("\n" + "=" * 60)
    print("RESUMO DA ANOTAÇÃO")
    print("=" * 60)
    for qid, qdata in state["queries"].items():
        annotated = sum(1 for c in qdata["pool"] if c["relevance"] is not None)
        total = len(qdata["pool"])
        rel = [c["relevance"] for c in qdata["pool"] if c["relevance"] is not None]
        pos = sum(1 for r in rel if r >= 2)
        print(f"  Q{qid:>2} — {annotated}/{total} anotados, {pos} relevantes (≥2)")
    print(f"\nArquivo: {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = sys.argv[1:]
    resume = "--resume" in args
    dry_run = "--dry-run" in args
    auto = "--auto" in args
    only_qid = None
    auto_workers = 5

    if "--query" in args:
        only_qid = int(args[args.index("--query") + 1])
    if "--auto-workers" in args:
        auto_workers = int(args[args.index("--auto-workers") + 1])

    state = load_state() if resume else {"queries": {}}

    queries_to_run = [q for q in QUERIES if only_qid is None or q["id"] == only_qid]

    for query in queries_to_run:
        qid = str(query["id"])
        print(f"\n{'=' * 80}")
        print(
            f"  Query {query['id']:>2}/10  │  {query['schema']}  │  {query['domain']}"
        )
        print(f'  "{query["text"]}"')
        print(f"{'=' * 80}")

        existing = state["queries"].get(qid, {})
        already_annotated = {
            str(c["chunk_id"]): c["relevance"]
            for c in existing.get("pool", [])
            if c["relevance"] is not None
        }

        if existing.get("pool") and not dry_run:
            pool = existing["pool"]
            cognitive_analysis = existing.get("cognitive_analysis")
            print(f"  Pool existente com {len(pool)} chunks.")
        else:
            pool, cognitive_analysis = build_pool(query, dry_run=dry_run)

        if dry_run:
            continue

        if auto:
            pool = auto_annotate_pool(
                query, pool, already_annotated, max_workers=auto_workers
            )
        else:
            pool = annotate_pool(query, pool, already_annotated)

        state["queries"][qid] = {
            "id": query["id"],
            "schema": query["schema"],
            "domain": query["domain"],
            "mode": query["mode"],
            "text": query["text"],
            "cognitive_analysis": cognitive_analysis,
            "pool": pool,
        }
        save_state(state)
        print(f"\n  [√] Query {query['id']} salva.")

    if not dry_run:
        print_summary(state)


if __name__ == "__main__":
    main()
