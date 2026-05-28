# Metodologia — IS-RAG
## Image-Schematic Retrieval-Augmented Generation for Deep Cognitive Search in Political Discourse

> Documento de trabalho. Cobre todo o pipeline implementado, decisões de design e divergências do plano original. Base para a seção *Methodology* do paper.

---

## 1. Corpus

### 1.1 Fonte
Discursos parlamentares coletados via API de Dados Abertos da Câmara dos Deputados (`dados.camara.leg.br/api/v2`). Período: 01/01/2025 a 31/12/2025.

### 1.2 Coleta (`coletor.py`, `collect_corpus.py`)
- Amostragem estratificada por espectro ideológico: esquerda (PT, PSOL, REDE, PSB), centro (MDB, UNIÃO, PSD, SOLIDARIEDADE) e direita (PL, NOVO, REPUBLICANOS, PP, AVANTE).
- Filtro de qualidade: discursos com menos de 300 caracteres descartados para garantir densidade argumentativa mínima.
- Deduplicação por `id_interno` antes de persistir.

### 1.3 Estatísticas do Corpus
| Atributo | Valor |
|---|---|
| Total de discursos | 121 |
| Partidos representados | 13 |
| Palavras por discurso (média) | 403 |
| Palavras por discurso (mediana) | 330 |
| Intervalo | 56 – 2.125 palavras |

Distribuição por partido: PL (20), PT (19), MDB (13), UNIÃO (12), PSD (12), REPUBLICANOS (10), PSOL (9), PSB (6), NOVO (5), PP (5), SOLIDARIEDADE (4), REDE (3), AVANTE (3).

---

## 2. Pré-processamento: Chunking Semântico

### 2.1 Estratégia (`ingestion.py → semantic_chunking()`)
Divisão por sentença (split em `". "`), acumulando sentenças até atingir o limite de ~150 palavras por chunk. Ao atingir o limite, o chunk corrente é fechado e um novo começa.

**Motivação da escolha:** chunks de 150 palavras preservam coerência intra-sentencial e mantêm o schema imagético de um único argumento sem diluí-lo com argumentos adjacentes.

### 2.2 Resultado
| Atributo | Valor |
|---|---|
| Total de chunks gerados | 171 |
| Chunks por discurso (média estimada) | 2,7 |
| Chunks por discurso (mediana estimada) | 2,0 |

---

## 3. Representação Vetorial (Embeddings de Texto)

### 3.1 Modelo
`paraphrase-multilingual-mpnet-base-v2` via `sentence-transformers` (local, sem custo de API).

| Atributo | Valor |
|---|---|
| Dimensão | 768d |
| Suporte multilíngue | sim (50+ idiomas, incluindo português) |
| Execução | CPU local |

**Divergência do plano original:** o plano inicial previa `gemini-embedding-001` (1536d via API). Migrado para modelo local para eliminar dependência de API e custo variável na geração de embeddings.

### 3.2 Dois Embeddings por Chunk
Cada chunk gera **dois vetores independentes**, armazenados em colunas separadas:

1. **`embedding`** — embedding do texto bruto do chunk (768d).
2. **`cognitive_embedding`** — embedding da serialização dos metadados cognitivos extraídos. Formato: `"SCHEMA: ancora → domínio | SCHEMA: ancora → domínio"`. Captura a estrutura imagética isolada do vocabulário literal.

---

## 4. Extração Cognitiva (Cognitive Parser)

### 4.1 Modelo LLM
`claude-haiku-4-5-20251001` via Anthropic Messages API.

**Divergência do plano original:** o plano previa Google Gemini com Structured Outputs. Migrado para Claude via Anthropic API por maior controle do prompt e independência de plataforma. Structured Outputs substituídos por extração de JSON via regex + validação Pydantic.

### 4.2 Taxonomia de Esquemas Imagéticos (Lakoff & Johnson, 1980; Talmy, 1988)
Três macroesquemas, cada um com quatro subtipos válidos:

| Schema | Subtipos |
|---|---|
| **CONTAINER** | INSIDE, OUTSIDE, BOUNDARY, INTRUSION |
| **PATH** | SOURCE, TRAJECTORY, GOAL, DIVERSION |
| **FORCE** | BLOCKAGE, COMPULSION, RESISTANCE, COUNTER_FORCE |

### 4.3 Prompt Engineering (`prompts/cognitive_analysis.md`)
- Instrução de papel: linguista cognitivo especialista em Lakoff e Talmy.
- Taxonomia estrita com descrições expandidas por subtipo.
- Lista fechada de domínios alvo (`target_domain_pt`): Economia, Política, Infraestrutura e Transportes, Segurança Pública, Justiça, Direitos Humanos e Cultura, Educação, Saúde, Meio Ambiente, Relações Internacionais, Outros.
- **Regra de Literalidade:** se o texto for puramente descritivo/protocolar, retornar `{"schemas": [], "details": []}`. Evita alucinação de schemas inexistentes.
- Três exemplos few-shot: dois com schemas (FORCE+PATH; FORCE) e um negativo (saudação protocolar → vazio).

### 4.4 Saída Estruturada (Pydantic)
```
CognitiveAnalysis
  schemas: List[str]          # ex: ["FORCE", "PATH"]
  details: List[ImageSchemaDetail]
    schema_name: str          # CONTAINER | PATH | FORCE
    sub_type: str             # ex: BLOCKAGE
    anchor_word_pt: str       # palavra/expressão que disparou o schema
    target_domain_pt: str     # domínio padronizado
```

### 4.5 Processamento Concorrente
Para reduzir o tempo de ingestão, cada chunk é enviado à API em uma chamada independente. As chamadas para todos os chunks de um discurso são disparadas em paralelo via `ThreadPoolExecutor(max_workers=5)` (configurável por `COGNITIVE_MAX_WORKERS`). A ordem original é preservada por indexação dos futures.

**Motivação:** a abordagem anterior enviava até 10 chunks por chamada (batch). O batch apresentava riscos de desalinhamento de índices (o LLM poderia omitir ou fundir itens literais), truncamento por limite de tokens de saída e impossibilidade de reprocessar chunks individuais com erro.

### 4.6 Distribuição de Schemas no Corpus (171 chunks)
| Schema | Ocorrências |
|---|---|
| PATH | 92 |
| CONTAINER | 88 |
| FORCE | 83 |
| Sem schema (literal) | — |

---

## 5. Banco de Dados

### 5.1 Stack
PostgreSQL 16 + extensão `pgvector`, executado via Docker (`docker-compose.yml`).

### 5.2 Schema da Tabela Principal (`document_chunks`)
```sql
CREATE TABLE document_chunks (
    id                  SERIAL PRIMARY KEY,
    content             TEXT NOT NULL,
    embedding           VECTOR(768),          -- texto bruto
    cognitive_embedding VECTOR(768),          -- serialização cognitiva
    cognitive_metadata  JSONB,                -- schemas, details
    source_metadata     JSONB,                -- orador, partido, UF, chunk_index
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Divergência do plano original:** o schema inicial tinha colunas relacionais (`id_interno`, `orador_nome`, etc.). Refatorado para arquitetura de 3 camadas com `source_metadata JSONB` e `cognitive_embedding` separado.

### 5.3 Índices
```sql
-- Busca vetorial por similaridade de cosseno (embedding de texto)
CREATE INDEX idx_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Busca vetorial no embedding cognitivo
CREATE INDEX idx_chunks_cognitive_emb
    ON document_chunks USING hnsw (cognitive_embedding vector_cosine_ops);

-- Filtro e busca em metadados cognitivos
CREATE INDEX idx_chunks_cognitive_meta
    ON document_chunks USING gin (cognitive_metadata);
```

---

## 6. Busca Híbrida com Cognitive Boosting

### 6.1 Análise da Query (`search.py → analyze_query()`)
Antes de buscar, a query do usuário é submetida ao mesmo prompt cognitivo (`cognitive_analysis.md`) via Anthropic API. O resultado extrai:
- `detected_schemas`: lista dos macroesquemas presentes na query.
- `detected_sub_types`: subtipos detectados.
- `detected_domains`: domínios alvo detectados.

**Detecção determinística (avaliação):** a análise cognitiva é pré-computada durante o pooling de anotação (`annotate_pool.py`) e persistida em `ground_truth_real.json` (`cognitive_analysis`). Em avaliações com `eval_real.py`, essa análise é passada deterministicamente ao `search.py` via `--precomputed-analysis`, eliminando chamadas LLM repetidas para a mesma query e a variância run-to-run associada. Queries sem `cognitive_analysis` armazenado recaem no comportamento padrão (chamada ao LLM).

### 6.2 Três Modos de Busca

| Modo | Vetor usado | Quando usar |
|---|---|---|
| `texto` | `embedding` (texto bruto) | query com vocabulário próximo ao corpus |
| `cognitivo` | `cognitive_embedding` | query com gap lexical alto; busca por estrutura imagética |
| `híbrido` | média(text_sim, cog_sim) | caso geral; balanceia os dois sinais |

### 6.3 Fórmula de Cognitive Boosting (Boosting Contínuo Proporcional)

```sql
schema_proportion = COUNT(details WHERE schema_name IN query_schemas)
                    / NULLIF(total_details, 0)

domain_proportion = COUNT(details WHERE target_domain_pt IN query_domains)
                    / NULLIF(total_details, 0)

final_score = vector_similarity
              * (1.0 + 0.4 * schema_proportion
                     + 0.3 * COALESCE(domain_proportion, 0))
```

**Motivação do design contínuo:** a versão anterior usava um bônus fixo de +0.2 quando qualquer schema coincidisse. O boosting proporcional recompensa documentos onde a correspondência cognitiva é ampla (todos os detalhes coincidem → bônus máximo de +0.7) versus parcial (um detalhe de cinco → bônus pequeno). Elimina o salto discreto de score que distorcia rankings.

**Pesos escolhidos:** `schema_proportion` recebe peso 0.4 (estrutura cognitiva é o sinal primário) e `domain_proportion` recebe 0.3 (domínio reforça mas não é determinante — o objetivo do IS-RAG é justamente recuperar documentos *cross-domain* com o mesmo schema).

### 6.4 Baseline (RAG Tradicional)
Flag `--baseline` em `search.py` pula a análise cognitiva da query e executa busca vetorial pura sobre `embedding` (texto bruto), sem nenhum boosting. Serve como linha de base para comparação quantitativa.

---

## 7. Avaliação

### 7.1 Métrica Principal: NDCG@K
*Normalized Discounted Cumulative Gain* com K=5. Escolhida por:
- Suportar relevância graduada (0–3), ao contrário de métricas binárias.
- Penalizar posicionamento ruim de documentos relevantes no ranking.
- Ser a métrica padrão de avaliação em TREC e sistemas de recuperação modernos.

**Métrica descartada — MRR:** o *Mean Reciprocal Rank* foi considerado e explicitamente rejeitado. Em cenários com múltiplos documentos relevantes com graus distintos, o MRR satura em 1.0 independentemente do ranking, tornando-o incapaz de distinguir IS-RAG do baseline.

### 7.2 Ground Truth — Corpus de Avaliação

O ground truth é construído sobre o corpus real (`corpus_camara_piloto.jsonl`, 171 chunks de 121 discursos parlamentares). Foram projetadas 10 queries manualmente para criar lexical gaps reais entre a linguagem do usuário e a linguagem dos documentos. O design seguiu três princípios:

1. **Cobertura taxonômica:** o conjunto alvo era 3 queries por macroesquema (FORCE, CONTAINER, PATH) mais uma query cross-schema (3-3-3-1). A distribuição inicial tinha 4 FORCE, 2 CONTAINER e 3 PATH; Q1 foi substituída de FORCE/BLOCKAGE para rebalancear. Após iteração de design (detalhada abaixo), o conjunto final válido resultou em 3-2-3-1 (9 queries).
2. **Lexical gap deliberado:** as queries evitam o vocabulário literal esperado nos documentos — usam linguagem metafórica assertiva em vez de termos políticos diretos. Isso força o sistema a recuperar via estrutura imagética, não por sobreposição lexical.
3. **Cenários cross-domain:** queries cujo schema esperado aparece em domínios diferentes do enunciado (ex: query sobre *"forças econômicas externas"* esperando recuperar discursos sobre política que partilham o schema FORCE).

**Formulação assertiva das queries:** as queries foram redigidas como afirmações metafóricas (ex: *"A carga tributária pressiona e empurra a população para uma situação econômica pior"*) em vez de perguntas analíticas. Essa escolha é metodologicamente necessária: o parser cognitivo (`cognitive_analysis.md`) aplica a *regra de literalidade* — textos não-metafóricos retornam `schemas: []`. Perguntas do tipo *"Quais deputados relatam que..."* são classificadas corretamente como literais, anulando a análise cognitiva da query e zerando o boosting.

As 10 queries foram inspecionadas manualmente contra uma amostra de discursos reais do corpus para verificar plausibilidade de matches antes da anotação formal.

Estratégia de anotação TREC-style pooling via `annotate_pool.py`:
1. Cada query é executada em modo IS-RAG (híbrido) e baseline (vetorial puro) em paralelo.
2. Os top-10 de cada sistema são unidos (~14–18 chunks únicos por query).
3. Cada chunk único é anotado com relevância 0–3 pelo LLM anotador (`claude-opus-4-7`) com retry exponencial em caso de rate limit. O modelo do anotador é configurável via variável de ambiente `ANNOTATOR_MODEL` (padrão: `claude-opus-4-7`). A escolha deliberada de um modelo distinto do parser cognitivo (`claude-haiku-4-5`) rompe a circularidade do ground truth: o anotador é independente do indexador, reduzindo o risco de eco (ver §7.5). O prompt do anotador inclui: taxonomia completa de referência; critérios de pontuação 0–3; **quatro exemplos calibrados** cobrindo todos os níveis de score (0/1/2/3), todos no schema FORCE/COMPULSION em contexto parlamentar; instrução de **chain-of-thought** (identificar palavra-âncora antes de atribuir score); e lista fechada de subtipos válidos para o campo `sub_type` da saída JSON.
4. Para scores ≥ 2, o anotador é obrigado a citar a palavra-âncora (`anchor_word_pt`) e o subtipo (`sub_type`) que justificam o score — evidência rastreável e verificável por humano.
5. A análise cognitiva da query (`cognitive_analysis`) é salva junto ao pool em `ground_truth_real.json` para uso determinístico em avaliações futuras (ver §6.1).
6. Resultados salvos em `ground_truth_real.json` para posterior cálculo de NDCG@K.

**Critério de inclusão na avaliação:** queries com zero documentos relevantes (score ≥ 2) no pool são excluídas do cálculo de NDCG, pois o IDCG seria 0 e a métrica ficaria indefinida (Buckley & Voorhees, 2004).

| Query | Schema / Domínio | Status | Chunks anotados | Relevantes (≥ 2) |
|---|---|---|---|---|
| Q1 | CONTAINER/INSIDE / Saúde | **excluída (IDCG=0)** | 18 | 0 |
| Q2 | FORCE/COMPULSION / Economia | válida | 15 | 2 |
| Q3 | FORCE/COUNTER\_FORCE / Justiça | válida | 16 | 5 |
| Q4 | FORCE/RESISTANCE / Política | válida | 15 | 5 |
| Q5 | CONTAINER/INTRUSION / Justiça | válida | 16 | 5 |
| Q6 | CONTAINER/OUTSIDE / Direitos Humanos | válida | 18 | 4 |
| Q7 | PATH/TRAJECTORY / Economia | válida | 15 | 1 |
| Q8 | PATH/SOURCE / Política | válida | 18 | 4 |
| Q9 | PATH/DIVERSION / Política | válida | 17 | 2 |
| Q10 | FORCE+PATH/DIVERSION / Rel. Internacional | válida | 14 | 1 |

**Iteração de design sobre Q1 e achado metodológico:** Q1 passou por três formulações. A versão original (FORCE/BLOCKAGE) foi substituída para corrigir a sobre-representação de FORCE. CONTAINER/BOUNDARY (13 ocorrências no corpus) e CONTAINER/INSIDE (80 ocorrências) foram tentadas em seguida — ambas retornaram IDCG=0. O padrão é metodologicamente relevante: CONTAINER/INSIDE, apesar de ser o subtipo mais frequente no corpus, usa metáforas espaciais altamente convencionalizadas no discurso parlamentar ("dentro do programa", "no interior das garantias"). O anotador classifica esses usos consistentemente como score 1 (domínio presente, schema não-saliente) em vez de ≥ 2 (instanciação metafórica clara). **Frequência de um schema no corpus não implica anotabilidade:** subtipos convencionalizados resistem à discriminação por relevância graduada. Q1 foi excluída pelo critério de inclusão (IDCG=0), resultando em conjunto final de **9 queries válidas** com distribuição **3 FORCE + 2 CONTAINER + 3 PATH + 1 Cross (3-2-3-1)**.

Q7 e Q10, com apenas 1 documento relevante, são os cenários mais exigentes: NDCG@5 = 1.0 apenas se o único documento relevante aparecer na primeira posição.

---

### 7.3 Resultados Quantitativos (NDCG@5)

> **Nota de versão.** Esta seção documenta duas fases de implementação. A **Fase 1** (§7.3.1) contém os resultados obtidos com a implementação inicial, que apresentava dois bugs encadeados. A **Fase 2** (§7.3.2–7.3.5) apresenta os resultados da implementação corrigida — números definitivos.

#### 7.3.1 Implementação Inicial — Fase 1 (resultados supersedidos por bugs)

> ⚠️ Resultados obtidos com dois bugs silenciosos: (1) `.env` no diretório raiz em vez de `src/`, fazendo o subprocess `search.py` não encontrar `ANTHROPIC_API_KEY` — o parser cognitivo falhava silenciosamente e retornava `details: []`; (2) `cognitive_text_from_query` construía strings do tipo `"FORCE: {full_query_text}"` em vez do formato compacto da ingestão `"SCHEMA: ancora → domínio"`. Com o bug ativo, o boosting era nulo (detalhes vazios → `schema_proportion = 0`) e todos os modos colapsavam para busca vetorial pura sobre o embedding usado por cada modo.

**Modo híbrido — 9 queries válidas (Fase 1, com bugs):**

| Cenário | IS-RAG | Baseline | Δ |
|---|---|---|---|
| Q2 FORCE/COMPULSION / Economia | 0.8375 | 0.5308 | **+0.3067** |
| Q3 FORCE/COUNTER\_FORCE / Justiça | 0.9093 | 0.6590 | **+0.2503** |
| Q4 FORCE/RESISTANCE / Política | 0.6548 | 0.5818 | +0.0730 |
| Q5 CONTAINER/INTRUSION / Justiça | 0.6403 | 0.9105 | −0.2702 |
| Q6 CONTAINER/OUTSIDE / Direitos Humanos | 0.7833 | 0.7226 | +0.0607 |
| Q7 PATH/TRAJECTORY / Economia | 0.6488 | 0.6488 | 0.0000 |
| Q8 PATH/SOURCE / Política | 0.5745 | 0.6972 | −0.1227 |
| Q9 PATH/DIVERSION / Política | 0.7777 | 0.8527 | −0.0750 |
| Q10 FORCE+PATH / Rel. Internacional | 0.5397 | 0.6488 | −0.1091 |
| **Média (9 queries)** | **0.7073** | **0.6947** | **+0.0126** |

O Δ=+0.0126 observado na Fase 1 não é consequência do boosting cognitivo funcionando corretamente, mas de uma propriedade acidental do modo híbrido: a média de `embedding` (768d) e `cognitive_embedding` (768d computado sobre texto da query com prefixo de schema) gerou uma distribuição de similaridade ligeiramente diferente da busca textual pura, suficiente para reordenar alguns documentos. Os modos texto e cognitivo retornavam Δ=0 exatamente porque, sem boosting real, reduziam-se à mesma busca vetorial pura que o baseline.

**Modos texto e cognitivo — Fase 1 (todos Δ=0, omitidos por redundância).**

---

#### 7.3.2 Modo Texto — Fase 2 (implementação corrigida)

Correções aplicadas: `.env` movido para `src/`; `cognitive_text_from_query` reescrita para serializar `details` no formato `"SCHEMA: ancora → domínio | SCHEMA: ancora → domínio"`, idêntico ao formato de ingestão.

Execução: `python eval_real.py --force-mode texto`

| Cenário | IS-RAG texto | Baseline | Δ |
|---|---|---|---|
| Q2 FORCE/COMPULSION / Economia | 0.7912 | 0.5308 | **+0.2604** |
| Q3 FORCE/COUNTER\_FORCE / Justiça | 0.8775 | 0.6590 | **+0.2185** |
| Q4 FORCE/RESISTANCE / Política | 0.8082 | 0.5818 | **+0.2264** |
| Q5 CONTAINER/INTRUSION / Justiça | 0.8643 | 0.9105 | −0.0462 |
| Q6 CONTAINER/OUTSIDE / Direitos Humanos | 0.7010 | 0.7226 | −0.0216 |
| Q7 PATH/TRAJECTORY / Economia | 0.8734 | 0.6488 | **+0.2246** |
| Q8 PATH/SOURCE / Política | 0.6837 | 0.6972 | −0.0135 |
| Q9 PATH/DIVERSION / Política | 0.9467 | 0.8527 | **+0.0940** |
| Q10 FORCE+PATH / Rel. Internacional | 0.5397 | 0.6488 | −0.1091 |
| **Média (9 queries)** | **0.7873** | **0.6947** | **+0.0926** |

**Achado principal: IS-RAG texto supera o baseline em +0.0926 NDCG@5 médio** (run 1) e +0.0769 (run 2) — ganho 7,3× e 5,5× maior, respectivamente, que o observado na Fase 1 com modo híbrido (+0.0126). Com o boosting cognitivo funcionando corretamente, o fator multiplicativo `(1 + 0.4σ + 0.3δ)` é suficientemente grande para reordenar o top-5 quando há correspondência cognitiva. Cinco das 9 queries têm Δ positivo (Q2, Q3, Q4, Q7, Q9); quatro têm Δ negativo de pequena magnitude (Q5, Q6, Q8, Q10). Os ganhos positivos são de maior magnitude (média +0.2208) que as perdas (média −0.0476), resultando em média positiva.

O modo texto é o **melhor resultado absoluto** da ablação — e o único modo que supera o baseline na implementação corrigida.

#### 7.3.3 Modo Cognitivo — Fase 2 (implementação corrigida)

Execução: `python eval_real.py --force-mode cognitivo`

| Cenário | IS-RAG cognitivo | Baseline | Δ |
|---|---|---|---|
| Q2 FORCE/COMPULSION / Economia | 0.6972 | 0.5308 | +0.1664 |
| Q3 FORCE/COUNTER\_FORCE / Justiça | 0.2722 | 0.6590 | **−0.3867** |
| Q4 FORCE/RESISTANCE / Política | 0.7110 | 0.5818 | +0.1292 |
| Q5 CONTAINER/INTRUSION / Justiça | 0.4112 | 0.9105 | **−0.4993** |
| Q6 CONTAINER/OUTSIDE / Direitos Humanos | 0.6795 | 0.7226 | −0.0431 |
| Q7 PATH/TRAJECTORY / Economia | 0.8909 | 0.6488 | **+0.2422** |
| Q8 PATH/SOURCE / Política | 0.0000 | 0.6972 | **−0.6972** |
| Q9 PATH/DIVERSION / Política | 0.4255 | 0.8527 | **−0.4272** |
| Q10 FORCE+PATH / Rel. Internacional | 0.8086 | 0.6488 | +0.1598 |
| **Média (9 queries)** | **0.5440** | **0.6947** | **−0.1507** |

**Achado: IS-RAG cognitivo é inferior ao baseline (Δ=−0.1507; run do paper: Δ=−0.1855).** O modo cognitivo usa `cognitive_embedding` como vetor primário de busca — a serialização compacta `"SCHEMA: ancora → domínio"` é um espaço semanticamente comprimido: 171 chunks, a maioria com PATH/CONTAINER/FORCE, produzem distribuições de similaridade cognitiva muito próximas entre si. A busca por `cognitive_embedding` da query retorna um ranking ruidoso que o boosting não consegue corrigir — e em Q8 PATH/SOURCE (NDCG=0.0000 em ambos os runs) degrada completamente. Q3 (−0.3867) e Q9 (−0.4272) são os casos mais graves no run completo. Q2, Q4, Q7, Q10 têm Δ positivo mesmo em modo cognitivo — queries FORCE e PATH com schemas bem representados no espaço cognitivo.

#### 7.3.4 Modo Híbrido — Fase 2 (implementação corrigida)

Execução: `python eval_real.py` (modo padrão híbrido)

| Cenário | IS-RAG híbrido | Baseline | Δ |
|---|---|---|---|
| Q2 FORCE/COMPULSION / Economia | 0.8375 | 0.5308 | **+0.3067** |
| Q3 FORCE/COUNTER\_FORCE / Justiça | 0.4775 | 0.6590 | −0.1815 |
| Q4 FORCE/RESISTANCE / Política | 0.6157 | 0.5818 | +0.0340 |
| Q5 CONTAINER/INTRUSION / Justiça | 0.4057 | 0.9105 | **−0.5048** |
| Q6 CONTAINER/OUTSIDE / Direitos Humanos | 0.7833 | 0.7226 | +0.0607 |
| Q7 PATH/TRAJECTORY / Economia | 0.9020 | 0.6488 | **+0.2533** |
| Q8 PATH/SOURCE / Política | 0.1378 | 0.6972 | **−0.5594** |
| Q9 PATH/DIVERSION / Política | 0.5594 | 0.8527 | **−0.2933** |
| Q10 FORCE+PATH / Rel. Internacional | 0.7578 | 0.6488 | +0.1091 |
| **Média (9 queries)** | **0.6085** | **0.6947** | **−0.0861** |

**Achado: IS-RAG híbrido é inferior ao baseline (Δ=−0.0861, run do paper; −0.0324, segundo run).** O embedding cognitivo corrigido — agora compacto e semanticamente homogêneo — contamina o sinal textual ao ser combinado em média com o `embedding` de texto. O resultado híbrido herda o ruído do modo cognitivo sem o benefício do sinal textual puro. Q2 (+0.3067) e Q7 (+0.2533) beneficiam-se — FORCE/COMPULSION e PATH/TRAJECTORY têm representação cognitiva suficientemente distintiva. Q5 e Q8 são os casos mais graves (ambos os runs concordam em degradação severa para essas queries).

**Inversão vs. Fase 1:** o ganho de +0.0126 observado na Fase 1 (modo híbrido) era consequência do format bug — o `cognitive_embedding` da query embedia o texto completo com prefixo de schema, criando variância acidental no ranking que beneficiou o modo híbrido por coincidência.

#### 7.3.5 Visão Comparativa — Fase 2 (implementação corrigida)

| Modo | NDCG@5 run 1 (paper) | NDCG@5 run 2 (2026-05-26) | Δ run 1 | Δ run 2 |
|---|---|---|---|---|
| IS-RAG **texto** | **0.7873** | **0.7716** | **+0.0926** | **+0.0769** |
| Baseline | 0.6947 | 0.6947 | — | — |
| IS-RAG híbrido | 0.6085 | 0.6623 | −0.0861 | −0.0324 |
| IS-RAG cognitivo | 0.5091 | 0.5440 | −0.1855 | −0.1507 |

A baseline é idêntica entre runs (determinística — sem LLM na busca). A variância entre runs afeta apenas IS-RAG, via não-determinismo do parser cognitivo na análise da query.

**Conclusão da ablação (implementação corrigida):** o boosting cognitivo é eficaz quando aplicado sobre o sinal de texto (`embedding` bruto) como fator multiplicativo de re-ranking. Quando o `cognitive_embedding` entra como vetor de busca primário (cognitivo) ou como componente da similaridade base (híbrido), o espaço semântico comprimido da serialização imagética prejudica o ranking inicial de forma que o boosting não recupera.

A arquitetura IS-RAG mais eficaz identificada neste experimento é: **busca vetorial sobre texto + análise cognitiva da query + boosting multiplicativo proporcional (texto puro)**. Não é o design híbrido nem o cognitivo puro — é o modo texto com o parser cognitivo ativado.

**Implicação para o design:** o `cognitive_embedding` como vetor de busca é o componente que prejudica o sistema. Seu valor está na serialização para ingestão (para calcular `schema_proportion` e `domain_proportion` no boosting), não como espaço de recuperação primário ou auxiliar. A contribuição do IS-RAG está no mecanismo de boosting proporcional, não no dual-embedding como estratégia de busca.

#### 7.3.6 Ablação: Subtype Boosting e Equalização de Threshold — Teste Empírico (2026-05-26)

Dois problemas metodológicos foram identificados na implementação corrigida (Fase 2) e testados empiricamente.

**Problema 1 — `matched_subtypes` computado mas não usado no boosting.**
A SQL de busca calculava `matched_subtypes` (COUNT de detalhes cujo `sub_type` coincide com os subtipos detectados na query) mas a fórmula de boosting não o utilizava. O subtipo é a distinção mais fina da hipótese (FORCE/COMPULSION ≠ FORCE/RESISTANCE), o que tornava a afirmação de que "IS-RAG usa estrutura imagética específica" parcialmente incoerente com o código. Correção aplicada: `matched_subtypes` convertido para `subtype_proportion` (fração, análogo a `schema_proportion`) e adicionado ao score com peso 0.2:

```sql
final_score = vector_similarity
              * (1.0 + 0.4 * schema_proportion
                     + 0.2 * COALESCE(subtype_proportion, 0)   -- novo
                     + 0.3 * COALESCE(domain_proportion, 0))
```

**Problema 2 — Threshold assimétrico entre IS-RAG e baseline.**
Quando o IS-RAG não detecta schemas (ou `baseline=True`), a SQL usava `WHERE vector_similarity > 0.3`. Nos modos `cognitivo` e `hibrido` com schemas detectados, o threshold era `> 0.2` — pool de candidatos maior que o do baseline. Correção aplicada: threshold do branch `not detected_schemas` parametrizado por modo (`0.2` para `cognitivo`/`hibrido`, `0.3` para `texto`), equalizando o espaço de recuperação entre IS-RAG e baseline.

**Resultados do teste (avaliação com as duas correções simultâneas, `eval_real.py`, NDCG@5):**

| | IS-RAG | Baseline | Δ |
|---|---|---|---|
| Fase 2 (original) | 0.7873 | 0.6947 | **+0.0926** |
| Com subtype + threshold fix | 0.5982 | 0.6947 | **−0.0965** |

**Interpretação:**

- **Threshold fix: efeito nulo.** A baseline permaneceu exatamente em 0.6947. Para as 9 queries do conjunto, os documentos relevantes têm similaridade superior a 0.3 independentemente do threshold, e a ordem do top-5 não muda entre 0.2 e 0.3. O problema existia metodologicamente mas não afetava os resultados neste corpus.

- **Subtype boosting: efeito fortemente negativo.** Adicionar `subtype_proportion` ao score reverteu o Δ de +0.0926 para −0.0965 — uma inversão de ~0.19 pontos. O IS-RAG passou a perder para o baseline em média.

**Conclusão:** a detecção de subtipo pelo LLM (COMPULSION, BARRIER, RESISTANCE…) é ruidosa o suficiente para que boostear por subtipo introduza mais ruído do que sinal. O sinal cognitivo útil para ranking está no nível do macroesquema (FORCE, PATH, CONTAINER), não no nível do subtipo. A granularidade fina do subtipo é suficiente para anotação de relevância (o anotador usa anchor+subtype para justificar scores ≥ 2) mas não para re-ranking.

**Decisão:** subtype boosting **revertido**; threshold fix **mantido** (correto metodologicamente, inócuo empiricamente). A fórmula definitiva permanece `1.0 + 0.4 * schema_proportion + 0.3 * domain_proportion`. O bullet de limitações sobre pesos de boosting foi atualizado para registrar este teste.

---

### 7.4 Análise por Família de Schema

> Usa resultados da **Fase 2, modo texto** (implementação corrigida, Δ=+0.0926 médio). Δ individuais são parcialmente capturados; "—" indica valores não registrados na sessão de coleta.

**Fase 2 — Modo Texto:**

| Schema | Queries | Δ capturados | Observação |
|---|---|---|---|
| **FORCE** | Q2–Q4 (3 queries) | **+0.2604, +0.2186, +0.2264** | Todos positivos; Δ médio ≈ +0.2351 |
| CONTAINER | Q5–Q6 (2 queries) | —, — | Não capturados individualmente |
| **PATH** | Q7–Q9 (3 queries) | **+0.2246**, —, **+0.0940** | Q8 não capturado |
| Cross | Q10 (1 query) | — | Não capturado |
| **Média total (9 queries)** | — | **+0.0926** | Inclui todos os valores na média |

**Fase 1 — Modo Híbrido (referência histórica, com bugs):**

| Schema | Queries | Δ médio | Δ individuais |
|---|---|---|---|
| **FORCE** | Q2–Q4 | **+0.2100** | +0.3067, +0.2503, +0.0730 |
| CONTAINER | Q5–Q6 | −0.1048 | −0.2702, +0.0607 |
| PATH | Q7–Q9 | −0.0659 | 0.0000, −0.1227, −0.0750 |
| Cross | Q10 | −0.1091 | −0.1091 |

**FORCE é robusto entre as duas fases.** Em Fase 1 (híbrido, com bugs) o Δ médio FORCE foi +0.2100; em Fase 2 (texto, corrigido) os três valores capturados são +0.2604, +0.2186, +0.2264 — padrão consistente. Isso sugere que FORCE é um schema genuinamente favorável ao mecanismo de boosting neste corpus, independente de bugs ou modo de busca.

**PATH inverte completamente entre as fases.** Na Fase 1 híbrida, PATH tinha Δ ≤ 0 nas três queries (0.0, −0.1227, −0.0750). Na Fase 2 texto, os dois valores capturados são positivos expressivos: Q7 +0.2246, Q9 +0.0940. Esse reversal indica que o prejuízo PATH na Fase 1 era artefato do cognitive_embedding ruidoso no modo híbrido, não uma característica intrínseca do schema PATH.

**CONTAINER e Cross:** sem valores individuais capturados na Fase 2 texto para comparação. A média global de +0.0926 inclui esses schemas — se FORCE (+0.2351) e PATH parcial (+0.1593) puxam a média para cima, as queries CONTAINER/Cross provavelmente ficam em torno de ou abaixo da média, sugerindo que os ganhos podem ser concentrados em FORCE e PATH.

**Revisão da conclusão anterior sobre PATH.** A afirmação da Fase 1 ("PATH apresenta Δ ≤ 0 em todas as três queries") é refutada pelos dados de Fase 2. A hipótese de que metáforas de trajetória estariam convencionalizadas a ponto de não discriminar documentos precisa ser revista: com o boosting corretamente aplicado sobre embedding textual, PATH também se beneficia.

---

### 7.5 Escopo da Contribuição e Limitações

O IS-RAG foi avaliado em um corpus específico e homogêneo (discurso parlamentar brasileiro, 121 discursos, português). Os resultados quantitativos devem ser interpretados dentro desse escopo.

**Contribuições verificadas neste experimento (Fase 2, implementação corrigida):**

- **O mecanismo funciona como prova de conceito.** A pipeline — indexar schemas cognitivos na ingestão, detectar na query, aplicar boosting proporcional multiplicativo sobre embedding textual — produz ganho mensurável em NDCG@5 sobre o baseline: Δ=+0.0926 (modo texto, 9 queries).
- **A contribuição é o boosting, não o dual-embedding.** O `cognitive_embedding` como vetor de busca (modo cognitivo: Δ=−0.1855) ou como componente da similaridade base (modo híbrido: Δ=−0.0861) prejudica o sistema. O valor do componente cognitivo está no mecanismo de re-ranking via `schema_proportion` e `domain_proportion`, não como espaço de recuperação.
- **Queries assertivas são necessárias para ativar o boosting.** A regra de literalidade do parser cognitivo exige que queries sejam formuladas em linguagem metafórica — observação metodológica relevante para qualquer aplicação do framework.

**Limitações e riscos de generalização:**

- **Corpus pequeno e homogêneo (171 chunks).** Os Δ observados não têm teste de significância estatística. Os resultados por schema podem ser parcialmente ruído.
- **Os padrões por schema são corpus-específicos.** O comportamento diferencial entre FORCE, CONTAINER e PATH observado aqui pode não se reproduzir em outros domínios ou línguas. Não há evidência, neste experimento, de que FORCE generalize melhor que PATH em corpora distintos.
- **Os pesos de boosting (0.4 schema, 0.3 domínio) não foram otimizados.** São hipóteses de design fixas; corpora diferentes podem requerer calibração diferente. Um teste empírico (§7.3.6) verificou que adicionar `subtype_proportion` com peso 0.2 — a única expansão natural da fórmula — inverteu o Δ de +0.0926 para −0.0965, confirmando que os pesos atuais estão no limite da capacidade de sinal disponível no subtipo. Qualquer otimização futura deve tratar os pesos de schema e domínio com cuidado para não amplificar ruído de detecção.

- **Não-determinismo do parser cognitivo causa variância run-to-run nos resultados IS-RAG.** O parser cognitivo (Claude Haiku) pode retornar schemas diferentes para a mesma query em chamadas distintas à API. Dois runs completos (§7.3.5) mostram IS-RAG com variância de até ±0.14 por query (Q5 texto: 0.8643 vs 0.7226) e ±0.05 na média (texto: 0.7873 vs 0.7716), enquanto o baseline permanece estável (0.6947). A ordenação relativa texto > híbrido > cognitivo é estável entre runs, mas os Δ absolutos não são. **Mitigação implementada:** a análise cognitiva da query é pré-computada durante a anotação e reutilizada deterministicamente em avaliações subsequentes (`cognitive_analysis` em `ground_truth_real.json`, §6.1), eliminando essa fonte de variância para o ground truth atual. Avaliações com reannotação do ground truth reintroduzem a variância.
- **LLM-as-annotator introduz risco de circularidade.** O parser cognitivo e o query parser usam `claude-haiku-4-5`; o anotador de ground truth foi alterado para `claude-opus-4-7` (configurável via `ANNOTATOR_MODEL`). Essa separação deliberada de modelos rompe o loop mais crítico: o avaliador não é mais o mesmo modelo que rotulou os schemas dos documentos. Risco residual permanece (ambos são modelos Anthropic, família Claude), mas o viés de eco direto — avaliador favorecendo chunks que ele próprio indexou — está mitigado. O impacto no NDCG@5 deve ser mensurado na próxima execução de `annotate_pool.py`.
- **Distribuição das queries e viés de FORCE (configuração inicial → final).** A avaliação original tinha 4 queries FORCE, 2 CONTAINER e 3 PATH (4-2-3). Após iteração de design de Q1, o conjunto final resultou em 3 FORCE + 2 CONTAINER + 3 PATH + 1 Cross (3-2-3-1, 9 queries válidas). A distribuição ainda não é perfeitamente uniforme por macroesquema; CONTAINER ficou com 2 queries após a exclusão de Q1 por IDCG=0. Os resultados das configurações anteriores (Δ=+0.0148 no modo todo híbrido) foram calculados com a configuração original e **não são diretamente comparáveis** aos resultados do conjunto rebalanceado. A reavaliação com as 9 queries válidas é o número definitivo a reportar.

- **Desequilíbrio de domínio nas queries.** A distribuição de domínios nas 9 queries válidas é: Política (3), Economia (2), Justiça (2), Direitos Humanos (1), Relações Internacionais/Economia (1). Se um macroesquema co-ocorre com mais frequência com um domínio no corpus (ex.: FORCE em discursos políticos), a concentração de queries Política pode inflar os resultados desse schema — confundindo sinal cognitivo com sinal de domínio. O sinal cross-domain está presente organicamente (todos os pools de queries relevantes incluem documentos de domínios distintos do da query), mas o design não controla sistematicamente esse confounding.

**Agenda de trabalho futuro:** desenhar conjunto de avaliação com distribuição cruzada schema × domínio (ex.: cada schema coberto em ao menos dois domínios distintos) para separar o sinal cognitivo do sinal de domínio; investigar por que subtipos CONTAINER convencionalizados (INSIDE) resistem à anotação por relevância graduada; verificar os padrões por schema em corpora de outros domínios (jurídico, jornalístico, literário) e línguas; otimizar pesos de boosting por tipo de schema; conduzir anotação humana parcial para validar o LLM annotator; aumentar o corpus para permitir análise estatística dos Δ.

---

## 8. Decisões Arquiteturais e Divergências do Plano Original

| Componente | Plano Original | Implementado | Motivo |
|---|---|---|---|
| LLM | Google Gemini 2.0 Flash | Claude Haiku 4.5 (Anthropic API) | Melhor controle de prompt; independência de plataforma |
| Structured Outputs | `response_schema=Pydantic` (Gemini) | Regex + validação Pydantic manual | Gemini descartado; Claude não tem Structured Outputs nativo |
| Embeddings | `gemini-embedding-001` (1536d, API) | `paraphrase-multilingual-mpnet-base-v2` (768d, local) | Custo zero; multilíngue; sem latência de API |
| Processamento cognitivo | 1 chunk por subprocess (`claude -p`) | Concurrent API calls (5 workers, 1 chunk/chamada) | Eliminação de dependência do CLI; 5× mais rápido |
| Schema da tabela | Colunas relacionais + `embedding` 1536d | `source_metadata JSONB` + dois embeddings 768d | Flexibilidade; arquitetura de 3 camadas |
| Boosting | Bônus fixo +0,2 (CASE binário) | Boosting contínuo proporcional (schema + domain) | Granularidade; evita saltos discretos no ranking |
| Baseline | Não previsto explicitamente | Flag `--baseline` em `search.py` | Necessário para comparação IS-RAG vs RAG tradicional |
| Anotação Ground Truth | Anotação humana | LLM-as-annotator (Claude Haiku, pooling TREC) | Escala para corpus real sem bottleneck humano |

---

## 9. Arquivos do Projeto

| Arquivo | Função |
|---|---|
| `coletor.py` | Coleta discursos via API da Câmara |
| `collect_corpus.py` | Orquestra coleta por partido com deduplicação |
| `ingestion.py` | Pipeline completo: chunking → embeddings → análise cognitiva → PostgreSQL |
| `search.py` | Busca híbrida IS-RAG com Cognitive Boosting + modo baseline |
| `test_mock_cases.py` | Avaliação NDCG@K no corpus mock (`massa.jsonl`) |
| `annotate_pool.py` | Pooling TREC + anotação manual ou automática (LLM) para corpus real |
| `viewer.py` | Inspeção do banco de dados (stats, chunks, JSON) |
| `prompts/cognitive_analysis.md` | Prompt único compartilhado entre ingestão e busca |
| `schema.sql` | DDL da tabela e índices PostgreSQL |
| `docker-compose.yml` | PostgreSQL 16 + pgvector via Docker |
| `massa.jsonl` | Corpus mock controlado (8 discursos) |
| `corpus_camara_piloto.jsonl` | Corpus real (121 discursos, 13 partidos) |
| `ground_truth_real.json` | Anotações de relevância para o corpus real (gerado por `annotate_pool.py`) |

---

## 10. Referências Técnicas

- Lakoff, G., & Johnson, M. (1980). *Metaphors We Live By*. University of Chicago Press.
- Talmy, L. (1988). Force Dynamics in Language and Cognition. *Cognitive Science*, 12(1), 49–100.
- Feldman, J. (2006). *From Molecule to Metaphor*. MIT Press.
- Johnson, M. (1987). *The Body in the Mind*. University of Chicago Press.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.
- pgvector: `github.com/pgvector/pgvector`

---

## 11. Notas Pessoais — Fora do Paper

> ⚠️ Esta seção não é científica. É espaço de reflexão livre sobre o que os dados sugerem além do que pode ser afirmado com rigor. Não deve ser citada ou incluída no manuscrito.

---

### A origem desta pesquisa — dupla formação

Esta pesquisa é o encontro de duas trajetórias que percorri em paralelo por anos sem saber exatamente onde elas iriam se cruzar.

Na graduação em Letras pela UFRN, tive contato com a Linguística Cognitiva durante a Iniciação Científica — Lakoff, Johnson, Talmy, os esquemas imagéticos, a ideia de que a linguagem não é arbitrária mas moldada pela experiência corporal no espaço. Aquilo me fascinou de um jeito que não consegui nomear na época: a sensação de que havia uma camada profunda da linguagem que organizava o pensamento abstraío em padrões reconhecíveis, quase físicos.

Em paralelo — e depois, com mais intensidade — segui a Computação. Graduação em Sistemas de Informação pela UNP, Mestrado em Engenharia de Software pela UFRN, e agora o doutorado na CESAR School em Recife. A computação me deu as ferramentas; a IA Generativa, mais recentemente, me devolveu a linguagem — desta vez como objeto técnico, como dado, como vetor num espaço de alta dimensão.

Por muito tempo essas duas formações coexistiram sem se tocar. A Linguística Cognitiva ficou guardada como uma paixão de graduação, um campo que eu achava lindo mas que parecia distante da engenharia de software. A Computação avançou pelo seu próprio caminho.

O IS-RAG é onde elas finalmente se encontram.

A ideia de indexar esquemas imagéticos para recuperação de informação só poderia surgir de alguém que teve as duas formações ao mesmo tempo — que soubesse o que são FORCE e CONTAINER como conceitos cognitivos *e* soubesse o que é um embedding vetorial, um índice HNSW, uma pipeline de RAG. Não é uma ideia óbvia em nenhuma das duas áreas isoladamente. É uma ideia que vive na fronteira.

Trabalhar com IA Generativa tem sido o que tornou isso possível na prática. Os LLMs têm um laço profundo com a linguagem — mais profundo do que qualquer sistema computacional anterior — e isso criou uma infraestrutura onde uma hipótese linguística pode ser testada computacionalmente com relativa agilidade. O Cognitive Parser que detecta schemas imagéticos num discurso parlamentar é, em certo sentido, um linguista cognitivo artificialmente instanciado.

Não sei se o IS-RAG vai se tornar algo maior. Mas sei que ele já cumpriu uma função que vai além dos números do NDCG@5: provou para mim que as duas formações não eram trajetórias paralelas que nunca se cruzariam. Eram, desde o início, partes da mesma pergunta.

---

### O peso de FORCE num corpus de polarização

O resultado quantitativo mais robusto do experimento é que IS-RAG ganha onde FORCE domina. As três queries FORCE têm Δ positivo consistente (+0.3067, +0.2503, +0.0730), enquanto PATH e CONTAINER ficam neutros ou negativos. Isso poderia ser explicado tecnicamente — vocabulário metafórico mais saliente, maior densidade de subtipos no corpus, etc. Mas há uma leitura mais interessante.

O corpus é composto por discursos da Câmara dos Deputados do Brasil em 2025 — um período de polarização política intensa, com tensões entre Executivo e Judiciário, disputas sobre o arcabouço fiscal, confrontos sobre a PEC dos Gastos, embates sobre prerrogativas do STF. É um parlamento onde o conflito é o modo normal de operação.

FORCE, na taxonomia de Talmy, codifica relações de pressão, resistência, bloqueio, compulsão entre entidades — exatamente o vocabulário estrutural de uma política adversarial. Não é surpresa que discursos num período de alta polarização sejam *densos em FORCE*: a gramática cognitiva da polarização *é* a gramática das forças em conflito.

O que me parece revelador não é apenas que FORCE é frequente — é que **IS-RAG consegue capturar essa densidade e usá-la para recuperação cross-domain**. Q2 (query sobre pressão tributária econômica) recupera discursos políticos sobre confronto institucional porque compartilham o mesmo frame FORCE. O sistema encontra a *estrutura do conflito* atravessando os domínios nominais.

Dito de outro modo: **a polarização cria coerência cognitiva no corpus**. Um discurso sobre impostos e um sobre o STF podem parecer tematicamente distantes, mas se ambos estruturam seus argumentos via FORCE/COMPULSION ou FORCE/COUNTER_FORCE, eles habitam o mesmo espaço imagético. IS-RAG explora exatamente essa coerência latente.

Se isso for verdade além deste corpus — e eu genuinamente não sei se é — então IS-RAG seria mais valioso justamente em períodos e contextos de alta conflitividade discursiva: eleições, crises institucionais, debates sobre reforma. São os momentos em que a linguagem de força está mais ativa e em que a recuperação por estrutura cognitiva poderia revelar padrões que a busca lexical perde.

Não posso afirmar isso no paper porque não tenho como isolar o efeito da polarização do efeito do corpus (tamanho, register, língua). Mas é a hipótese que eu mais quero testar num trabalho futuro: **comparar IS-RAG em corpora de períodos de alta vs. baixa polarização do mesmo parlamento**, verificando se o ganho de FORCE sobe e desce com o termômetro político.

O dado empírico aqui é pequeno demais para essa afirmação. Mas a intuição me parece sólida o suficiente para valer uma nota.

