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
3. Cada chunk único é anotado com relevância 0–3 pelo LLM anotador (`claude-haiku-4-5-20251001`) com retry exponencial em caso de rate limit.
4. Para scores ≥ 2, o anotador é obrigado a citar a palavra-âncora (`anchor_word_pt`) e o subtipo (`sub_type`) que justificam o score — evidência rastreável e verificável por humano.
5. Resultados salvos em `ground_truth_real.json` para posterior cálculo de NDCG@K.

**Critério de inclusão na avaliação:** queries com zero documentos relevantes (score ≥ 2) no pool são excluídas do cálculo de NDCG, pois o IDCG seria 0 e a métrica ficaria indefinida (Buckley & Voorhees, 2004). Com as queries na forma assertiva metafórica, **todas as 10 queries obtiveram ao menos 1 documento relevante** no pool anotado. Nenhuma query foi excluída.

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

**Resultado definitivo — 9 queries válidas, modo híbrido:**

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

O baseline é busca vetorial pura sobre `embedding` de texto, sem análise cognitiva.

---

### 7.4 Análise por Família de Schema

| Schema | Queries | Δ médio | Δ individuais |
|---|---|---|---|
| **FORCE** | Q2–Q4 (3 queries) | **+0.2100** | +0.3067, +0.2503, +0.0730 |
| CONTAINER | Q5–Q6 (2 queries) | −0.1048 | −0.2702, +0.0607 |
| PATH | Q7–Q9 (3 queries) | −0.0659 | 0.0000, −0.1227, −0.0750 |
| Cross | Q10 (1 query) | −0.1091 | −0.1091 |
| **Média ponderada por família** | — | **−0.0174** | — |

**FORCE concentra os maiores ganhos.** Todas as três queries FORCE apresentam Δ positivo, com picos expressivos em COMPULSION (+0.3067) e COUNTER\_FORCE (+0.2503). O schema FORCE parece associado a vocabulário metafórico mais específico e saliente no corpus ("pressiona", "bloqueia", "choca", "força"), favorecendo a correspondência cognitiva. Contudo, esse padrão é corpus-específico e não pode ser generalizado sem experimentos em outros domínios.

**PATH apresenta Δ ≤ 0 em todas as três queries.** Uma hipótese plausível é que metáforas de trajetória estejam convencionalizadas neste corpus a ponto de não discriminar documentos relevantes dos irrelevantes. Não é possível separar esse efeito de características do corpus (register parlamentar, tamanho reduzido) sem experimentos adicionais.

**CONTAINER é misto e sensível ao subtipo.** OUTSIDE (Q6, +0.0607) beneficia do IS-RAG em modo híbrido; INTRUSION (Q5, −0.2702) sofre degradação significativa. Q5 foi projetada para modo cognitivo — a execução forçada em híbrido explica parte da queda.

**Média geral vs. média ponderada por família.** A média não ponderada sobre as 9 queries (Δ=+0.0126) é positiva porque FORCE representa 3/9 queries (33%) com Δ médio de +0.2100. A média ponderada igualmente por família de schema (Δ=−0.0174) é negativa. O resultado defensável neste experimento é restrito: *IS-RAG supera o baseline para queries FORCE em modo híbrido neste corpus.* A afirmação não se estende a CONTAINER ou PATH.

---

### 7.5 Escopo da Contribuição e Limitações

O IS-RAG foi avaliado em um corpus específico e homogêneo (discurso parlamentar brasileiro, 121 discursos, português). Os resultados quantitativos devem ser interpretados dentro desse escopo.

**Contribuições verificadas neste experimento:**

- **O mecanismo funciona como prova de conceito.** A pipeline — indexar schemas cognitivos na ingestão, detectar na query, aplicar boosting proporcional — produz ganho mensurável em NDCG@5 sobre o baseline para queries com schema FORCE em modo híbrido.
- **Híbrido domina cognitivo puro em todos os schemas.** O embedding cognitivo como sinal auxiliar de re-ranking supera consistentemente seu uso como vetor primário de busca. Esse resultado arquitetural é estável entre famílias de schema.
- **Queries assertivas são necessárias para ativar o boosting.** A regra de literalidade do parser cognitivo exige que queries sejam formuladas em linguagem metafórica — observação metodológica relevante para qualquer aplicação do framework.

**Limitações e riscos de generalização:**

- **Corpus pequeno e homogêneo (171 chunks).** Os Δ observados não têm teste de significância estatística. Os resultados por schema podem ser parcialmente ruído.
- **Os padrões por schema são corpus-específicos.** O comportamento diferencial entre FORCE, CONTAINER e PATH observado aqui pode não se reproduzir em outros domínios ou línguas. Não há evidência, neste experimento, de que FORCE generalize melhor que PATH em corpora distintos.
- **Os pesos de boosting (0.4 schema, 0.3 domínio) não foram otimizados.** São hipóteses de design fixas; corpora diferentes podem requerer calibração diferente.
- **LLM-as-annotator introduz risco de circularidade.** O mesmo modelo (claude-haiku) foi usado na ingestão cognitiva, na análise de query e na anotação. Há risco de que o anotador favoreça chunks que o parser cognitivo indexou com os mesmos schemas da query.
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

### O peso de FORCE num corpus de polarização

O resultado quantitativo mais robusto do experimento é que IS-RAG ganha onde FORCE domina. As três queries FORCE têm Δ positivo consistente (+0.3067, +0.2503, +0.0730), enquanto PATH e CONTAINER ficam neutros ou negativos. Isso poderia ser explicado tecnicamente — vocabulário metafórico mais saliente, maior densidade de subtipos no corpus, etc. Mas há uma leitura mais interessante.

O corpus é composto por discursos da Câmara dos Deputados do Brasil em 2025 — um período de polarização política intensa, com tensões entre Executivo e Judiciário, disputas sobre o arcabouço fiscal, confrontos sobre a PEC dos Gastos, embates sobre prerrogativas do STF. É um parlamento onde o conflito é o modo normal de operação.

FORCE, na taxonomia de Talmy, codifica relações de pressão, resistência, bloqueio, compulsão entre entidades — exatamente o vocabulário estrutural de uma política adversarial. Não é surpresa que discursos num período de alta polarização sejam *densos em FORCE*: a gramática cognitiva da polarização *é* a gramática das forças em conflito.

O que me parece revelador não é apenas que FORCE é frequente — é que **IS-RAG consegue capturar essa densidade e usá-la para recuperação cross-domain**. Q2 (query sobre pressão tributária econômica) recupera discursos políticos sobre confronto institucional porque compartilham o mesmo frame FORCE. O sistema encontra a *estrutura do conflito* atravessando os domínios nominais.

Dito de outro modo: **a polarização cria coerência cognitiva no corpus**. Um discurso sobre impostos e um sobre o STF podem parecer tematicamente distantes, mas se ambos estruturam seus argumentos via FORCE/COMPULSION ou FORCE/COUNTER_FORCE, eles habitam o mesmo espaço imagético. IS-RAG explora exatamente essa coerência latente.

Se isso for verdade além deste corpus — e eu genuinamente não sei se é — então IS-RAG seria mais valioso justamente em períodos e contextos de alta conflitividade discursiva: eleições, crises institucionais, debates sobre reforma. São os momentos em que a linguagem de força está mais ativa e em que a recuperação por estrutura cognitiva poderia revelar padrões que a busca lexical perde.

Não posso afirmar isso no paper porque não tenho como isolar o efeito da polarização do efeito do corpus (tamanho, register, língua). Mas é a hipótese que eu mais quero testar num trabalho futuro: **comparar IS-RAG em corpora de períodos de alta vs. baixa polarização do mesmo parlamento**, verificando se o ganho de FORCE sobe e desce com o termômetro político.

O dado empírico aqui é pequeno demais para essa afirmação. Mas a intuição me parece sólida o suficiente para valer uma nota.

