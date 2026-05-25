# Projeto IS-RAG (Image-Schematic Retrieval-Augmented Generation)
## Documento de Especificação Conceitual e Plano de Desenvolvimento

Este documento serve como a especificação técnica inicial para o desenvolvimento do **IS-RAG** (ou **MetRAG**), um framework inovador de RAG Híbrido que estende a busca vetorial tradicional por meio da extração e pareamento de **Esquemas Imagéticos (Image Schemas)** fundamentados na Linguística Cognitiva (Lakoff, Johnson, Feldman).

---

## 1. Visão Geral e Objetivo
O RAG tradicional falha em capturar estruturas de pensamento subjacentes e ideologias ocultas em textos complexos quando há um **Lexical-Cognitive Gap** (o usuário e o texto usam metáforas físicas distintas para expressar o mesmo conceito abstrato). O objetivo do IS-RAG é criar um pipeline de busca capaz de classificar, indexar e recuperar documentos com base em suas estruturas cognitivas profundas.

### Stack Tecnológica Escolhida:
- **Linguagem:** Python 3.11+
- **Banco de Dados:** PostgreSQL 16+ com a extensão `pgvector` e suporte a `JSONB`.
- **Camada de LLM:** ~~Google Gemini API~~ → **Claude Code CLI** (`claude -p -` via subprocess stdin). Structured Outputs substituídos por Pydantic + extração de JSON da resposta.
- **Embeddings:** ~~`gemini-embedding-001` (1536d)~~ → **`paraphrase-multilingual-mpnet-base-v2`** via `sentence-transformers` (768d, local, sem custo de API).

---

## 2. Conceitos-Chave da Linguística Cognitiva Aplicada

### A. Esquemas Imagéticos (Image Schemas)
Padrões dinâmicos abstratos de nossas experiências sensório-motoras usados para estruturar conceitos abstratos. O projeto focará inicialmente em três macroesquemas:
1. **CONTAINER (Recipiente):** Noções de dentro/fora, limites, contenção, invasão.
2. **PATH (Percurso):** Noções de origem, trajetória, destino, direção, barreiras.
3. **FORCE (Força):** Noções de barreira, impulso, resistência, atração, bloqueio.

### B. Componentes Arquiteturais do Sistema
- **Cognitive Parser:** Componente em Python que analisa os chunks de texto em português e extrai os Esquemas Imagéticos mapeados estruturalmente em formato JSON (chaves em inglês para padronização acadêmica).
- **Cognitive Boosting:** Algoritmo SQL híbrido que pondera o score de similaridade vetorial (cosseno) injetando um bônus quando há correspondência estrutural de esquemas entre a query do usuário e o chunk armazenado.

---

## 3. Banco de Dados e Modelagem de Dados

### Estrutura da Tabela PostgreSQL
```sql
-- Ativar a extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Criar a tabela de documentos/chunks
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    id_interno VARCHAR(50) NOT NULL,
    orador_nome VARCHAR(100),
    orador_partido VARCHAR(20),
    orador_uf VARCHAR(5),
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- 1536 para text-embedding-3-small ou Ada-002
    cognitive_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Criar índices para performance
CREATE INDEX idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_cognitive ON document_chunks USING gin (cognitive_metadata);
```

### Formato do Campo `cognitive_metadata` (JSONB)
```json
{
  "schemas": ["PATH", "FORCE"],
  "details": [
    {
      "schema_name": "FORCE",
      "sub_type": "BARRIER",
      "anchor_word_pt": "barrou",
      "target_domain_pt": "planos de expansão"
    }
  ]
}
```

---

## 4. Estruturas Pydantic para o Cognitive Parser
```python
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

client = genai.Client()  # Lê GEMINI_API_KEY do ambiente

class ImageSchemaDetail(BaseModel):
    schema_name: str = Field(description="Must be CONTAINER, PATH, or FORCE")
    sub_type: str = Field(description="Ex: BARRIER, COMPULSION, ENABLEMENT, INSIDE, OUTSIDE, SOURCE, TARGET")
    anchor_word_pt: str = Field(description="A palavra ou expressão em português que disparou o esquema")
    target_domain_pt: str = Field(description="O conceito abstrato ou domínio alvo em português sendo discutido")

class CognitiveAnalysis(BaseModel):
    schemas: List[str] = Field(description="List of unique schema names present in the text (CONTAINER, PATH, FORCE)")
    details: List[ImageSchemaDetail] = Field(description="Detailed image schematic mappings found in the text")
```

---

## 5. Scripts de Coleta e Ingestão

> Consulte os scripts reais no workspace:
> - [coletor.py](file:///Users/yanjustino/Documents/yanjustino.me/yanjustino.research/is-rag/coletor.py): Coleta discursos de deputados via API da Câmara e salva em `corpus_camara_piloto.jsonl`.
> - [ingestion.py](file:///Users/yanjustino/Documents/yanjustino.me/yanjustino.research/is-rag/ingestion.py): Pipeline de chunking semântico, extração de Image Schemas via `gemini-2.0-flash` (Structured Outputs) e geração de embeddings 1536d via `gemini-embedding-001`, com persistência no PostgreSQL.
> - [search.py](file:///Users/yanjustino/Documents/yanjustino.me/yanjustino.research/is-rag/search.py): Busca híbrida IS-RAG com Cognitive Boosting SQL sobre `pgvector`.

```python
# Exemplo simplificado do Cognitive Parser com Google Gemini
from google import genai
from google.genai import types

client = genai.Client()  # Lê GEMINI_API_KEY do ambiente

# Extrair Image Schemas com Structured Output
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=f"Analise o texto: {chunk}",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CognitiveAnalysis,
    ),
)
metadata = response.parsed.model_dump()

# Gerar embedding 1536d com Gemini
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunk,
    config=types.EmbedContentConfig(output_dimensionality=1536)
)
embedding = result.embeddings[0].values
```

---

## 6. Lógica do Cognitive Boosting (SQL)
```sql
-- Exemplo de query buscando similaridade vetorial + bônus por esquema imagético coincidente
WITH vector_search AS (
    SELECT 
        id,
        content,
        cognitive_metadata,
        (1 - (embedding <=> %s::vector)) AS vector_similarity
    FROM document_chunks
)
SELECT 
    id,
    content,
    vector_similarity,
    -- Se o JSONB contiver o esquema coincidente nos metadados, adiciona um bônus de 0.2 no score
    CASE 
        WHEN cognitive_metadata @> %s::jsonb THEN vector_similarity + 0.2
        ELSE vector_similarity
    END AS final_score
FROM vector_search
WHERE vector_similarity > 0.4 -- Limiar mínimo de similaridade semântica
ORDER BY final_score DESC
LIMIT 5;
```

---

## 7. Plano de Desenvolvimento (Roadmap do Experimento)

### Fase 1: Coleta e Preparação do Corpus (Semana 1)
- [x] Implementar e rodar o script de extração da **API de Dados Abertos da Câmara dos Deputados** → `coletor.py`.
- [x] Filtrar discursos políticos com mais de 300 caracteres para garantir densidade argumentativa.
- [x] Implementar estratégia de *chunking* semântico (blocos de aproximadamente 150 palavras) mantendo metadados do orador → `semantic_chunking()` em `ingestion.py`.
- [x] Salvar os dados tratados em formato `corpus_camara_piloto.jsonl`.

### Fase 2: Infraestrutura de Banco de Dados (Semana 1)
- [x] Criar arquivo `docker-compose.yml` para configurar a instância do PostgreSQL com a extensão `pgvector`. Schema aplicado automaticamente via `docker-entrypoint-initdb.d`.
- [x] Iniciar o contêiner do banco usando `docker compose up -d`.
- [x] Criar a tabela `document_chunks` e configurar os índices GIN e HNSW → `schema.sql`. **Divergência:** schema refatorado para arquitetura de 3 camadas com `source_metadata JSONB` e `cognitive_embedding VECTOR(768)` separados; colunas relacionais do plano original eliminadas.

### Fase 3: Desenvolvimento do Cognitive Parser (Semana 2)
- [x] Definir schemas estritos com Pydantic para garantir integridade do JSON de saída → `ImageSchemaDetail`, `CognitiveAnalysis` em `ingestion.py` e `search.py`.
- [x] Desenvolver prompts especializados baseados em *Metaphors We Live By* (Lakoff & Johnson) → `prompts/cognitive_analysis.md` (few-shot, externalizado, compartilhado entre ingestão e busca).
- [x] Integrar com LLM via Claude Code CLI (`claude -p -` via subprocess stdin) com extração de JSON por regex + validação Pydantic. **Divergência:** Gemini substituído por Claude; Structured Outputs substituídos por parsing manual com fallback.
- [x] Criar scripts de processamento em lote → `ingestion.py` com log de progresso por chunk (`embedding → claude → salvando`).

### Fase 4: Busca Metafórica Híbrida (Semana 3)
- [x] Implementar o classificador de query → `analyze_query()` em `search.py`; detecta `schemas`, `sub_types` e `domains`.
- [x] Desenvolver a query SQL de busca híbrida com Cognitive Boosting → `search.py`. **Além do plano:** boosting contínuo proporcional (`schema_proportion` + `domain_proportion`) em 3 modos (texto / cognitivo / híbrido), substituindo o bônus fixo de +0.2 do plano original.
- [~] Implementar interface de comparação RAG Tradicional vs IS-RAG → modo sem esquemas detectados já executa busca vetorial pura como baseline, mas não há relatório comparativo lado a lado explícito.

### Fase 5: Validação e Avaliação (Semana 4)
- [~] Criar conjunto de dados de referência (Ground Truth) → `massa.jsonl` com 8 discursos mock controlados (5 políticos + 3 cross-domain em saúde, educação e meio ambiente) e `test_mock_cases.py` com 4 cenários anotados com relevância graduada (0–3). **Pendente:** escala para queries adicionais com anotação humana ou comitê de LLMs.
- [x] Avaliar quantitativamente usando **NDCG@K** comparando RAG Tradicional (busca vetorial pura via `--baseline`) com IS-RAG (busca cognitiva + boosting proporcional). MRR descartado: inadequado para cenários com múltiplos documentos relevantes em graus distintos — satura em 1.0 independentemente do ranking.

  | Cenário | Modo | NDCG@5 IS-RAG | NDCG@5 Baseline | Δ |
  |---|---|---|---|---|
  | PATH / Economia como jornada | hibrido | 0.9762 | 0.9434 | +0.0329 |
  | FORCE / Obstrução legislativa | texto | 0.7027 | 0.8243 | **-0.1217** |
  | CONTAINER / Estado como estrutura fechada | cognitivo | 0.8595 | 0.8954 | -0.0359 |
  | FORCE cross-domain / Meio Ambiente | cognitivo | 0.9679 | 0.5730 | **+0.3949** |
  | **Média** | — | **0.8766** | **0.8090** | **+0.0676** |

  Achado principal: no cenário cross-domain (4), o baseline não recuperou o documento alvo (rank=fora do top-5); o IS-RAG o recuperou em rank 1 via schema FORCE compartilhado entre domínios lexicalmente distintos.

- [~] Realizar análise qualitativa (Estudos de Caso): cenário 4 documentado como caso de sucesso (baseline_rank=fora, IS-RAG_rank=1, Δ=+0.3949). **Limitação identificada:** falso positivo de schema CONTAINER no cenário 2 (query FORCE detectada como CONTAINER+FORCE pelo Cognitive Parser), causando regressão de -0.1217 — candidato a análise de erro no paper.
- [ ] Escrever o pre-print em inglês e preparar para submissão no arXiv.

---

## 8. Estrutura de Escrita do Paper (arXiv Pre-print)
O paper será estruturado em inglês, abordando a generalização da arquitetura fora do eixo do inglês:
- **Title:** `IS-RAG: Image-Schematic Retrieval-Augmented Generation for Deep Cognitive Search in Political Discourse`
- **Abstract:** O problema do Lexical-Cognitive Gap nos LLM-embeddings atuais e como a busca híbrida com esquemas imagéticos de Lakoff resolve isso, demonstrando a tese em Português.
- **Introduction:** Fundamentos da Cognição Corporificada e limitações dos embeddings semânticos densos tradicionais.
- **Related Work:** Detecção de metáforas e modelos computacionais (Wachowiak & Gromann, Shutova, Feldman).
- **Methodology:** O pipeline do IS-RAG (Cognitive Parser, Hibrid Indexing, Cognitive Boosting).
- **Experiments & Evaluation:** Resultados quantitativos (NDCG@K, MRR) e qualitativos (Estudos de caso em discursos políticos).
- **Discussion & Future Work:** A universalidade cultural dos esquemas sensório-motores na IA.
