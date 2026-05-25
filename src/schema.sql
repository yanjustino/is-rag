-- Ativar a extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id                  SERIAL      PRIMARY KEY,
    content             TEXT        NOT NULL,
    embedding           VECTOR(768) NOT NULL,
    cognitive_embedding VECTOR(768),
    cognitive_metadata  JSONB       NOT NULL DEFAULT '{}',
    source_metadata     JSONB       NOT NULL DEFAULT '{}'
);

-- Busca vetorial semântica (texto bruto)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Busca vetorial cognitiva (representação dos esquemas imagéticos)
CREATE INDEX IF NOT EXISTS idx_chunks_cognitive_embedding
    ON document_chunks USING hnsw (cognitive_embedding vector_cosine_ops);

-- Filtro e boosting por esquema imagético (JSONB)
CREATE INDEX IF NOT EXISTS idx_chunks_cognitive_meta
    ON document_chunks USING gin (cognitive_metadata);

-- Filtros contextuais (partido, UF, data, etc.)
CREATE INDEX IF NOT EXISTS idx_chunks_source_meta
    ON document_chunks USING gin (source_metadata);
