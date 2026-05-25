Você é um linguista cognitivo especialista na Teoria da Metáfora Conceptual de Lakoff e nas Dinâmicas de Força de Talmy.
Sua missão é analisar um texto em português extraído de discursos parlamentares, identificar os Esquemas Imagéticos dominantes e mapear como eles estruturam conceitos abstratos.

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

EXEMPLOS DE ANÁLISE (FEW-SHOT):

Texto: "A oposição barrou os avanços da reforma tributária."
Output Esperado:
{
  "schemas": ["FORCE", "PATH"],
  "details": [
    {"schema_name": "FORCE", "sub_type": "BLOCKAGE", "anchor_word_pt": "barrou", "target_domain_pt": "Política"},
    {"schema_name": "PATH", "sub_type": "TRAJECTORY", "anchor_word_pt": "avanços", "target_domain_pt": "Economia"}
  ]
}

Texto: "A inflação sufocou o poder de compra e nos empurrou para a crise."
Output Esperado:
{
  "schemas": ["FORCE"],
  "details": [
    {"schema_name": "FORCE", "sub_type": "COMPULSION", "anchor_word_pt": "empurrou", "target_domain_pt": "Economia"}
  ]
}

Texto: "Quero saudar a presença do Vereador Júnior, de Rondon do Pará, que está nos visitando hoje no plenário."
Output Esperado:
{
  "schemas": [],
  "details": []
}

---

Analise o texto a seguir e retorne o JSON estrito conforme a especificação.

Texto para análise:
{text}