#!/usr/bin/env python3
import requests
import json
import time
import os
from pathlib import Path
import sys
from datetime import datetime

class CamaraDataCollector:
    def __init__(self):
        self.headers = {"Accept": "application/json"}

    def obter_deputados(self, limite: int = 50, partido: str = None, uf: str = None, pagina: int = 1) -> list:
        """Obtém a lista de deputados com filtros opcionais."""
        url = "https://dadosabertos.camara.leg.br/api/v2/deputados"
        params = {"itens": limite, "ordenarPor": "nome", "pagina": pagina}
        if partido:
            params["siglaPartido"] = partido.upper()
        if uf:
            params["siglaUf"] = uf.upper()
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=15)
            if response.status_code == 200:
                return response.json().get("dados", [])
            else:
                print(f"[X] Erro ao obter deputados: {response.status_code}")
                return []
        except Exception as e:
            print(f"[X] Erro de conexão ao obter deputados: {e}")
            return []

    def coletar_discursos(self, data_inicio: str, data_fim: str,
                          max_deputados: int = 20,
                          partido: str = None,
                          uf: str = None,
                          pagina: int = 1) -> list:
        """
        Coleta discursos de deputados ativos em um intervalo de datas.
        Filtra por partido e/ou UF se informados.
        Usa pagina para pular blocos de deputados já coletados.
        """
        deputados = self.obter_deputados(limite=max_deputados, partido=partido, uf=uf, pagina=pagina)
        if not deputados:
            print("[!] Nenhum deputado encontrado com os filtros aplicados.")
            return []

        filtro_desc = " | ".join(filter(None, [
            f"partido={partido.upper()}" if partido else None,
            f"UF={uf.upper()}" if uf else None,
            f"página={pagina}",
        ]))
        print(f"[+] {len(deputados)} deputados ({filtro_desc}). Iniciando coleta de discursos...")
        discursos_coletados = []

        for idx, dep in enumerate(deputados):
            dep_id = dep.get("id")
            dep_nome = dep.get("nome")
            dep_partido = dep.get("siglaPartido")
            dep_uf = dep.get("siglaUf")

            print(f"[*] [{idx+1}/{len(deputados)}] {dep_nome} ({dep_partido}-{dep_uf})...")

            url_discursos = f"https://dadosabertos.camara.leg.br/api/v2/deputados/{dep_id}/discursos"
            params = {
                "dataInicio": data_inicio,
                "dataFim": data_fim,
                "ordenarPor": "dataHoraInicio",
                "ordem": "DESC"
            }

            try:
                response = requests.get(url_discursos, params=params, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    dados = response.json().get("dados", [])
                    validos = 0
                    for item in dados:
                        texto_bruto = item.get("transcricao") or item.get("sumario") or item.get("texto")
                        if not texto_bruto or len(texto_bruto.strip()) < 300:
                            continue
                        discurso_limpo = {
                            "id_interno": f"CAMARA_{item.get('id') or dep_id}",
                            "data_hora": item.get("dataHoraInicio"),
                            "orador_nome": dep_nome,
                            "orador_partido": dep_partido,
                            "orador_uf": dep_uf,
                            "texto": self._limpar_texto(texto_bruto)
                        }
                        discursos_coletados.append(discurso_limpo)
                        validos += 1
                    print(f"    [+] {len(dados)} discursos encontrados, {validos} válidos (>300 chars).")
                else:
                    print(f"    [X] Erro {response.status_code} para deputado {dep_id}.")
            except Exception as e:
                print(f"    [X] Erro de conexão para {dep_nome}: {e}")

            time.sleep(1)

        print(f"[√] Coleta finalizada. Total de discursos válidos: {len(discursos_coletados)}")
        return discursos_coletados

    def _limpar_texto(self, texto: str) -> str:
        if not texto:
            return ""
        return " ".join(texto.split())

    def salvar_dados(self, dados: list, nome_arquivo: str = "corpus_camara_piloto.jsonl", append: bool = False):
        """Salva em JSON Lines. Se append=True, adiciona ao arquivo existente."""
        mode = "a" if append else "w"
        with open(nome_arquivo, mode, encoding="utf-8") as f:
            for item in dados:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        acao = "adicionados a" if append else "salvos em"
        print(f"[√] {len(dados)} registros {acao}: {nome_arquivo}")


def uso():
    print("Uso:")
    print("  python coletor.py                          # 20 deputados, ordem alfabética, página 1")
    print("  python coletor.py --partido PT             # filtra por partido")
    print("  python coletor.py --uf SP                  # filtra por estado")
    print("  python coletor.py --pagina 2               # pula os primeiros 20, pega próximos 20")
    print("  python coletor.py --max 10 --partido PL    # 10 deputados do PL")
    print("  python coletor.py --append                 # adiciona ao corpus existente (não sobrescreve)")
    sys.exit(0)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--help" in args:
        uso()

    DATA_INICIO = "2025-01-01"
    DATA_FIM    = "2025-12-31"
    MAX         = 20
    PARTIDO     = None
    UF          = None
    PAGINA      = 1
    APPEND      = "--append" in args

    try:
        if "--max" in args:
            MAX = int(args[args.index("--max") + 1])
        if "--partido" in args:
            PARTIDO = args[args.index("--partido") + 1]
        if "--uf" in args:
            UF = args[args.index("--uf") + 1]
        if "--pagina" in args:
            PAGINA = int(args[args.index("--pagina") + 1])
    except (IndexError, ValueError):
        uso()

    collector = CamaraDataCollector()
    dataset = collector.coletar_discursos(
        data_inicio=DATA_INICIO,
        data_fim=DATA_FIM,
        max_deputados=MAX,
        partido=PARTIDO,
        uf=UF,
        pagina=PAGINA,
    )

    if dataset:
        collector.salvar_dados(dataset, str(Path(__file__).parent / "data" / "corpus_camara_piloto.jsonl"), append=APPEND)
        print("\nPrévia:")
        for doc in dataset[:2]:
            print(f"  {doc['orador_nome']} ({doc['orador_partido']}-{doc['orador_uf']}): {doc['texto'][:120]}...")
    else:
        print("[!] Nenhum discurso coletado.")
