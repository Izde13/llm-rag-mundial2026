"""
Fase 4 — Vocabulario BPE -> IDs enteros.

tokenizer_bpe.py (Fase 1) tokeniza texto a una lista de simbolos-string
(ej. "Marcador</w>"), pero nn.Embedding de PyTorch necesita ids enteros:
es una tabla [vocab_size, dim] donde la fila i es el vector del token i.
Este modulo agrega esa capa de mapeo string <-> id que faltaba, mas dos
tokens especiales:

- <PAD> (id 0): relleno para igualar la longitud de secuencias dentro de
  un mismo batch (el loop de entrenamiento procesa varias preguntas a la
  vez, todas del mismo largo fijo).
- <UNK> (id 1): simbolo no visto durante el entrenamiento del vocabulario.
  No deberia ocurrir con texto del propio corpus (BPE cae a caracteres
  sueltos en el peor caso), pero es la salvaguarda estandar de la
  industria para texto nuevo.

Los merges se cargan desde data/processed/tokenizer_bpe_stats.json (Fase 1)
en vez de reentrenar BPE aqui.
"""

import json
import sys
from pathlib import Path

from tokenizer_bpe import FIN_PALABRA, tokenizar_con_merges

BPE_STATS_JSON = Path(__file__).parent.parent / "data" / "processed" / "tokenizer_bpe_stats.json"

TOKEN_PAD = "<PAD>"
TOKEN_UNK = "<UNK>"


def cargar_merges(ruta: Path = BPE_STATS_JSON) -> list[tuple[str, str]]:
    with open(ruta, encoding="utf-8") as f:
        stats = json.load(f)
    return [tuple(par) for par in stats["merges"]]


def construir_vocab_ids(
    merges: list[tuple[str, str]], textos: list[str]
) -> tuple[dict[str, int], dict[int, str]]:
    """Recorre los simbolos que produce el BPE sobre `textos` y les asigna un id.

    Los tokens especiales van primero (ids fijos 0 y 1) para que no dependan
    de qué símbolos trajo el corpus.
    """
    simbolos: set[str] = set()
    for texto in textos:
        simbolos.update(tokenizar_con_merges(texto, merges))

    simbolo_a_id = {TOKEN_PAD: 0, TOKEN_UNK: 1}
    for simbolo in sorted(simbolos):
        simbolo_a_id[simbolo] = len(simbolo_a_id)

    id_a_simbolo = {idx: simbolo for simbolo, idx in simbolo_a_id.items()}
    return simbolo_a_id, id_a_simbolo


def codificar_ids(texto: str, merges: list[tuple[str, str]], simbolo_a_id: dict[str, int]) -> list[int]:
    simbolos = tokenizar_con_merges(texto, merges)
    return [simbolo_a_id.get(simbolo, simbolo_a_id[TOKEN_UNK]) for simbolo in simbolos]


def decodificar_ids(ids: list[int], id_a_simbolo: dict[int, str]) -> str:
    simbolos = [id_a_simbolo[idx] for idx in ids if id_a_simbolo[idx] not in (TOKEN_PAD,)]
    texto = "".join(simbolos)
    return texto.replace(FIN_PALABRA, " ").strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    corpus_json = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
    with open(corpus_json, encoding="utf-8") as f:
        corpus = json.load(f)
    preguntas = [registro["pregunta"] for registro in corpus]

    merges = cargar_merges()
    simbolo_a_id, id_a_simbolo = construir_vocab_ids(merges, preguntas)

    print(f"Tamano vocabulario (con especiales): {len(simbolo_a_id)}")

    ejemplo = preguntas[0]
    ids = codificar_ids(ejemplo, merges, simbolo_a_id)
    recuperado = decodificar_ids(ids, id_a_simbolo)

    print(f"\nEjemplo:")
    print(f"  texto:        {ejemplo}")
    print(f"  ids:          {ids}")
    print(f"  decodificado: {recuperado}")


if __name__ == "__main__":
    main()
