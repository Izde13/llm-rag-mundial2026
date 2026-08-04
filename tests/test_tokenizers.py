"""
Pruebas de la Fase 1.

Verifican la propiedad no-negociable de cualquier tokenizador: debe ser
reversible (codificar y luego decodificar recupera el texto original), y
que BPE nunca pierda caracteres ni produzca tokens vacios al fusionar.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer_bpe import entrenar_bpe, tokenizar_con_merges  # noqa: E402
from tokenizer_char import codificar, construir_vocabulario, decodificar  # noqa: E402

TEXTOS_EJEMPLO = [
    "¿Quién gana Partido 1?",
    "Marcador Partido 1 — México vs Sudáfrica",
    "¿Cuántos córners tuvo Colombia en la Jornada 2?",
]


def test_char_level_es_reversible():
    char_a_id, id_a_char = construir_vocabulario(TEXTOS_EJEMPLO)

    for texto in TEXTOS_EJEMPLO:
        ids = codificar(texto, char_a_id)
        recuperado = decodificar(ids, id_a_char)
        assert recuperado == texto


def test_char_level_vocabulario_cubre_todos_los_caracteres():
    char_a_id, _ = construir_vocabulario(TEXTOS_EJEMPLO)

    todos_los_caracteres = set("".join(TEXTOS_EJEMPLO))
    assert todos_los_caracteres.issubset(char_a_id.keys())


def test_bpe_no_produce_tokens_vacios():
    _, merges = entrenar_bpe(TEXTOS_EJEMPLO, num_merges=50)

    for texto in TEXTOS_EJEMPLO:
        tokens = tokenizar_con_merges(texto, merges)
        assert all(token != "" for token in tokens)


def test_bpe_conserva_todos_los_caracteres_originales():
    _, merges = entrenar_bpe(TEXTOS_EJEMPLO, num_merges=50)

    for texto in TEXTOS_EJEMPLO:
        tokens = tokenizar_con_merges(texto, merges)
        reconstruido = "".join(tokens).replace("</w>", " ").strip()
        assert reconstruido == texto


def test_bpe_reduce_o_mantiene_longitud_frente_a_char_level():
    _, merges = entrenar_bpe(TEXTOS_EJEMPLO, num_merges=50)

    for texto in TEXTOS_EJEMPLO:
        num_tokens_bpe = len(tokenizar_con_merges(texto, merges))
        assert num_tokens_bpe <= len(texto)
