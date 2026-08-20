"""
Pruebas de la Fase 5 (RAG): retrieval + construccion de prompt.

Las funciones puras (filtros de metadata, armado de prompt, corte de
generacion) se prueban sin depender de Chroma ni del modelo entrenado.
Las que si dependen de esos artefactos (coleccion persistida, corpus real)
usan datos reales del corpus con skipif, igual que test_vector_store_chroma.py
- no se mockea el corpus para no divergir de la forma real de los datos.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bpe_vocab import TOKEN_UNK, cargar_merges, construir_vocab_ids  # noqa: E402
from rag import (  # noqa: E402
    CORPUS_JSON,
    abrir_coleccion,
    cargar_respuestas,
    construir_prompt,
    construir_where,
    extraer_filtro_jornada,
    extraer_filtro_partido,
    generar_con_corte,
    recuperar_contexto,
)
from transformer_model import TransformerLM  # noqa: E402
from vector_store_chroma import CHROMA_DIR  # noqa: E402


# --- extraer_filtro_jornada / extraer_filtro_partido / construir_where ---


def test_extraer_filtro_jornada_detecta_numero_y_lo_normaliza():
    assert extraer_filtro_jornada("¿Quién gana en la Jornada 3?") == {"jornada": "J03"}
    assert extraer_filtro_jornada("resultados jornada 12") == {"jornada": "J12"}


def test_extraer_filtro_jornada_es_insensible_a_mayusculas():
    assert extraer_filtro_jornada("JORNADA 1") == {"jornada": "J01"}


def test_extraer_filtro_jornada_sin_mencion_devuelve_none():
    assert extraer_filtro_jornada("¿Quién gana Partido 1?") is None


def test_extraer_filtro_partido_detecta_numero_y_lo_normaliza():
    assert extraer_filtro_partido("¿Quién gana Partido 3?") == {"partido": "P03"}
    assert extraer_filtro_partido("partido 11") == {"partido": "P11"}


def test_extraer_filtro_partido_sin_mencion_devuelve_none():
    assert extraer_filtro_partido("¿Quién gana la Jornada 1?") is None


def test_construir_where_sin_filtros_devuelve_none():
    assert construir_where("¿Cómo va el mundial?") is None


def test_construir_where_un_solo_filtro_no_usa_and():
    assert construir_where("¿Quién gana Partido 3?") == {"partido": "P03"}


def test_construir_where_combina_jornada_y_partido_con_and():
    resultado = construir_where("Jornada 2 Partido 5")
    assert resultado == {"$and": [{"jornada": "J02"}, {"partido": "P05"}]}


# --- construir_prompt ---


def test_construir_prompt_concatena_contexto_y_query_en_lineas():
    contexto = [
        ("¿Quién gana Partido 1?", "México"),
        ("Marcador Partido 1", "2-0"),
    ]
    prompt = construir_prompt("¿Quién gana Partido 2?", contexto)

    lineas = prompt.split("\n")
    assert lineas == [
        "¿Quién gana Partido 1? México",
        "Marcador Partido 1 2-0",
        "¿Quién gana Partido 2?",
    ]


def test_construir_prompt_con_contexto_vacio_solo_deja_la_query():
    assert construir_prompt("¿Quién gana Partido 2?", []) == "¿Quién gana Partido 2?"


# --- cargar_respuestas (usa el corpus.json real) ---


@pytest.mark.skipif(not CORPUS_JSON.exists(), reason="requiere corpus.json generado")
def test_cargar_respuestas_devuelve_resultado_oficial_por_pregunta_id():
    respuestas = cargar_respuestas(["J01_P01_MARCADOR"])
    assert respuestas == {"J01_P01_MARCADOR": "2-0"}


@pytest.mark.skipif(not CORPUS_JSON.exists(), reason="requiere corpus.json generado")
def test_cargar_respuestas_preserva_el_orden_de_los_ids_pedidos():
    ids = ["J01_P01_MARCADOR", "J01_P01_MARCADOR"]
    respuestas = cargar_respuestas(ids)
    assert list(respuestas.keys()) == ["J01_P01_MARCADOR"]


# --- recuperar_contexto (requiere la coleccion Chroma persistida de Fase 3) ---


@pytest.mark.skipif(not CHROMA_DIR.exists(), reason="requiere chroma_db de Fase 3 generado")
def test_recuperar_contexto_devuelve_k_pares_pregunta_respuesta_reales():
    from embeddings import EMBEDDINGS_NPY, IDS_JSON

    if not EMBEDDINGS_NPY.exists() or not IDS_JSON.exists():
        pytest.skip("requiere embeddings de Fase 2 generados")

    import json

    import numpy as np

    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids = json.load(f)
    idx_ancla = ids.index("J01_P01_MARCADOR")

    coleccion = abrir_coleccion()
    contexto = recuperar_contexto(coleccion, embeddings[idx_ancla], k=3)

    assert len(contexto) == 3
    for pregunta, respuesta in contexto:
        assert isinstance(pregunta, str) and pregunta
        assert isinstance(respuesta, str) and respuesta


@pytest.mark.skipif(not CHROMA_DIR.exists(), reason="requiere chroma_db de Fase 3 generado")
def test_recuperar_contexto_respeta_el_filtro_where():
    from embeddings import EMBEDDINGS_NPY, IDS_JSON

    if not EMBEDDINGS_NPY.exists() or not IDS_JSON.exists():
        pytest.skip("requiere embeddings de Fase 2 generados")

    import json

    import numpy as np

    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids = json.load(f)
    idx_ancla = ids.index("J03_P01_AMARILLA")

    coleccion = abrir_coleccion()
    contexto = recuperar_contexto(coleccion, embeddings[idx_ancla], k=5, where={"jornada": "J03"})

    preguntas_j03 = {p for p in cargar_textos_de_jornada("J03")}
    assert len(contexto) == 5
    for pregunta, _ in contexto:
        assert pregunta in preguntas_j03


def cargar_textos_de_jornada(jornada: str) -> set[str]:
    import json

    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)
    return {r["pregunta"] for r in corpus if r["pregunta_id"].startswith(jornada)}


# --- generar_con_corte (modelo TransformerLM chico, sin pesos entrenados) ---


def _armar_vocab_y_merges_de_prueba():
    merges = cargar_merges()
    textos = ["¿Quién gana Partido 1? México", "Marcador Partido 1 2-0"]
    simbolo_a_id, id_a_simbolo = construir_vocab_ids(merges, textos)
    return merges, simbolo_a_id, id_a_simbolo


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath(
        "data", "processed", "tokenizer_bpe_stats.json"
    ).exists(),
    reason="requiere tokenizer_bpe_stats.json de Fase 1 generado",
)
def test_generar_con_corte_para_al_encontrar_simbolo_con_interrogacion():
    import torch

    merges, simbolo_a_id, id_a_simbolo = _armar_vocab_y_merges_de_prueba()
    vocab_size = len(simbolo_a_id)

    id_interrogacion = next(
        idx for simbolo, idx in simbolo_a_id.items() if "?" in simbolo
    )

    modelo = TransformerLM(
        vocab_size=vocab_size, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_seq_len=32
    )

    # Se fuerza (sin entrenar) a que el modelo siempre prediga el simbolo "?"
    # para verificar que generar_con_corte corta ahi, sin agregarlo al texto,
    # en vez de seguir generando hasta max_nuevos_tokens.
    with torch.no_grad():
        modelo.cabeza_salida.weight.zero_()
        modelo.cabeza_salida.bias.zero_()
        modelo.cabeza_salida.bias[id_interrogacion] = 100.0

    resultado = generar_con_corte(
        modelo, "Marcador Partido 1", merges, simbolo_a_id, id_a_simbolo,
        max_seq_len=32, max_nuevos_tokens=20,
    )

    assert "?" not in resultado


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath(
        "data", "processed", "tokenizer_bpe_stats.json"
    ).exists(),
    reason="requiere tokenizer_bpe_stats.json de Fase 1 generado",
)
def test_generar_con_corte_para_al_encontrar_unk():
    import torch

    merges, simbolo_a_id, id_a_simbolo = _armar_vocab_y_merges_de_prueba()
    vocab_size = len(simbolo_a_id)
    id_unk = simbolo_a_id[TOKEN_UNK]

    modelo = TransformerLM(
        vocab_size=vocab_size, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_seq_len=32
    )

    with torch.no_grad():
        modelo.cabeza_salida.weight.zero_()
        modelo.cabeza_salida.bias.zero_()
        modelo.cabeza_salida.bias[id_unk] = 100.0

    resultado = generar_con_corte(
        modelo, "Marcador Partido 1", merges, simbolo_a_id, id_a_simbolo,
        max_seq_len=32, max_nuevos_tokens=20,
    )

    # El corte ocurre antes de agregar <UNK> a la secuencia generada; el
    # texto resultante es equivalente a decodificar solo el prompt original.
    from bpe_vocab import codificar_ids, decodificar_ids

    esperado = decodificar_ids(codificar_ids("Marcador Partido 1", merges, simbolo_a_id), id_a_simbolo)
    assert resultado == esperado
