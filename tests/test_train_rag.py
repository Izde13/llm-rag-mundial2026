"""
Pruebas de la Fase 5 (reentrenamiento para RAG).

`generar_ejemplos_rag` depende de la coleccion Chroma persistida (retrieval
real) - se prueba con datos reales del corpus (skipif si faltan, igual que
el resto de la suite). `cargar_preguntas_unicas` es pura y se prueba con un
corpus sintetico chico para no depender de artefactos generados.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import CORPUS_JSON  # noqa: E402
from train_rag import cargar_preguntas_unicas, generar_ejemplos_rag  # noqa: E402
from vector_store_chroma import CHROMA_DIR  # noqa: E402

CORPUS_SINTETICO = [
    {"pregunta_id": "J01_P01_MARCADOR", "pregunta": "Marcador Partido 1", "resultado_oficial": "2-0"},
    {"pregunta_id": "J01_P01_MARCADOR", "pregunta": "Marcador Partido 1 (dup)", "resultado_oficial": "2-0"},
    {"pregunta_id": "J01_P02_RESULTADO", "pregunta": "¿Quién gana Partido 2?", "resultado_oficial": "México"},
]


def test_cargar_preguntas_unicas_deduplica_por_pregunta_id_quedandose_con_la_primera():
    ids, preguntas = cargar_preguntas_unicas(CORPUS_SINTETICO)

    assert ids == ["J01_P01_MARCADOR", "J01_P02_RESULTADO"]
    assert preguntas == ["Marcador Partido 1", "¿Quién gana Partido 2?"]


def test_cargar_preguntas_unicas_con_corpus_vacio_devuelve_listas_vacias():
    assert cargar_preguntas_unicas([]) == ([], [])


@pytest.mark.skipif(not CHROMA_DIR.exists(), reason="requiere chroma_db de Fase 3 generado")
@pytest.mark.skipif(not CORPUS_JSON.exists(), reason="requiere corpus.json generado")
def test_generar_ejemplos_rag_agrupa_variantes_por_pregunta_y_todas_terminan_en_la_respuesta():
    import json

    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)
    ids, _ = cargar_preguntas_unicas(corpus)
    total_preguntas = len(ids)

    ks = (2,)
    variantes_por_k = 1
    ejemplos_por_pregunta = generar_ejemplos_rag(ks=ks, variantes_por_k=variantes_por_k)

    assert len(ejemplos_por_pregunta) == total_preguntas
    for variantes in ejemplos_por_pregunta:
        assert len(variantes) == len(ks) * variantes_por_k


@pytest.mark.skipif(not CHROMA_DIR.exists(), reason="requiere chroma_db de Fase 3 generado")
@pytest.mark.skipif(not CORPUS_JSON.exists(), reason="requiere corpus.json generado")
def test_generar_ejemplos_rag_cada_variante_termina_en_la_respuesta_real_de_su_pregunta():
    import json

    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)
    ids, _ = cargar_preguntas_unicas(corpus)
    id_a_respuesta = {}
    for registro in corpus:
        id_a_respuesta.setdefault(registro["pregunta_id"], registro["resultado_oficial"])

    ejemplos_por_pregunta = generar_ejemplos_rag(ks=(2,), variantes_por_k=1)

    for pregunta_id, variantes in zip(ids, ejemplos_por_pregunta):
        respuesta_esperada = id_a_respuesta[pregunta_id]
        for texto in variantes:
            assert texto.rstrip().endswith(respuesta_esperada)
