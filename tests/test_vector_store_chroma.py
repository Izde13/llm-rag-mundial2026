"""
Pruebas de la Fase 3 (ChromaDB).

Confirma que la coleccion persistida devuelve los mismos vecinos y scores
que top_k_similares por fuerza bruta (Fase 2) - la misma equivalencia que ya
se verifico para FAISS, ahora sobre la capa de persistencia de Chroma.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import EMBEDDINGS_NPY, IDS_JSON, top_k_similares  # noqa: E402
from vector_store_chroma import (  # noqa: E402
    buscar,
    cargar_textos,
    construir_coleccion,
    extraer_metadata,
)


def test_extraer_metadata_separa_jornada_y_tipo():
    assert extraer_metadata("J03_P05_MARCADOR") == {"jornada": "J03", "tipo_pregunta": "MARCADOR"}
    assert extraer_metadata("J06_D01_PENALES") == {"jornada": "J06", "tipo_pregunta": "PENALES"}


@pytest.mark.skipif(not EMBEDDINGS_NPY.exists(), reason="requiere embeddings de Fase 2 generados")
def test_chroma_coincide_con_fuerza_bruta_sobre_embeddings_reales():
    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids = json.load(f)
    preguntas = cargar_textos(ids)

    idx_ancla = ids.index("J01_P01_MARCADOR")
    coleccion = construir_coleccion(ids, preguntas, embeddings)

    resultado_chroma = buscar(coleccion, embeddings[idx_ancla], k=5, excluir_id="J01_P01_MARCADOR")
    resultado_fuerza_bruta = top_k_similares(idx_ancla, embeddings, k=5)

    ids_chroma = [pid for pid, _ in resultado_chroma]
    ids_fuerza_bruta = [ids[i] for i, _ in resultado_fuerza_bruta]
    assert ids_chroma == ids_fuerza_bruta

    scores_chroma = [score for _, score in resultado_chroma]
    scores_fuerza_bruta = [sim for _, sim in resultado_fuerza_bruta]
    assert np.allclose(scores_chroma, scores_fuerza_bruta, atol=1e-3)


@pytest.mark.skipif(not EMBEDDINGS_NPY.exists(), reason="requiere embeddings de Fase 2 generados")
def test_buscar_con_where_solo_devuelve_resultados_de_la_jornada_filtrada():
    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids = json.load(f)
    preguntas = cargar_textos(ids)

    idx_ancla = ids.index("J03_P01_AMARILLA")
    coleccion = construir_coleccion(ids, preguntas, embeddings)

    resultado = buscar(
        coleccion,
        embeddings[idx_ancla],
        k=5,
        excluir_id="J03_P01_AMARILLA",
        where={"jornada": "J03"},
    )

    assert len(resultado) == 5
    assert all(pid.startswith("J03_") for pid, _ in resultado)
