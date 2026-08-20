"""
Pruebas de la Fase 3 (FAISS).

Vectores sinteticos para propiedades basicas del indice, y una prueba con los
embeddings reales de Fase 2 que confirma que IndexFlatIP (normalizado)
devuelve exactamente los mismos resultados que top_k_similares por fuerza
bruta - la equivalencia matematica que motiva usar IP en vez de L2 aqui.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import EMBEDDINGS_NPY, IDS_JSON, top_k_similares  # noqa: E402
from vector_store_faiss import buscar, construir_indice, normalizar  # noqa: E402


def test_normalizar_produce_vectores_de_norma_uno():
    vectores = np.array([[3.0, 4.0], [1.0, 0.0]])
    normalizados = normalizar(vectores)
    normas = np.linalg.norm(normalizados, axis=1)
    assert np.allclose(normas, 1.0)


def test_buscar_excluye_el_id_indicado():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    ids = ["a", "b", "c", "d"]
    index = construir_indice(embeddings)

    resultado = buscar(index, ids, embeddings[0], k=3, excluir_id="a")
    ids_devueltos = [pid for pid, _ in resultado]

    assert "a" not in ids_devueltos
    assert len(ids_devueltos) == 3


def test_buscar_ordena_de_mayor_a_menor():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    ids = ["a", "b", "c", "d"]
    index = construir_indice(embeddings)

    resultado = buscar(index, ids, embeddings[0], k=3, excluir_id="a")
    scores = [score for _, score in resultado]

    assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(not EMBEDDINGS_NPY.exists(), reason="requiere embeddings de Fase 2 generados")
def test_faiss_coincide_con_fuerza_bruta_sobre_embeddings_reales():
    import json

    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids = json.load(f)

    idx_ancla = ids.index("J01_P01_MARCADOR")
    index = construir_indice(embeddings)

    resultado_faiss = buscar(index, ids, embeddings[idx_ancla], k=5, excluir_id="J01_P01_MARCADOR")
    resultado_fuerza_bruta = top_k_similares(idx_ancla, embeddings, k=5)

    ids_faiss = [pid for pid, _ in resultado_faiss]
    ids_fuerza_bruta = [ids[i] for i, _ in resultado_fuerza_bruta]
    assert ids_faiss == ids_fuerza_bruta

    scores_faiss = [score for _, score in resultado_faiss]
    scores_fuerza_bruta = [sim for _, sim in resultado_fuerza_bruta]
    assert np.allclose(scores_faiss, scores_fuerza_bruta, atol=1e-4)
