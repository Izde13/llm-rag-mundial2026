"""
Pruebas de la Fase 2.

Verifican las propiedades matematicas de similitud_coseno y top_k_similares
con vectores sinteticos (no cargan el modelo real, para que las pruebas sean
rapidas y deterministas).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import similitud_coseno, top_k_similares  # noqa: E402


def test_similitud_coseno_vector_consigo_mismo_es_uno():
    vector = np.array([0.3, -0.5, 0.8])
    assert np.isclose(similitud_coseno(vector, vector), 1.0)


def test_similitud_coseno_vectores_ortogonales_es_cero():
    vector_a = np.array([1.0, 0.0])
    vector_b = np.array([0.0, 1.0])
    assert np.isclose(similitud_coseno(vector_a, vector_b), 0.0)


def test_similitud_coseno_vectores_opuestos_es_menos_uno():
    vector_a = np.array([1.0, 2.0])
    vector_b = np.array([-1.0, -2.0])
    assert np.isclose(similitud_coseno(vector_a, vector_b), -1.0)


def test_similitud_coseno_es_invariante_a_la_magnitud():
    vector_a = np.array([1.0, 1.0])
    vector_b = np.array([2.0, 2.0])
    assert np.isclose(similitud_coseno(vector_a, vector_b), 1.0)


def test_top_k_similares_excluye_la_pregunta_objetivo():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    resultado = top_k_similares(idx_pregunta=0, embeddings=embeddings, k=3)
    indices_devueltos = [i for i, _ in resultado]
    assert 0 not in indices_devueltos


def test_top_k_similares_ordena_de_mayor_a_menor():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    resultado = top_k_similares(idx_pregunta=0, embeddings=embeddings, k=3)
    similitudes = [sim for _, sim in resultado]
    assert similitudes == sorted(similitudes, reverse=True)


def test_top_k_similares_encuentra_el_vector_mas_parecido():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    resultado = top_k_similares(idx_pregunta=0, embeddings=embeddings, k=1)
    indice_mas_similar, _ = resultado[0]
    assert indice_mas_similar == 1
