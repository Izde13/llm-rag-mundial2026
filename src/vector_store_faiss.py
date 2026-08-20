"""
Fase 3 - Vector Database: indexar embeddings para busqueda eficiente.

FAISS (Facebook AI Similarity Search) envuelve una matriz de vectores en un
"indice" con un metodo .search(query, k). Se usa IndexFlatIP (inner product)
en vez de IndexFlatL2 (distancia euclidiana) porque los embeddings de
MiniLM se comparan por similitud coseno, y coseno == producto punto solo si
los vectores estan normalizados a norma 1 primero.

Con 442 vectores, IndexFlatIP sigue siendo busqueda exacta por fuerza bruta
(igual que top_k_similares de Fase 2), pero implementada en C++ vectorizado.
El valor de esta fase es la interfaz (crear indice -> anadir vectores ->
buscar), antes de pasar a un indice aproximado (ANN) que si lo necesitaria
un corpus mucho mas grande.
"""

import json
from pathlib import Path

import faiss
import numpy as np

EMBEDDINGS_NPY = Path(__file__).parent.parent / "data" / "processed" / "embeddings.npy"
IDS_JSON = Path(__file__).parent.parent / "data" / "processed" / "embeddings_ids.json"


def normalizar(vectores: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    return vectores / normas


def construir_indice(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalizar(embeddings))
    return index


def buscar(
    index: faiss.IndexFlatIP,
    ids: list[str],
    vector_query: np.ndarray,
    k: int = 5,
    excluir_id: str | None = None,
) -> list[tuple[str, float]]:
    query_normalizado = normalizar(vector_query.reshape(1, -1))
    scores, indices = index.search(query_normalizado, k + 1)

    resultados = []
    for idx, score in zip(indices[0], scores[0]):
        pregunta_id = ids[int(idx)]
        if pregunta_id == excluir_id:
            continue
        if len(resultados) == k:
            break
        resultados.append((pregunta_id, float(score)))
    return resultados


def main() -> None:
    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids: list[str] = json.load(f)

    index = construir_indice(embeddings)
    print(f"Indice FAISS construido: {index.ntotal} vectores, dimension {embeddings.shape[1]}")

    idx_ancla = ids.index("J01_P01_MARCADOR")
    vector_ancla = embeddings[idx_ancla]

    resultados = buscar(index, ids, vector_ancla, k=5, excluir_id="J01_P01_MARCADOR")

    print("\nTop-5 similares a J01_P01_MARCADOR (FAISS, IndexFlatIP):")
    for pregunta_id, score in resultados:
        print(f"{score:.4f}  {pregunta_id}")


if __name__ == "__main__":
    main()
