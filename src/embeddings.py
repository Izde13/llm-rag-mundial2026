"""
Fase 2 - Embeddings: de tokens a vectores semanticos.

Usa un modelo pre-entrenado liviano (all-MiniLM-L6-v2, via
sentence-transformers) para convertir cada pregunta del corpus en un vector
de 384 numeros. El modelo ya viene entrenado (no se entrena nada aqui) - solo
se consulta, igual que se usaria un diccionario ya escrito.

La similitud coseno se implementa a mano (no se usa la de la libreria) para
que la formula quede explicita.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
EMBEDDINGS_NPY = Path(__file__).parent.parent / "data" / "processed" / "embeddings.npy"
IDS_JSON = Path(__file__).parent.parent / "data" / "processed" / "embeddings_ids.json"

NOMBRE_MODELO = "all-MiniLM-L6-v2"


def cargar_preguntas(ruta_corpus: Path) -> tuple[list[str], list[str]]:
    with open(ruta_corpus, encoding="utf-8") as f:
        corpus = json.load(f)

    vistos: dict[str, str] = {}
    for registro in corpus:
        vistos.setdefault(registro["pregunta_id"], registro["pregunta"])

    ids = list(vistos.keys())
    preguntas = list(vistos.values())
    return ids, preguntas


def generar_embeddings(textos: list[str], modelo: SentenceTransformer) -> np.ndarray:
    return modelo.encode(textos, show_progress_bar=True)


def similitud_coseno(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    producto_punto = np.dot(vec_a, vec_b)
    norma_a = np.linalg.norm(vec_a)
    norma_b = np.linalg.norm(vec_b)
    return producto_punto / (norma_a * norma_b)


def top_k_similares(idx_pregunta: int, embeddings: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    vector_objetivo = embeddings[idx_pregunta]

    normas = np.linalg.norm(embeddings, axis=1)
    norma_objetivo = np.linalg.norm(vector_objetivo)
    productos_punto = embeddings @ vector_objetivo

    similitudes = productos_punto / (normas * norma_objetivo)
    similitudes[idx_pregunta] = -1

    indices_top_k = np.argsort(similitudes)[::-1][:k]
    return [(int(i), float(similitudes[i])) for i in indices_top_k]


def main() -> None:
    ids, preguntas = cargar_preguntas(CORPUS_JSON)

    modelo = SentenceTransformer(NOMBRE_MODELO)
    embeddings = generar_embeddings(preguntas, modelo)

    print(f"\nPreguntas embebidas: {len(preguntas)}")
    print(f"Dimension de cada vector: {embeddings.shape[1]}")

    EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY, embeddings)
    with open(IDS_JSON, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    print(f"\nEmbeddings guardados en: {EMBEDDINGS_NPY}")
    print(f"IDs guardados en: {IDS_JSON}")


if __name__ == "__main__":
    main()
