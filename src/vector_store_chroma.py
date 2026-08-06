"""
Fase 3 - Vector Database: ChromaDB, persistencia local practica.

A diferencia de FAISS (solo guarda vectores, requiere mapear ids a mano),
ChromaDB guarda vector + texto + id + metadata juntos en una coleccion
persistida en disco (data/processed/chroma_db/). No genera embeddings
propios aqui: se le entregan los ya calculados en Fase 2 (embeddings.py),
para no recalcular ni desviarse del modelo ya validado.

La metadata (jornada, tipo_pregunta) permite combinar busqueda semantica
con filtros exactos en una sola query (parametro `where`) - algo que FAISS
puro no ofrece, y que en un RAG real es tan importante como la similitud
misma (ej. "lo mas parecido, pero solo de este mes/departamento/usuario").
"""

import json
from pathlib import Path

import chromadb
import numpy as np

from embeddings import EMBEDDINGS_NPY, IDS_JSON

CORPUS_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "processed" / "chroma_db"
NOMBRE_COLECCION = "preguntas_mundial"


def cargar_textos(ids: list[str]) -> list[str]:
    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)

    id_a_texto = {}
    for registro in corpus:
        id_a_texto.setdefault(registro["pregunta_id"], registro["pregunta"])

    return [id_a_texto[pregunta_id] for pregunta_id in ids]


def extraer_metadata(pregunta_id: str) -> dict[str, str]:
    partes = pregunta_id.split("_")
    return {"jornada": partes[0], "partido": partes[1], "tipo_pregunta": partes[-1]}


def construir_coleccion(
    ids: list[str], preguntas: list[str], embeddings: np.ndarray
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if NOMBRE_COLECCION in [c.name for c in client.list_collections()]:
        client.delete_collection(NOMBRE_COLECCION)

    metadatas = [extraer_metadata(pregunta_id) for pregunta_id in ids]

    coleccion = client.create_collection(NOMBRE_COLECCION)
    coleccion.add(ids=ids, embeddings=embeddings.tolist(), documents=preguntas, metadatas=metadatas)
    return coleccion


def buscar(
    coleccion: chromadb.Collection,
    vector_query: np.ndarray,
    k: int = 5,
    excluir_id: str | None = None,
    where: dict | None = None,
) -> list[tuple[str, float]]:
    resultado = coleccion.query(
        query_embeddings=[vector_query.tolist()], n_results=k + 1, where=where
    )

    resultados = []
    for pregunta_id, distancia in zip(resultado["ids"][0], resultado["distances"][0]):
        if pregunta_id == excluir_id:
            continue
        if len(resultados) == k:
            break
        # Chroma devuelve distancia (menor = mas parecido); se convierte a
        # similitud coseno (mayor = mas parecido) para comparar con Fase 2/3.
        similitud = 1 - distancia / 2
        resultados.append((pregunta_id, similitud))
    return resultados


def main() -> None:
    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids: list[str] = json.load(f)
    preguntas = cargar_textos(ids)

    coleccion = construir_coleccion(ids, preguntas, embeddings)
    print(f"Coleccion Chroma construida: {coleccion.count()} documentos, persistida en {CHROMA_DIR}")

    idx_ancla = ids.index("J01_P01_MARCADOR")
    resultados = buscar(coleccion, embeddings[idx_ancla], k=5, excluir_id="J01_P01_MARCADOR")

    print("\nTop-5 similares a J01_P01_MARCADOR (ChromaDB):")
    for pregunta_id, similitud in resultados:
        print(f"{similitud:.4f}  {pregunta_id}")

    # Ejemplo de filtro combinado: imposible con FAISS puro sin programarlo
    # a mano. "¿Quién recibe la primera tarjeta amarilla?" se repite
    # literalmente en muchas jornadas (mismo caso que J01_P01_RESULTADO en
    # Fase 2) - sin filtro, los vecinos mas cercanos son copias identicas de
    # otras jornadas. Con where={"jornada": "J03"} se acota la busqueda
    # semantica a preguntas de la misma jornada.
    idx_amarilla = ids.index("J03_P01_AMARILLA")
    print('\nVecinos de "J03_P01_AMARILLA" SIN filtro (puede traer otras jornadas):')
    for pregunta_id, similitud in buscar(coleccion, embeddings[idx_amarilla], k=3, excluir_id="J03_P01_AMARILLA"):
        print(f"{similitud:.4f}  {pregunta_id}")

    print('\nVecinos de "J03_P01_AMARILLA" CON filtro where={"jornada": "J03"}:')
    resultados_filtrados = buscar(
        coleccion,
        embeddings[idx_amarilla],
        k=3,
        excluir_id="J03_P01_AMARILLA",
        where={"jornada": "J03"},
    )
    for pregunta_id, similitud in resultados_filtrados:
        print(f"{similitud:.4f}  {pregunta_id}")


if __name__ == "__main__":
    main()
