"""
Fase 3 - Entregable: comparacion de los tres metodos de busqueda por similitud.

Corre la misma consulta (vecinos de J01_P01_MARCADOR) con fuerza bruta
(Fase 2), FAISS y ChromaDB (Fase 3), y confirma que los tres devuelven
resultados identicos - la diferencia entre ellos es de arquitectura
(persistencia, interfaz, escalabilidad), no de precision, con este tamano
de corpus.
"""

import json
import sys
from pathlib import Path

import numpy as np

from embeddings import EMBEDDINGS_NPY, IDS_JSON, top_k_similares
from vector_store_chroma import buscar as buscar_chroma
from vector_store_chroma import cargar_textos, construir_coleccion
from vector_store_faiss import buscar as buscar_faiss
from vector_store_faiss import construir_indice

OUTPUT_JSON = Path(__file__).parent.parent / "data" / "processed" / "vector_store_comparacion.json"

PREGUNTA_ANCLA = "J01_P01_MARCADOR"
K = 5


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids: list[str] = json.load(f)
    with open(Path(__file__).parent.parent / "data" / "processed" / "corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)

    id_a_texto = {}
    for registro in corpus:
        id_a_texto.setdefault(registro["pregunta_id"], registro["pregunta"])

    idx_ancla = ids.index(PREGUNTA_ANCLA)

    # Fuerza bruta (Fase 2)
    resultado_fuerza_bruta = [
        (ids[i], sim) for i, sim in top_k_similares(idx_ancla, embeddings, k=K)
    ]

    # FAISS (Fase 3)
    indice_faiss = construir_indice(embeddings)
    resultado_faiss = buscar_faiss(indice_faiss, ids, embeddings[idx_ancla], k=K, excluir_id=PREGUNTA_ANCLA)

    # ChromaDB (Fase 3)
    preguntas = cargar_textos(ids)
    coleccion_chroma = construir_coleccion(ids, preguntas, embeddings)
    resultado_chroma = buscar_chroma(coleccion_chroma, embeddings[idx_ancla], k=K, excluir_id=PREGUNTA_ANCLA)

    print(f'Pregunta ancla: "{id_a_texto[PREGUNTA_ANCLA]}" ({PREGUNTA_ANCLA})\n')
    print(f"{'Fuerza bruta':<28}{'FAISS':<28}{'ChromaDB':<28}")
    for fb, fa, ch in zip(resultado_fuerza_bruta, resultado_faiss, resultado_chroma):
        print(f"{fb[1]:.4f} {fb[0]:<20}{fa[1]:.4f} {fa[0]:<20}{ch[1]:.4f} {ch[0]:<20}")

    ids_iguales = (
        [pid for pid, _ in resultado_fuerza_bruta]
        == [pid for pid, _ in resultado_faiss]
        == [pid for pid, _ in resultado_chroma]
    )
    print(f"\nLos tres metodos devuelven el mismo orden de resultados: {ids_iguales}")

    resultado = {
        "pregunta_ancla": PREGUNTA_ANCLA,
        "k": K,
        "fuerza_bruta": [{"pregunta_id": pid, "similitud": round(sim, 4)} for pid, sim in resultado_fuerza_bruta],
        "faiss": [{"pregunta_id": pid, "similitud": round(sim, 4)} for pid, sim in resultado_faiss],
        "chromadb": [{"pregunta_id": pid, "similitud": round(sim, 4)} for pid, sim in resultado_chroma],
        "coinciden": ids_iguales,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"\nSalida guardada en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
