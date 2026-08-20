"""
Fase 2 - Entregable: comparacion de embeddings sobre preguntas del corpus.

Reutiliza los embeddings ya generados por embeddings.py (no los recalcula) y
muestra, para un par de preguntas ancla, cuales son las mas parecidas
semanticamente segun similitud coseno.

Las preguntas se deduplican por pregunta_id (ver embeddings.cargar_preguntas)
porque el corpus tiene una fila por cada (pregunta, participante) - comparar
sobre las 6582 filas crudas compara texto identico consigo mismo miles de
veces sin aportar nada.
"""

import json
import sys
from pathlib import Path

import numpy as np

from embeddings import EMBEDDINGS_NPY, IDS_JSON, top_k_similares

OUTPUT_JSON = Path(__file__).parent.parent / "data" / "processed" / "embeddings_comparacion.json"

PREGUNTAS_ANCLA = ["J01_P01_MARCADOR", "J01_P01_RESULTADO"]
K = 5


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    embeddings = np.load(EMBEDDINGS_NPY)
    with open(IDS_JSON, encoding="utf-8") as f:
        ids = json.load(f)
    with open(Path(__file__).parent.parent / "data" / "processed" / "corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)

    id_a_texto = {}
    for registro in corpus:
        id_a_texto.setdefault(registro["pregunta_id"], registro["pregunta"])

    resultado = {"total_preguntas_unicas": len(ids), "k": K, "ejemplos": []}

    for pregunta_id in PREGUNTAS_ANCLA:
        idx_objetivo = ids.index(pregunta_id)
        texto_objetivo = id_a_texto[pregunta_id]

        vecinos = [
            {"pregunta_id": ids[i], "pregunta": id_a_texto[ids[i]], "similitud": round(sim, 4)}
            for i, sim in top_k_similares(idx_objetivo, embeddings, k=K)
        ]

        resultado["ejemplos"].append(
            {"pregunta_id": pregunta_id, "pregunta": texto_objetivo, "top_k_similares": vecinos}
        )

        print(f'Pregunta ancla: "{texto_objetivo}" ({pregunta_id})')
        for vecino in vecinos:
            print(f"  {vecino['similitud']:.4f}  |  \"{vecino['pregunta']}\"")
        print()

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"Salida guardada en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
