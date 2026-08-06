"""
Fase 5 - RAG completo: retrieval (Fase 2 + Fase 3) + generacion.

Pipeline: query -> embedding (MiniLM) -> busqueda en ChromaDB -> top-k
resultados reales del corpus -> prompt con esos datos inyectados -> el
transformer propio continua el texto.

Usa modelo_fase5.pt (train_rag.py), no modelo_fase4.pt: el modelo de Fase 4
nunca vio, en su entrenamiento, la forma "contexto + pregunta" que arma este
pipeline (solo vio oraciones sueltas del corpus) y por eso no se ancla de
forma confiable al dato inyectado. modelo_fase5.pt se reentreno desde cero
con ejemplos sinteticos de esa forma exacta (ver train_rag.py).

El LLM propio nunca "memoriza" las respuestas por fuerza bruta (864K
parametros no alcanzan para memorizar 442 preguntas). El buscador
(MiniLM+Chroma) encuentra el dato correcto; el LLM aprendio a leerlo del
contexto que se le da y repetirlo/parafrasearlo, en vez de inventarlo.
"""

import re
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from bpe_vocab import cargar_merges, codificar_ids, decodificar_ids
from embeddings import NOMBRE_MODELO
from generar import cargar_modelo
from vector_store_chroma import CHROMA_DIR, NOMBRE_COLECCION, cargar_textos

import chromadb

CORPUS_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
MODELO_RAG_PATH = Path(__file__).parent.parent / "data" / "processed" / "modelo_fase5.pt"


def embeder_query(texto: str, modelo_embeddings: SentenceTransformer):
    return modelo_embeddings.encode(texto)


def abrir_coleccion() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(NOMBRE_COLECCION)


def cargar_respuestas(ids: list[str]) -> dict[str, str]:
    import json

    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)

    id_a_respuesta = {}
    for registro in corpus:
        id_a_respuesta.setdefault(registro["pregunta_id"], registro["resultado_oficial"])

    return {pregunta_id: id_a_respuesta[pregunta_id] for pregunta_id in ids}


def extraer_filtro_jornada(query: str) -> dict[str, str] | None:
    """Detecta "jornada N" en la query y arma el filtro where de Chroma.

    Metadata extraction previo al retrieval: la busqueda semantica sola no
    "ancla" bien queries conversacionales contra un corpus de plantillas
    rigidas (Fase 3) - acotar por jornada, cuando la query la menciona,
    evita que el top-k se llene de resultados de otras jornadas.
    """
    match = re.search(r"jornada\s+(\d+)", query, re.IGNORECASE)
    if not match:
        return None
    return {"jornada": f"J{int(match.group(1)):02d}"}


def extraer_filtro_partido(query: str) -> dict[str, str] | None:
    """Detecta "Partido N" en la query y arma el filtro where de Chroma.

    Sin esto, un numero de partido explicito en la query no ancla bien la
    busqueda semantica: el corpus repite estructura casi identica entre
    preguntas de tarjeta/corner de partidos distintos, y esas dominan el
    top-k por similitud general aunque la query pida "quien gana" de un
    partido puntual (comprobado con J03_P03_RESULTADO cayendo al puesto #12
    de 33 sobre "Partido 3").
    """
    match = re.search(r"partido\s+(\d+)", query, re.IGNORECASE)
    if not match:
        return None
    return {"partido": f"P{int(match.group(1)):02d}"}


def construir_where(query: str) -> dict | None:
    """Combina los filtros de metadata detectados en la query (jornada, partido)."""
    filtros = [f for f in (extraer_filtro_jornada(query), extraer_filtro_partido(query)) if f]
    if not filtros:
        return None
    if len(filtros) == 1:
        return filtros[0]
    return {"$and": filtros}


def recuperar_contexto(
    coleccion: chromadb.Collection, vector_query, k: int = 3, where: dict | None = None
) -> list[tuple[str, str]]:
    """Devuelve k pares (pregunta, respuesta) reales del corpus, mas parecidos a la query."""
    resultado = coleccion.query(query_embeddings=[vector_query.tolist()], n_results=k, where=where)
    ids = resultado["ids"][0]
    preguntas = cargar_textos(ids)
    id_a_respuesta = cargar_respuestas(ids)
    return [(pregunta, id_a_respuesta[pregunta_id]) for pregunta_id, pregunta in zip(ids, preguntas)]


def construir_prompt(query: str, contexto: list[tuple[str, str]]) -> str:
    """Arma el texto de entrada del LLM: hechos recuperados + la query sin responder.

    Imita el formato "pregunta: respuesta" del corpus (no instrucciones en
    lenguaje natural), porque el LLM de Fase 4 solo sabe continuar texto con
    el estilo que vio en el entrenamiento, no seguir ordenes.
    """
    lineas = [f"{pregunta} {respuesta}" for pregunta, respuesta in contexto]
    lineas.append(query)
    return "\n".join(lineas)


def generar_con_corte(
    modelo,
    prompt: str,
    merges: list[tuple[str, str]],
    simbolo_a_id: dict[str, int],
    id_a_simbolo: dict[int, str],
    max_seq_len: int,
    max_nuevos_tokens: int = 20,
) -> str:
    """Genera texto autoregresivo (greedy, igual que generar() de Fase 4) pero
    parando apenas aparece la primera senal de que termino la respuesta util.

    El modelo no tiene token <EOS> - nunca fue entrenado para "saber" donde
    termina una respuesta, asi que sin este corte seguiria generando ruido
    hasta completar max_nuevos_tokens (ver docs/tutor/fase-5). Se corta por
    simbolo, no por texto ya decodificado: decodificar_ids junta simbolos sin
    separador (el espacio sale del propio marcador </w> de cada simbolo), asi
    que cortar el string final por longitud de caracteres puede caer a mitad
    de palabra. Se para en el primer simbolo nuevo que sea "?" (el corpus
    siempre arranca ahi una pregunta nueva) o <UNK> (senal de que el modelo
    ya salio del terreno que conoce bien).
    """
    ids_prompt = codificar_ids(prompt, merges, simbolo_a_id)
    ids = list(ids_prompt)

    with torch.no_grad():
        for _ in range(max_nuevos_tokens):
            entrada = torch.tensor([ids[-max_seq_len:]])
            logits = modelo(entrada)
            siguiente_id = torch.argmax(logits[0, -1]).item()

            simbolo = id_a_simbolo[siguiente_id]
            if "?" in simbolo or simbolo == "<UNK>":
                break

            ids.append(siguiente_id)

    return decodificar_ids(ids, id_a_simbolo)


def rag(query: str, k: int = 3, max_nuevos_tokens: int = 20) -> tuple[str, str]:
    modelo_embeddings = SentenceTransformer(NOMBRE_MODELO)
    coleccion = abrir_coleccion()

    vector_query = embeder_query(query, modelo_embeddings)
    where = construir_where(query)
    # Con filtro por partido, el grupo tiene 5 tipos de pregunta
    # (MARCADOR/RESULTADO/AMARILLA/ROJA/CORNER) y el orden semantico entre
    # ellos no es confiable (ver extraer_filtro_partido) - se piden todos
    # para no dejar el dato correcto afuera por el corte de k.
    if where is not None and "partido" in str(where):
        k = max(k, 5)
    contexto = recuperar_contexto(coleccion, vector_query, k=k, where=where)
    prompt = construir_prompt(query, contexto)

    modelo_lm, simbolo_a_id = cargar_modelo(MODELO_RAG_PATH)
    id_a_simbolo = {v: k_ for k_, v in simbolo_a_id.items()}
    merges = cargar_merges()
    max_seq_len = modelo_lm.embedding.position_embedding.num_embeddings

    respuesta = generar_con_corte(
        modelo_lm, prompt, merges, simbolo_a_id, id_a_simbolo, max_seq_len,
        max_nuevos_tokens=max_nuevos_tokens,
    )

    return prompt, respuesta


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    query = sys.argv[1] if len(sys.argv) > 1 else "¿Quién gana Partido 11?"

    prompt, respuesta = rag(query)

    print("--- Contexto recuperado + query (prompt) ---")
    print(prompt)
    print("\n--- Generacion del LLM propio (continua el prompt) ---")
    print(respuesta)


if __name__ == "__main__":
    main()
