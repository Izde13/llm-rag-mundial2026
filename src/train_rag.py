"""
Fase 5 - Reentrenamiento para RAG: el LLM de Fase 4 nunca vio, durante el
entrenamiento, la forma "contexto + pregunta" que rag.py le presenta ahora.
Aprendio a hablar el "idioma" del corpus, pero no aprendio a anclarse al
contexto recien inyectado - por eso, en generacion, a veces ignora el dato
recuperado y sigue con otra plantilla del corpus (comprobado en docs/tutor).

Este script genera un dataset sintetico con exactamente esa forma (usando
el propio retrieval de Chroma para armar el contexto de cada ejemplo, igual
que hara rag.py en produccion) y reentrena el TransformerLM desde cero con
los mismos hiperparametros de Fase 4, para que el patron "leer el contexto
y repetir el dato" quede aprendido en los pesos.

No pisa modelo_fase4.pt (entregable ya documentado de esa fase): guarda en
modelo_fase5.pt.
"""

import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from bpe_vocab import TOKEN_PAD, cargar_merges, construir_vocab_ids
from rag import abrir_coleccion, cargar_respuestas, construir_prompt
from train import evaluar, preparar_lote, un_paso_de_entrenamiento
from transformer_model import TransformerLM
from vector_store_chroma import buscar

CORPUS_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
MODELO_PATH = Path(__file__).parent.parent / "data" / "processed" / "modelo_fase5.pt"


def cargar_preguntas_unicas(corpus: list[dict]) -> tuple[list[str], list[str]]:
    vistos: dict[str, str] = {}
    for registro in corpus:
        vistos.setdefault(registro["pregunta_id"], registro["pregunta"])
    return list(vistos.keys()), list(vistos.values())


def generar_ejemplos_rag(
    ks: tuple[int, ...] = (2, 3, 4), variantes_por_k: int = 2
) -> list[list[str]]:
    """Arma varios textos de entrenamiento por cada pregunta del corpus, con
    la misma forma que construye rag.py: k hechos recuperados (via Chroma,
    retrieval real, no aleatorio) + la pregunta + su respuesta real.

    Variedad en dos ejes, sin inventar datos falsos (todo sigue siendo texto
    real del corpus, solo reensamblado distinto):
    - `ks`: cuantos hechos de contexto trae cada ejemplo (2 a 4) - para que
      el modelo no memorice "el dato bueno esta en la posicion N", tiene que
      aprender a encontrarlo con distintas cantidades de "ruido" alrededor.
    - `variantes_por_k`: mismo contexto recuperado, pero con el orden
      barajado en cada variante - por lo mismo, el dato relevante no siempre
      cae en la misma posicion.

    Devuelve una lista de listas: todas las variantes de una misma pregunta
    van juntas, para poder mandar la pregunta ENTERA a train o val (nunca
    mezclada) y que loss_val mida generalizacion real, no memorizacion de la
    misma pregunta vista con otro orden.
    """
    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)
    ids, preguntas = cargar_preguntas_unicas(corpus)
    id_a_pregunta = dict(zip(ids, preguntas))
    id_a_respuesta = cargar_respuestas(ids)

    coleccion = abrir_coleccion()

    ejemplos_por_pregunta = []
    for pregunta_id in ids:
        vector_query = coleccion.get(ids=[pregunta_id], include=["embeddings"])["embeddings"][0]
        pregunta = id_a_pregunta[pregunta_id]
        respuesta = id_a_respuesta[pregunta_id]

        variantes = []
        for k in ks:
            vecinos = buscar(coleccion, vector_query, k=k, excluir_id=pregunta_id)
            contexto_base = [(id_a_pregunta[vid], id_a_respuesta[vid]) for vid, _ in vecinos]

            for _ in range(variantes_por_k):
                contexto = contexto_base.copy()
                random.shuffle(contexto)
                prompt = construir_prompt(pregunta, contexto)
                variantes.append(f"{prompt} {respuesta}")

        ejemplos_por_pregunta.append(variantes)

    return ejemplos_por_pregunta


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(0)
    torch.manual_seed(0)

    print("Generando ejemplos sinteticos contexto+pregunta->respuesta (via retrieval real)...")
    ejemplos_por_pregunta = generar_ejemplos_rag()
    total_ejemplos = sum(len(v) for v in ejemplos_por_pregunta)
    print(f"{len(ejemplos_por_pregunta)} preguntas -> {total_ejemplos} ejemplos (variantes de contexto).\n")

    todos_los_textos = [texto for variantes in ejemplos_por_pregunta for texto in variantes]
    merges = cargar_merges()
    simbolo_a_id, _ = construir_vocab_ids(merges, todos_los_textos)
    vocab_size = len(simbolo_a_id)

    # Split por PREGUNTA, no por ejemplo: todas las variantes de una misma
    # pregunta van al mismo lado, para que loss_val mida generalizacion a
    # preguntas nunca vistas, no memorizacion de la misma pregunta con otro
    # orden de contexto.
    preguntas_barajadas = ejemplos_por_pregunta.copy()
    random.shuffle(preguntas_barajadas)
    corte = int(len(preguntas_barajadas) * 0.9)
    textos_train = [texto for variantes in preguntas_barajadas[:corte] for texto in variantes]
    textos_val = [texto for variantes in preguntas_barajadas[corte:] for texto in variantes]

    max_len = 128  # ejemplos mas largos que Fase 4 (contexto + pregunta + respuesta)
    batch_size = 32
    num_epocas = 20
    lr = 3e-4
    # Mas capacidad que Fase 4 (d_model=128, num_heads=4, d_ff=512, num_layers=4,
    # ~874K params): la tarea de RAG (ubicar cual linea del contexto responde
    # la pregunta y copiar su dato) es una relacion mas indirecta que "hablar
    # el idioma del corpus" (Fase 4) - mas num_layers da mas pasadas de
    # "razonamiento" sobre la secuencia, mas d_model da mas espacio por token
    # para cargar esa relacion. d_ff se mantiene en 4x d_model (proporcion del
    # paper original). Sigue dentro del rango 1-10M que fija PROYECTO_BASE.md.
    d_model = 192
    num_heads = 6  # 192 / 6 = 32 (division exacta requerida por MultiHeadAttention)
    d_ff = 768
    num_layers = 6

    modelo = TransformerLM(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=max_len)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    funcion_perdida = nn.CrossEntropyLoss(ignore_index=simbolo_a_id[TOKEN_PAD])

    total_params = sum(p.numel() for p in modelo.parameters())
    print("=== Entrenamiento TransformerLM para RAG (Fase 5) ===")
    print(f"Ejemplos: {total_ejemplos} -> {len(textos_train)} train / {len(textos_val)} val")
    print(f"vocab_size={vocab_size}, d_model={d_model}, num_layers={num_layers}, parametros={total_params:,}")
    print(f"max_len={max_len}, batch_size={batch_size}, epocas={num_epocas}, lr={lr}\n")

    # Early stopping: nos quedamos con los pesos de la epoca que mejor
    # generalizo (menor loss_val), no con los de la ultima epoca. Sin esto,
    # con suficiente capacidad y epocas el modelo empieza a memorizar el set
    # de entrenamiento en vez de aprender el patron general -- loss_train
    # sigue bajando pero loss_val deja de mejorar o empeora (overfitting,
    # ver docs/tutor/fase-5). state_dict() devuelve tensores por referencia,
    # por eso hace falta deepcopy: sin copiar, seguirian modificandose en
    # las epocas siguientes junto con el resto del entrenamiento.
    mejor_loss_val = float("inf")
    mejor_epoca = 0
    mejor_state_dict = None

    for epoca in range(1, num_epocas + 1):
        random.shuffle(textos_train)
        loss_acumulado = 0.0
        num_lotes = 0

        for i in range(0, len(textos_train), batch_size):
            lote_textos = textos_train[i : i + batch_size]
            entrada, objetivo = preparar_lote(lote_textos, merges, simbolo_a_id, max_len)
            loss = un_paso_de_entrenamiento(modelo, optimizador, funcion_perdida, entrada, objetivo)
            loss_acumulado += loss
            num_lotes += 1

        loss_train = loss_acumulado / num_lotes
        loss_val = evaluar(modelo, funcion_perdida, textos_val, merges, simbolo_a_id, max_len, batch_size)

        marca = ""
        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoca
            mejor_state_dict = copy.deepcopy(modelo.state_dict())
            marca = "  (mejor hasta ahora)"

        print(f"epoca {epoca}/{num_epocas}  loss_train={loss_train:.4f}  loss_val={loss_val:.4f}{marca}")

    print(f"\nMejor epoca: {mejor_epoca}/{num_epocas} (loss_val={mejor_loss_val:.4f})")

    MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modelo_state_dict": mejor_state_dict,
            "simbolo_a_id": simbolo_a_id,
            "hiperparametros": {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "num_layers": num_layers,
                "max_seq_len": max_len,
            },
        },
        MODELO_PATH,
    )
    print(f"Modelo guardado en: {MODELO_PATH}")


if __name__ == "__main__":
    main()
