"""
Fase 6 - UI minima sobre el pipeline de RAG (src/rag.py).

No agrega logica nueva: solo le pone una cara visual a rag_con_recursos()
para poder probar el sistema sin terminal. Se muestran dos partes por
separado (contexto recuperado / respuesta generada) para que se vea el
pipeline completo -- retrieval y generacion son piezas distintas (ver
docs/fase-5-rag.md) y mezclarlas en un solo bloque de texto lo esconderia.

Como correrlo:
    ./.venv/Scripts/uv.exe run streamlit run src/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from rag import MODELO_RAG_PATH, cargar_recursos_rag, rag_con_recursos
from vector_store_chroma import CHROMA_DIR

st.set_page_config(page_title="RAG - Polla Mundial 2026", page_icon="⚽")


@st.cache_resource(show_spinner="Cargando modelos (MiniLM + TransformerLM propio)...")
def _recursos():
    return cargar_recursos_rag()


st.title("⚽ RAG con LLM propio — Polla Mundial 2026")
st.caption(
    "Retrieval (MiniLM + ChromaDB, Fase 2-3) + generación con un transformer "
    "entrenado desde cero (Fase 4-5). Ver `docs/fase-5-rag.md` en el repo "
    "para el detalle del pipeline."
)

if not CHROMA_DIR.exists() or not MODELO_RAG_PATH.exists():
    st.error(
        "Faltan artefactos generados. Corré, en orden: "
        "`data_prep.py`, `tokenizer_bpe.py`, `embeddings.py`, "
        "`vector_store_chroma.py`, `train.py`, `train_rag.py`."
    )
    st.stop()

query = st.text_input(
    "Pregunta sobre el Mundial 2026",
    placeholder="¿Quién gana Partido 11?",
)
preguntar = st.button("Preguntar", type="primary")

if preguntar and query.strip():
    modelo_embeddings, coleccion, modelo_lm, simbolo_a_id, id_a_simbolo, merges = _recursos()

    with st.spinner("Buscando contexto y generando respuesta..."):
        prompt, respuesta, contexto = rag_con_recursos(
            query, modelo_embeddings, coleccion, modelo_lm, simbolo_a_id, id_a_simbolo, merges,
        )

    st.subheader("1. Contexto recuperado")
    st.caption("Hechos reales del corpus que ChromaDB encontró más relevantes para la pregunta.")
    for pregunta, respuesta_dato in contexto:
        st.markdown(f"- **{pregunta}** → {respuesta_dato}")

    st.subheader("2. Respuesta generada")
    st.caption("El transformer propio continúa el prompt (contexto + pregunta) con esta salida:")
    st.success(respuesta if respuesta.strip() else "(el modelo no generó texto nuevo)")

    with st.expander("Ver el prompt completo enviado al modelo"):
        st.code(prompt, language=None)
elif preguntar:
    st.warning("Escribí una pregunta primero.")
