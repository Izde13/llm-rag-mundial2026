# IA Local desde Cero: LLM Propio + RAG con Datos del Mundial 2026

## Objetivo

Construir, desde cero y de forma 100% local, un sistema de IA que combine:
1. Un **LLM propio** (transformer entrenado desde cero, sin usar modelos pre-entrenados como Ollama)
2. Un sistema **RAG** (Retrieval-Augmented Generation) que use ese LLM como generador

Proyecto formativo, pensado para aprender cada pieza a fondo y servir como material de enseñanza para otras personas (repo documentado paso a paso).

## Contexto y datos

Se reutilizan los datos recopilados durante el proyecto de la "polla mundialista" del Mundial 2026 (partidos, jornadas, estadísticas como corners, etc.) como corpus de entrenamiento y como base documental del RAG.

## Restricciones de hardware

- CPU y GPU muy básica
- Esto define el tamaño objetivo del LLM: modelo pequeño (~1-10M parámetros), contexto corto (64-128 tokens), pocas capas (4-6), embeddings pequeños (128-256 dims)

## Conceptos clave a aprender (en orden)

1. **Tokens** — cómo un modelo "ve" el texto; tokenización
2. **Embeddings** — de tokens a vectores semánticos, similitud
3. **Vector databases** — indexación y búsqueda por similitud a escala
4. **Arquitectura Transformer** — self-attention, multi-head attention, positional encoding, layer norm, feed-forward, causal masking
5. **Entrenamiento de un LLM** — loop de entrenamiento, loss, generación de texto
6. **RAG** — retrieval + generación aumentada, cómo se combinan ambos sistemas

## Diferencia LLM vs RAG (para tener presente en el README de enseñanza)

- **LLM**: el modelo generativo en sí. Predice el siguiente token dado un input. Todo lo que "sabe" está en sus pesos, aprendido en el entrenamiento.
- **RAG**: no es un modelo, es una técnica que envuelve a un LLM. Antes de generar, busca información relevante en una base de datos externa (vector DB) y se la inyecta al LLM como contexto en el prompt. El LLM no memoriza los datos externos, los lee en el momento.

En este proyecto, el LLM propio (pequeño y limitado en capacidad de generación libre) se apoya en el retrieval para no depender de haber memorizado todo el corpus — el RAG hace más fácil su tarea: en vez de generar desde cero, completa/parafrasea alrededor del contexto recuperado.

## Roadmap por fases

### Fase 0 — Setup y datos
- Estructura del repo: `data/`, `src/`, `notebooks/`, `tests/`
- Exportar los datos del mundial a JSON/CSV como corpus
- Entorno: Python + venv, gestor de dependencias (`uv` o `poetry`)

### Fase 1 — Tokens
- Concepto: tokenización, BPE
- Construir un tokenizer propio (char-level primero, BPE simple después)
- Entregable: script que tokeniza el corpus y muestra estadísticas

### Fase 2 — Embeddings
- Concepto: de tokens a vectores semánticos, similitud coseno
- Uso de un modelo pre-entrenado liviano (`sentence-transformers`, ej. `all-MiniLM-L6-v2`) para embeber el corpus
- Entregable: notebook comparando embeddings y similitud entre partidos/jornadas

### Fase 3 — Vector Database
- Concepto: indexación, ANN (approximate nearest neighbor)
- FAISS primero (bajo nivel, entender el algoritmo), luego ChromaDB (persistencia local práctica)
- Entregable: pipeline que indexa el corpus y responde queries por similitud

### Fase 4 — LLM propio desde cero
- Concepto: arquitectura transformer completa (self-attention, multi-head attention, positional encoding, layer norm, feed-forward, causal masking)
- Implementación en PyTorch puro (sin `transformers` de HuggingFace)
- Loop de entrenamiento con el corpus del mundial
- Entregable: modelo entrenado que genera texto con "sabor" al corpus

### Fase 5 — RAG completo
- Integración: query → embedding → búsqueda en vector DB → top-k resultados → contexto inyectado al prompt → generación con el LLM propio
- Entregable: `python rag.py "¿qué tal jugó Colombia en la jornada 2?"` funcionando end-to-end

### Fase 6 — Portafolio y enseñanza
- README con diagrama de arquitectura, demo, decisiones técnicas explicadas
- Documentación pensada para que otras personas aprendan del repo (notebooks/markdowns explicativos junto al código)
- Opcional: CLI o UI mínima (Streamlit/Gradio)

## Flujo de trabajo

- Planeación y discusión conceptual: chat
- Construcción del repo, código, entrenamiento y commits: VSCode + Claude Code
