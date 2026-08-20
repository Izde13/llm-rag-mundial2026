# Fase 2 — Embeddings

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): pasar
de tokens (ids sin relación entre sí, Fase 1) a **vectores semánticos**,
donde la cercanía geométrica entre dos vectores refleja similitud de
significado entre los textos que representan. Se usa un modelo pre-entrenado
liviano (`all-MiniLM-L6-v2`, vía `sentence-transformers`) en vez de entrenar
uno propio — entrenar embeddings desde cero requiere corpus del orden de
miles de millones de pares de oraciones, inviable con las restricciones de
hardware del proyecto y con las 442 preguntas únicas de este corpus.

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/embeddings.py`](../src/embeddings.py) | Genera y guarda los embeddings del corpus; funciones de similitud coseno y búsqueda top-k |
| [`src/embeddings_stats.py`](../src/embeddings_stats.py) | Comparación de ejemplos representativos (entregable de la fase) |
| [`tests/test_embeddings.py`](../tests/test_embeddings.py) | Pruebas de las propiedades matemáticas de similitud coseno y top-k |

## 1. Qué es un embedding y por qué no es lo mismo que un token-id

Un token-id (Fase 1) es una etiqueta arbitraria: un número de fila en una
tabla de vocabulario, sin relación matemática entre ids vecinos. Un
**embedding** es un vector de números reales (384 dimensiones, en este
proyecto) donde la *distancia geométrica* entre dos vectores refleja
*similitud semántica* entre los textos que representan — típicamente medida
con **similitud coseno** (el coseno del ángulo entre dos vectores, invariante
a su magnitud).

La cercanía no es una propiedad matemática inherente al texto: es una
propiedad **aprendida por entrenamiento**. El modelo parte de una tabla de
pesos inicializada al azar (mismo mecanismo que cualquier red neuronal) y se
entrena sobre pares de oraciones con relación conocida (ej. post + respuesta
de Reddit, título + resumen de un paper, la misma oración en dos idiomas).
En cada ronda, si el modelo calcula mal la similitud de un par, se ajustan
los vectores en la dirección que reduce ese error (descenso de gradiente).
Después de millones de rondas sobre datos reales, palabras/oraciones que
aparecen en contextos parecidos convergen a zonas cercanas del espacio — no
por diseño explícito, sino como consecuencia estadística de minimizar el
error de predicción sobre el corpus de entrenamiento.

**Dato histórico:** la primera demostración pública influyente de esto fue
`word2vec` (Mikolov et al., Google, 2013), con el ejemplo `vector("Rey") -
vector("Hombre") + vector("Mujer") ≈ vector("Reina")` — relaciones
semánticas codificadas como aritmética vectorial, emergentes del
entrenamiento, no programadas a mano.

## 2. `all-MiniLM-L6-v2`: un modelo ya entrenado, no entrenado aquí

Todo el proceso de entrenamiento descrito arriba **ya ocurrió**, antes de
este proyecto, hecho por el equipo que publicó el modelo (22M de parámetros,
entrenado sobre ~1000 millones de pares de oraciones). Lo que se descarga es
el resultado final — la tabla de pesos ya ajustada — y se usa en modo
**consulta**: se le entrega texto plano, internamente lo tokeniza con su
propio tokenizer BPE (independiente y sin relación con el tokenizer de Fase
1), busca cada token en su tabla ya entrenada, y combina los vectores de
todos los tokens de la oración en un único vector (**mean pooling**:
promedio componente a componente) — el *sentence embedding* de la oración
completa.

Nada de este proceso modifica los pesos de MiniLM. Fase 2 es 100% inferencia
(uso), no entrenamiento — a diferencia de Fase 4, donde sí se va a entrenar
un modelo propio desde cero y se va a ver este mismo mecanismo de ajuste de
pesos ejecutándose con datos reales del proyecto.

**Comparación con la industria:** empresas como OpenAI (`text-embedding-3`)
o Cohere entrenan y sirven sus propios modelos de embeddings como servicio
pago; `sentence-transformers` (Reimers & Gurevych, 2019) es el estándar
abierto equivalente para modelos livianos que corren en CPU local, que es lo
que las restricciones de hardware de este proyecto requieren.

## 3. Generación de embeddings (`src/embeddings.py`)

- `cargar_preguntas(ruta_corpus)`: carga `corpus.json` y **deduplica por
  `pregunta_id`**. El corpus tiene una fila por cada par
  `(pregunta, participante)` — 6582 filas, pero solo 442 preguntas
  realmente distintas (ver sección 5, fue un hallazgo hecho a mitad de la
  fase). Comparar embeddings sobre las 6582 filas crudas compararía texto
  idéntico consigo mismo cientos de veces, sin aportar información.
- `generar_embeddings(textos, modelo)`: llama a
  `SentenceTransformer("all-MiniLM-L6-v2").encode(...)` sobre la lista de
  preguntas únicas.
- `similitud_coseno(vec_a, vec_b)`: implementada a mano
  (`dot(a,b) / (norma(a) * norma(b))`) en vez de usar una función de la
  librería, para que la fórmula quede explícita.
- `top_k_similares(idx_pregunta, embeddings, k)`: dado el índice de una
  pregunta, calcula su similitud coseno contra **todas** las demás filas de
  la matriz de una sola vez (`embeddings @ vector_objetivo`, producto
  matriz-vector — la versión vectorizada de comparar contra cada fila en un
  loop), excluye la comparación consigo misma, y devuelve las `k` más
  parecidas ordenadas de mayor a menor (`np.argsort` invertido). Es
  búsqueda por fuerza bruta (*k-NN exacto*): compara contra el corpus
  completo cada vez, sin ningún índice — la alternativa eficiente
  (*approximate nearest neighbor*, ej. FAISS/HNSW) es justamente el tema de
  Fase 3, motivado por el costo de este enfoque cuando el corpus crece.

**Salida:** `data/processed/embeddings.npy` (matriz 442×384) y
`data/processed/embeddings_ids.json` (lista de `pregunta_id` en el mismo
orden de filas, para poder cruzar índice → pregunta).

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/embeddings.py
```

## 4. Comparación final (`src/embeddings_stats.py`)

Entregable de la fase: toma dos preguntas ancla de distinto tipo
(`J01_P01_MARCADOR` y `J01_P01_RESULTADO`) y muestra su top-5 más similar
sobre el corpus completo de 442 preguntas únicas.

**Resultado real, pregunta de marcador** (`"Marcador Partido 1 — México vs
Sudáfrica"`):

```
0.8380  Marcador Partido 5 — República Checa vs Sudáfrica
0.7890  Marcador Partido 10 — Sudáfrica vs Corea del Sur
0.7865  Marcador Partido 6 — México vs Corea del Sur
0.7758  Marcador Partido 1 — Panamá vs Inglaterra
0.7519  Marcador Dieciseisavos 1 — Sudáfrica vs Canadá (90 min reglamentarios)
```

Similitud alta y graduada entre preguntas que comparten equipo o estructura
— exactamente el comportamiento esperado de un embedding bien entrenado.

**Resultado real, pregunta de resultado** (`"¿Quién gana Partido 1?"`):

```
1.0000  ¿Quién gana Partido 1?
1.0000  ¿Quién gana Partido 1?
1.0000  ¿Quién gana Partido 1?
1.0000  ¿Quién gana Partido 1?
0.9295  ¿Quién gana Partido 2?
```

Caso límite real (no un bug): el texto de la pregunta no incluye la
jornada, así que `"¿Quién gana Partido 1?"` aparece **literalmente idéntico**
en `pregunta_id` distintos (`J01_P01_RESULTADO`, `J02_P01_RESULTADO`,
`J04_P01_RESULTADO`, `J05_P01_RESULTADO`, `J06_P01_RESULTADO`) — mismo
string, mismo vector, similitud 1.0. Ilustra un límite real de comparar por
embeddings: el modelo solo puede diferenciar lo que el texto distingue: dos
preguntas idénticas en texto pero distintas en metadata (jornada) son
indistinguibles para MiniLM. Dato a tener presente para Fase 5 (RAG): el
identificador único de recuperación no puede ser solo el texto de la
pregunta.

**Salida:** `data/processed/embeddings_comparacion.json`.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/embeddings_stats.py
```

## 5. Hallazgo de datos: 6582 filas, 442 preguntas únicas

Al construir la comparación se detectó que buscar el "más similar" sobre
las 6582 filas crudas del corpus devolvía siempre similitud 1.0000 con la
misma pregunta repetida. Investigado con datos reales:

```
Filas totales en corpus.json: 6582
pregunta_id únicos: 442
```

El corpus (heredado del diseño de Fase 0) tiene una fila por cada
`(pregunta, participante)` — 442 preguntas × ~14-17 participantes cada una.
Para cualquier análisis sobre **contenido semántico de preguntas** (no sobre
respuestas de participantes), hay que deduplicar por `pregunta_id` primero.
Esta decisión quedó implementada en `cargar_preguntas` y es la razón por la
que Fase 2 trabaja sobre 442 vectores, no 6582.

## 6. Verificación (tests)

[`tests/test_embeddings.py`](../tests/test_embeddings.py) verifica las
propiedades matemáticas de `similitud_coseno` y `top_k_similares` con
vectores sintéticos (sin cargar el modelo real, para que las pruebas sean
rápidas y deterministas):

- `test_similitud_coseno_vector_consigo_mismo_es_uno`
- `test_similitud_coseno_vectores_ortogonales_es_cero`
- `test_similitud_coseno_vectores_opuestos_es_menos_uno`
- `test_similitud_coseno_es_invariante_a_la_magnitud`
- `test_top_k_similares_excluye_la_pregunta_objetivo`
- `test_top_k_similares_ordena_de_mayor_a_menor`
- `test_top_k_similares_encuentra_el_vector_mas_parecido`

**Cómo correr las pruebas:**

```bash
./.venv/Scripts/uv.exe run pytest tests/test_embeddings.py -v
```

```
tests/test_embeddings.py::test_similitud_coseno_vector_consigo_mismo_es_uno PASSED
tests/test_embeddings.py::test_similitud_coseno_vectores_ortogonales_es_cero PASSED
tests/test_embeddings.py::test_similitud_coseno_vectores_opuestos_es_menos_uno PASSED
tests/test_embeddings.py::test_similitud_coseno_es_invariante_a_la_magnitud PASSED
tests/test_embeddings.py::test_top_k_similares_excluye_la_pregunta_objetivo PASSED
tests/test_embeddings.py::test_top_k_similares_ordena_de_mayor_a_menor PASSED
tests/test_embeddings.py::test_top_k_similares_encuentra_el_vector_mas_parecido PASSED
```

## 7. Pendiente

- Fase 3 del roadmap: vector database — indexar los 442 embeddings para
  búsqueda eficiente (FAISS primero, luego ChromaDB), en vez de la
  comparación por fuerza bruta usada aquí (ver
  [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
- Nota para Fase 5 (RAG): el identificador de recuperación no puede
  depender solo del texto de la pregunta, porque hay texto idéntico
  repetido entre jornadas distintas (sección 5).
