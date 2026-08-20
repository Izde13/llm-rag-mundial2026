# Fase 3 — Vector Database

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): pasar
de "comparar por fuerza bruta" (Fase 2) a usar una **vector database** de
verdad — indexación y búsqueda eficiente por similitud, primero con FAISS
(bajo nivel, entender el algoritmo) y luego con ChromaDB (persistencia local
práctica, más cercano a como se usará en Fase 5 para RAG).

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/vector_store_faiss.py`](../src/vector_store_faiss.py) | Índice FAISS (`IndexFlatIP`) sobre los embeddings de Fase 2 |
| [`src/vector_store_chroma.py`](../src/vector_store_chroma.py) | Colección ChromaDB persistente, con metadata y filtros combinados |
| [`src/vector_store_stats.py`](../src/vector_store_stats.py) | Comparación de los tres métodos lado a lado (entregable de la fase) |
| [`tests/test_vector_store_faiss.py`](../tests/test_vector_store_faiss.py) | Pruebas de FAISS + equivalencia con fuerza bruta |
| [`tests/test_vector_store_chroma.py`](../tests/test_vector_store_chroma.py) | Pruebas de ChromaDB + filtro por metadata |

## 1. Por qué la fuerza bruta deja de alcanzar

`top_k_similares` (Fase 2) compara la consulta contra **todos** los vectores
del corpus cada vez — O(n·d), n vectores de d dimensiones. Con 442 vectores
es instantáneo, pero no escala: con millones de vectores y muchas búsquedas
por segundo, recorrer el corpus completo en cada consulta se vuelve el
cuello de botella.

La solución de la industria es **ANN — Approximate Nearest Neighbor**:
renunciar a encontrar siempre el vecino exacto a cambio de encontrar un
vecino casi seguro correcto, mucho más rápido. Es el mismo trade-off
exactitud-vs-velocidad que aparece en muchos sistemas a gran escala.

**Dato histórico:** FAISS (*Facebook AI Similarity Search*) lo publicó Meta
en 2017, tras necesitar indexar miles de millones de vectores para búsqueda
de similitud a escala de toda la plataforma. Hoy es el estándar de facto
para este problema.

## 2. FAISS: `IndexFlatIP` (`src/vector_store_faiss.py`)

Un índice FAISS envuelve la matriz de vectores y expone `.search(query, k)`.
Se usa `IndexFlatIP` (*inner product*, producto punto) en vez de
`IndexFlatL2` (distancia euclidiana) porque los embeddings de MiniLM se
comparan por **similitud coseno**, y coseno = producto punto **solo si los
vectores están normalizados a norma 1 primero** — identidad matemática:

```
coseno(a, b) = (a·b) / (|a| × |b|)
```

Si `|a| = |b| = 1`, el denominador desaparece y `coseno(a,b) = a·b`. Por
eso `normalizar()` divide cada vector por su norma antes de construir el
índice y antes de cada búsqueda — con eso, `IndexFlatIP` calcula coseno sin
tener que dividir en cada comparación (a diferencia de `similitud_coseno`
de Fase 2, que divide en cada llamada). Con 442 vectores no hay ganancia de
rendimiento perceptible, pero es el mismo patrón — precalcular una vez lo
que si no se repetiría en cada consulta — que sí importa a escala de
millones de vectores y miles de búsquedas por segundo.

Con este tamaño de corpus, `IndexFlatIP` sigue siendo búsqueda exacta por
fuerza bruta, solo que en C++ vectorizado — el valor pedagógico está en la
**interfaz** (crear índice → añadir vectores → buscar), antes de pasar a un
índice aproximado real (`IndexIVFFlat`, `IndexHNSWFlat`) que un corpus de
este tamaño no necesita.

**Bug encontrado durante la construcción:** la función `buscar()` pedía
`k+1` resultados para poder excluir la pregunta ancla de la respuesta, pero
el filtro se aplicaba *después* de truncar la lista a tamaño `k` — si el
ancla no caía exactamente en el primer lugar, el resultado final quedaba
con menos de `k` elementos. Se corrigió pasando `excluir_id` explícito a
`buscar()` y filtrando dentro del mismo bucle que cuenta hasta `k`, no
después. Lección: cuando se combina "pedir de más" con "excluir después",
la exclusión debe ocurrir en el mismo punto donde se cuenta el límite, o se
pierden resultados válidos silenciosamente.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/vector_store_faiss.py
```

## 3. ChromaDB: persistencia y metadata (`src/vector_store_chroma.py`)

FAISS solo guarda vectores — por eso Fase 2/3 necesitan dos archivos
separados (`embeddings.npy` + `embeddings_ids.json`) cruzados a mano por
índice de fila. **ChromaDB** es una vector database completa: guarda
vector + texto + id + metadata juntos en una **colección** persistida en
disco (`data/processed/chroma_db/`), sin recalcular nada al reabrirla.

**Qué queda en disco:** `chroma.sqlite3` (ids, textos y metadata, en SQLite
normal) más una carpeta por colección física con el índice **HNSW**
(*Hierarchical Navigable Small World*) que Chroma construye por debajo por
defecto — un grafo en capas donde se salta desde nodos "lejanos y
generales" a nodos "cercanos y específicos", la implementación real del
algoritmo ANN mencionado en la sección 1. Con 442 vectores la aproximación
es esencialmente perfecta (no hay diferencia práctica frente a exacto),
pero es la primera vez en la fase que se toca, aunque sea indirectamente,
un índice ANN de verdad.

**Metadata y filtros combinados:** `extraer_metadata(pregunta_id)` separa
`jornada` y `tipo_pregunta` del propio id (ej. `J03_P05_MARCADOR` →
`{"jornada": "J03", "tipo_pregunta": "MARCADOR"}`) y se guarda junto a cada
vector. Esto habilita el parámetro `where` de `collection.query(...)`:
combinar búsqueda semántica con un filtro exacto en una sola consulta, algo
que FAISS puro no ofrece sin programarlo aparte — y que en un RAG real es
tan importante como la similitud misma (ej. "lo más parecido, pero solo de
este departamento/usuario/fecha").

**Ejemplo con datos reales** (`"¿Quién recibe la primera tarjeta
amarilla?"`, texto idéntico repetido entre jornadas — mismo caso límite que
`J01_P01_RESULTADO` en Fase 2):

Sin filtro, los vecinos más cercanos son copias idénticas de **otras
jornadas** (similitud 1.0000, pero de `J04`, no de `J03`). Con
`where={"jornada": "J03"}`, la búsqueda se acota a la propia jornada, y el
tercer resultado deja de ser una copia exacta y pasa a ser semánticamente
distinto: `J03_P03_ROJA` (tarjeta roja) con 0.8894 de similitud — la
primera vez en la fase que un resultado combina "cercano en significado" y
"cumple una condición exacta" a la vez.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/vector_store_chroma.py
```

## 4. Comparación final (`src/vector_store_stats.py`)

Entregable de la fase: corre la misma consulta (vecinos de
`J01_P01_MARCADOR`, `k=5`) con los tres métodos y confirma que coinciden.

**Resultado real:**

```
0.8380  J03_P05_MARCADOR
0.7890  J05_P10_MARCADOR
0.7865  J03_P06_MARCADOR
0.7758  J06_P01_MARCADOR
0.7519  J06_D01_MARCADOR
```

Idéntico en fuerza bruta, FAISS y ChromaDB. Confirma con datos reales que,
con este tamaño de corpus, la diferencia entre los tres métodos es de
**arquitectura** (persistencia, interfaz, capacidad de filtrar por
metadata, escalabilidad a corpus grandes) y no de precisión — los tres
resuelven exactamente el mismo problema matemático.

**Salida:** `data/processed/vector_store_comparacion.json`.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/vector_store_stats.py
```

## 5. Verificación (tests)

- [`tests/test_vector_store_faiss.py`](../tests/test_vector_store_faiss.py):
  propiedades del índice con vectores sintéticos, y equivalencia numérica
  contra `top_k_similares` sobre los embeddings reales del corpus.
- [`tests/test_vector_store_chroma.py`](../tests/test_vector_store_chroma.py):
  equivalencia contra fuerza bruta, extracción de metadata, y que
  `where={"jornada": ...}` solo devuelve resultados de esa jornada.

**Cómo correr las pruebas:**

```bash
./.venv/Scripts/uv.exe run pytest tests/test_vector_store_faiss.py tests/test_vector_store_chroma.py -v
```

```
tests/test_vector_store_faiss.py::test_normalizar_produce_vectores_de_norma_uno PASSED
tests/test_vector_store_faiss.py::test_buscar_excluye_el_id_indicado PASSED
tests/test_vector_store_faiss.py::test_buscar_ordena_de_mayor_a_menor PASSED
tests/test_vector_store_faiss.py::test_faiss_coincide_con_fuerza_bruta_sobre_embeddings_reales PASSED
tests/test_vector_store_chroma.py::test_extraer_metadata_separa_jornada_y_tipo PASSED
tests/test_vector_store_chroma.py::test_chroma_coincide_con_fuerza_bruta_sobre_embeddings_reales PASSED
tests/test_vector_store_chroma.py::test_buscar_con_where_solo_devuelve_resultados_de_la_jornada_filtrada PASSED
```

## 6. Pendiente

- `data/processed/chroma_db/` acumula una carpeta huérfana por cada vez que
  se corre el script (`delete_collection` + `create_collection` no limpia
  la carpeta física vieja). No es un problema funcional, pero convendría
  cambiar a `get_or_create_collection` o limpiar el directorio antes de
  reconstruir, si se sigue iterando sobre este script.
- Fase 4 del roadmap: arquitectura transformer completa desde cero (ver
  [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)) — el vector store de esta fase
  se retoma en Fase 5 (RAG) como la pieza de *retrieval*.
