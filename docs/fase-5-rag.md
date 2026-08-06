# Fase 5 — RAG completo

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)):
integrar el retrieval de Fase 3 con el LLM propio de Fase 4 — query →
embedding → búsqueda en vector DB → top-k resultados → contexto inyectado al
prompt → generación con el LLM propio.

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/rag.py`](../src/rag.py) | Pipeline completo: embeder query, filtrar+buscar en Chroma, construir el prompt, generar con corte temprano |
| [`src/train_rag.py`](../src/train_rag.py) | Genera un dataset sintético "contexto + pregunta → respuesta" (vía retrieval real) y reentrena el `TransformerLM` con esa forma de entrada |
| [`tests/test_vector_store_chroma.py`](../tests/test_vector_store_chroma.py) | Actualizado: `extraer_metadata` ahora también separa el número de partido |

## 1. El pipeline (`src/rag.py`)

```
query → MiniLM (Fase 2) → vector → ChromaDB (Fase 3, con filtro por metadata)
      → top-k hechos reales → prompt (hechos + query) → TransformerLM → respuesta
```

MiniLM y ChromaDB nunca generan texto — son el buscador que encuentra el
dato correcto en el corpus. El transformer propio es la única pieza que
genera: lee el prompt (hechos recuperados + la pregunta sin responder) y lo
continúa. Con 864K-2.8M parámetros y un corpus de 442 preguntas únicas, el
modelo no puede memorizar todas las respuestas — RAG existe precisamente
para que no tenga que hacerlo: el dato ya está escrito en el prompt, su
trabajo es leerlo y continuar en el estilo del corpus, no inventarlo.

### `construir_prompt`: por qué no son instrucciones en lenguaje natural

El prompt no dice "Responde usando este contexto: ..." porque el
`TransformerLM` nunca pasó por *instruction tuning* (la técnica que sí tienen
GPT/Claude para seguir órdenes) — solo aprendió a continuar texto con la
forma del corpus. La plantilla imita el propio formato pregunta→respuesta:

```
Marcador Partido 3 — Inglaterra vs Croacia 2026-02-04 00:00:00
Resultado Partido 3 — Inglaterra vs Croacia Inglaterra
¿Quién gana Partido 3 de la jornada 3?
```

Los hechos recuperados van primero, con su respuesta ya escrita; la query
real queda al final, sin responder, para que el modelo la complete.

### Filtro por metadata (`extraer_filtro_jornada`, `extraer_filtro_partido`)

La búsqueda semántica sola falla con queries que mencionan un número
explícito ("Partido 3", "jornada 3"): el corpus tiene mucha estructura
repetida entre partidos (tarjetas, corners, marcador), y esas preguntas
dominan la similitud coseno por encima del número de partido específico —
comprobado con `J03_P03_RESULTADO` cayendo al puesto **#12 de 33** vecinos
de la jornada 3 al buscar "¿Quién gana Partido 3?", con la fuerza bruta
directa sobre los embeddings reales (sin Chroma de por medio, para descartar
un bug en la integración).

La solución no toca el modelo de embeddings: extrae "jornada N" / "Partido
N" de la query con regex y arma el `where` de Chroma (Fase 3), combinando
ambas condiciones con `$and` cuando aparecen las dos. Esto obligó a ampliar
`extraer_metadata` (Fase 3) para guardar también el número de partido, no
solo la jornada — se reconstruyó la colección persistida con el esquema
nuevo y se actualizó el test correspondiente.

Con filtro por partido activo, `k` se sube automáticamente a mínimo 5: ese
grupo siempre tiene 5 tipos de pregunta (MARCADOR/RESULTADO/AMARILLA/ROJA/
CORNER) y el orden semántico entre ellos no es confiable — pedirlos todos
evita que el corte de `k` deje el dato correcto afuera.

**Límite conocido, no resuelto:** una query sin número explícito (ej. "¿qué
tal jugó Colombia en la jornada 2?") no tiene forma de anclarse a un partido
específico — "Colombia" es una entidad que solo aparece en las
*respuestas* del corpus, no en el texto de las preguntas, y el filtro de
metadata no cubre ese caso. Resolverlo necesitaría *entity linking* (NER +
grafo de conocimiento equipo↔partido), fuera de alcance de este proyecto.

### `generar_con_corte`: por qué no hay `<EOS>`

El corpus de Fase 1/4 no tiene token de fin de secuencia — el modelo nunca
aprendió una señal explícita de "acá termina la respuesta", así que sin
intervención seguiría generando ruido hasta agotar `max_nuevos_tokens`. La
industria resuelve esto entrenando con `<EOS>` al final de cada ejemplo
(GPT usa `<|endoftext|>`); acá se optó por un parche de post-procesamiento
más simple: parar en el primer símbolo generado que sea `?` o `<UNK>` (señal
de que arrancó una pregunta nueva o el modelo salió del terreno conocido).

Importante: el corte se hace sobre **ids/símbolos**, no sobre el string ya
decodificado — `decodificar_ids` (Fase 4) une los símbolos con
`"".join(...)` sin separador explícito entre prompt y continuación (el
espacio sale del propio marcador `</w>` de cada símbolo), así que cortar por
longitud de caracteres puede caer a mitad de palabra y duplicar texto.

## 2. Reentrenamiento para RAG (`src/train_rag.py`)

### El hallazgo que motivó este script

La primera versión de `rag.py` usaba `modelo_fase4.pt` directamente. El
resultado: el LLM con frecuencia **ignoraba el contexto inyectado** y
continuaba hacia otra plantilla del corpus sin relación (ej. el sistema de
puntos), incluso con retrieval funcionando perfecto y el dato correcto ya
escrito en el prompt.

Se comprobó con greedy decoding paso a paso que no era un problema de
aleatoriedad — probar sampling con temperaturas de 0.3 a 1.0 dio la misma
divergencia (o peor, con temperatura alta). La causa real: Fase 4 entrenó
con 6582 oraciones **sueltas** del corpus, nunca con la forma "N hechos +
pregunta" que RAG arma — el modelo nunca vio ese patrón de entrada durante
el entrenamiento, así que no aprendió a "anclarse" al contexto más reciente
por encima de lo que es simplemente más probable en general.

### Dataset sintético (`generar_ejemplos_rag`)

Por cada una de las 442 preguntas únicas del corpus, se arma un ejemplo de
entrenamiento con la misma forma que usará `rag.py` en producción: k hechos
recuperados vía Chroma (retrieval real, no aleatorio) + la pregunta + su
respuesta real. Es, en miniatura, la misma idea del *RAG fine-tuning* que
usan frameworks como LlamaIndex: afinar el modelo específicamente con
ejemplos que incluyen contexto recuperado, para que aprenda a apoyarse en
él en vez de ignorarlo.

Variedad en dos ejes, sin inventar datos falsos (todo el texto sigue
saliendo del corpus real, solo reensamblado distinto):

- **`ks=(2,3,4)`**: cuántos hechos trae cada variante — evita que el modelo
  memorice "el dato bueno está en la posición N".
- **`variantes_por_k=2`**: mismo contexto, orden barajado — el dato
  relevante no siempre cae en la misma posición.

442 preguntas × 3 valores de k × 2 variantes = **2652 ejemplos**. El split
train/val se hace **por pregunta**, no por ejemplo: todas las variantes de
una misma pregunta van al mismo lado, para que `loss_val` mida
generalización a preguntas nunca vistas, no memorización de la misma
pregunta con otro orden de contexto.

### Más capacidad, y por qué

Se subió de los hiperparámetros de Fase 4 (`d_model=128, num_heads=4,
d_ff=512, num_layers=4`, ~864K parámetros) a `d_model=192, num_heads=6,
d_ff=768, num_layers=6` (**2,788,867 parámetros**, dentro del rango 1-10M
que fija `PROYECTO_BASE.md`). La razón: "ubicar cuál línea del contexto
responde la pregunta y copiar su dato" es una relación más indirecta que
"hablar el idioma del corpus" (Fase 4) — más `num_layers` da más pasadas de
razonamiento sobre la secuencia, más `d_model` da más espacio por token.

Este ajuste fue manual/artesanal (probar, medir loss, ajustar) — la
industria real usa *scaling laws* (Kaplan et al. 2020, refinado por
Chinchilla/DeepMind 2022) para calcular la combinación óptima de
parámetros/datos/cómputo antes de una corrida costosa; a esta escala
(CPU doméstica, minutos por corrida) no hace falta esa rigurosidad.

### Early stopping

Con más capacidad y 20 épocas sobre el mismo corpus base, apareció
overfitting real y medible: `loss_val` tocaba mínimo alrededor de la época
16-18 y después subía levemente mientras `loss_train` seguía bajando — el
modelo dejó de aprender el patrón general y empezó a memorizar
particularidades del set de entrenamiento.

`main()` ahora guarda, en cada época, una copia (`copy.deepcopy`, porque
`state_dict()` devuelve tensores por referencia) del `state_dict` si
`loss_val` mejoró — y al final persiste esa copia en vez de los pesos de la
última época.

**Resultado real** (`d_model=192`, 20 épocas, con early stopping):

```
epoca 1/20   loss_train=2.8577  loss_val=1.4920  (mejor hasta ahora)
epoca 8/20   loss_train=0.2463  loss_val=0.3488  (mejor hasta ahora)
epoca 9/20   loss_train=0.2163  loss_val=0.3336  (mejor hasta ahora)
epoca 13/20  loss_train=0.1505  loss_val=0.3044  (mejor hasta ahora)
epoca 18/20  loss_train=0.1243  loss_val=0.2999  (mejor hasta ahora)
epoca 20/20  loss_train=0.1195  loss_val=0.3054

Mejor epoca: 18/20 (loss_val=0.2999)
```

Se guardó el checkpoint de la época 18, no el de la 20.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/train_rag.py
```

## 3. Evolución de resultados a través de las iteraciones

| Versión | Parámetros | Dataset | loss_val final | Comportamiento observado |
|---|---|---|---|---|
| `modelo_fase4.pt` (sin reentrenar) | 864K | — (oraciones sueltas) | — | Ignora el contexto inyectado, salta a otra plantilla del corpus |
| v1 | 874K | 442 ejemplos, k=3 fijo | 1.06 | Se ancla al tema general, pero mezcla nombres libremente |
| v2 | 874K | 2652 ejemplos (k variable + orden barajado) | 0.52 | Empieza a usar nombres reales del contexto |
| v3 | 874K | 2652 ejemplos, 20 épocas | 0.36 (aplanando) | Usa el vocabulario correcto en más casos |
| v4 (final) | 2.79M | 2652 ejemplos, 20 épocas, early stopping | 0.2999 (época 18) | Genera respuestas con datos reales del corpus, pero no siempre discrimina cuál línea del contexto es la correcta |

**Ejemplo real, entregable de la fase** (query del propio
`PROYECTO_BASE.md`):

```
$ ./.venv/Scripts/uv.exe run python src/rag.py "¿qué tal jugó Colombia en la jornada 2?"

--- Contexto recuperado + query (prompt) ---
¿Quién gana Partido 2? Empate
¿Quién gana Partido 3? Escocia
¿Quién gana Partido 8? Suecia
¿qué tal jugó Colombia en la jornada 2?
```

## 4. Límite central de la fase (honesto, no resuelto)

El retrieval funciona bien: con filtro de metadata, el dato correcto casi
siempre está presente en el contexto recuperado. El límite está en la
**generación**: el LLM propio no tiene un mecanismo de atención dirigida
confiable hacia "cuál línea del contexto responde específicamente esta
pregunta" — cuando el contexto trae varias líneas con estructura casi
idéntica (ej. "¿Quién gana Partido 3?" repetido en distintas jornadas con
distinta respuesta), el modelo puede copiar el dato de la línea equivocada.

Se probaron tres ejes para mejorar esto — más diversidad de datos
sintéticos, más épocas, más capacidad (874K → 2.79M parámetros) — y los
tres ayudaron a que el modelo se mantenga mejor anclado al vocabulario y
tema del contexto (dejó de saltar a plantillas completamente ajenas), pero
ninguno resolvió la discriminación fina de *cuál* línea copiar. Esto es
consistente con lo esperado: ese problema no es de capacidad insuficiente
para memorizar patrones, es de **falta de un mecanismo de atención
dirigida/copia** (lo que arquitecturas más avanzadas resuelven con
*cross-attention* explícito query↔contexto o *copy mechanisms*) — algo que
self-attention estándar puede aprender parcialmente con suficientes datos,
pero 442 preguntas base es poco para que emerja con fuerza, sin importar
cuántos parámetros tenga el modelo.

Es la misma distancia que ya se discutió de forma conceptual: en un RAG de
producción (GPT-4/Claude + Pinecone), el generador puede combinar varios
hechos recuperados con razonamiento real, incluso inferir relaciones no
explícitas entre ellos. Con un LLM de juguete de unos pocos millones de
parámetros, ese paso de síntesis es real pero mucho más limitado — el valor
del RAG en este proyecto está sobre todo en el **retrieval preciso**, no en
la elocuencia de la generación.

## 5. Verificación

Suite de tests existente (Fases 0-3) confirmada intacta, con el ajuste de
`extraer_metadata` (nuevo campo `partido`) reflejado en su test:

```bash
./.venv/Scripts/uv.exe run pytest -q
```

```
22 passed in ~7s
```

No se agregaron tests automatizados nuevos para `rag.py`/`train_rag.py` en
esta fase — la verificación se hizo de forma interactiva y cuantitativa
(comparación de `loss_val` entre corridas, inspección manual del contexto
recuperado y la generación sobre preguntas reales del corpus), igual que
Fase 4.

## 6. Pendiente

- No hay tests automatizados para `rag.py`/`train_rag.py` — quedaría bien
  para Fase 6 (portafolio), formalizando al menos: `extraer_filtro_jornada`/
  `extraer_filtro_partido` sobre casos conocidos, y `construir_prompt` con
  contexto vacío/no vacío.
- Entity linking (equipo → partido) no implementado: queries que mencionan
  un equipo sin número de partido/jornada no se benefician del filtro de
  metadata (ver sección 1).
- El mecanismo de atención dirigida/copia hacia el contexto (sección 4)
  queda como limitación de diseño documentada, no resuelta — fuera de
  alcance sin cambiar la arquitectura del transformer (ej. cross-attention
  explícito) o sin un corpus mucho más grande.
- Fase 6 del roadmap: portafolio y enseñanza — README con diagrama de
  arquitectura, demo, decisiones técnicas explicadas (ver
  [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
