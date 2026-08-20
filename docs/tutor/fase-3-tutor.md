# Tutor — Fase 3: Vector Database

## Explicación del tutor

**Por qué la fuerza bruta deja de alcanzar.** `top_k_similares` (Fase 2)
compara la consulta contra los 442 vectores del corpus cada vez —
O(n·d), n vectores de d dimensiones. Con n=442 es instantáneo, pero con
n=1 millón (ej. todos los mensajes de un chat de soporte) cada búsqueda
recorre un millón de comparaciones, y en sistemas reales (buscadores,
recomendadores, RAG a escala) el corpus puede tener cientos de millones o
miles de millones de vectores, con búsquedas ocurriendo muchas veces por
segundo. La solución de la industria es ANN (Approximate Nearest
Neighbor): renunciar a encontrar siempre el vecino exacto a cambio de
velocidad — el mismo trade-off exactitud-vs-velocidad de muchos sistemas a
gran escala.

**Dato histórico:** FAISS (*Facebook AI Similarity Search*) lo publicó
Meta en 2017, tras necesitar indexar miles de millones de vectores para
búsqueda de similitud a escala de toda la plataforma.

**Qué es un índice FAISS, en términos sencillos.** FAISS es el "buscador
de puntos cercanos en el mapa": se le entregan todas las coordenadas (los
442 vectores embebidos en Fase 2), las guarda organizadas internamente, y
cuando se le pregunta "¿qué está más cerca de este punto?" devuelve los
vecinos más próximos — rápido, incluso con 100 millones de puntos en vez
de 442. No entiende el contenido (no sabe de fútbol); todo el
"entendimiento" semántico ya quedó capturado en Fase 2 por MiniLM. FAISS
solo resuelve la parte de **búsqueda eficiente** sobre coordenadas ya
calculadas. La interfaz central es: crear índice → añadir vectores →
buscar.

**`IndexFlatIP` y por qué hay que normalizar.** Se usa `IndexFlatIP`
(inner product / producto punto) en vez de `IndexFlatL2` (distancia
euclidiana) porque los embeddings de MiniLM se comparan por similitud
coseno, y `coseno(a,b) = (a·b)/(|a|×|b|)`. Si se normalizan los vectores a
norma 1 antes (`|a|=|b|=1`), el denominador desaparece y
`coseno(a,b) = a·b` — producto punto de vectores normalizados es
exactamente igual a similitud coseno, no una aproximación. Por eso
`normalizar()` divide cada vector por su norma antes de indexar y antes de
cada búsqueda: mueve la división fuera del bucle de búsqueda (se hace una
vez, no en cada comparación), lo cual no se nota con 442 vectores pero
importa a escala de millones.

**Bug real encontrado y corregido:** `buscar()` pedía `k+1` resultados
para poder excluir la pregunta ancla, pero el filtro se aplicaba
*después* de truncar la lista a tamaño `k` — si el ancla no caía
exactamente en el primer lugar, el resultado final quedaba con menos de
`k` elementos. Se corrigió pasando `excluir_id` explícito y filtrando
dentro del mismo bucle que cuenta hasta `k`. Lección: al combinar "pedir
de más" con "excluir después", la exclusión debe ocurrir en el mismo
punto donde se cuenta el límite.

**FAISS vs. ChromaDB.** FAISS solo guarda vectores — no sabe que el punto
#37 corresponde a `J03_P05_MARCADOR` ni cuál es su texto; por eso Fase 2/3
necesitan `embeddings.npy` + `embeddings_ids.json` como archivos
separados, cruzados a mano por índice de fila, y no hay persistencia (se
reconstruye el índice desde cero cada vez que se corre el script).
ChromaDB es una vector database completa: guarda vector + texto + id +
metadata juntos en una colección persistida en disco
(`data/processed/chroma_db/`), lista para reabrirse sin recalcular nada.
En una frase: FAISS es el motor de búsqueda por cercanía; ChromaDB es la
base de datos completa (con ese tipo de motor por dentro) que además
guarda, persiste y organiza todo lo demás alrededor — la diferencia entre
tener solo el algoritmo y tener el sistema completo listo para usarse en
una aplicación real.

**Qué queda en disco en `chroma_db/`:** `chroma.sqlite3` (ids, textos,
metadata — SQLite normal) más una carpeta por colección física con el
índice **HNSW** (Hierarchical Navigable Small World) que Chroma construye
por debajo por defecto: un grafo en capas donde se salta de nodos
"lejanos y generales" a nodos "cercanos y específicos" — la
implementación real del algoritmo ANN mencionado arriba. Con 442 vectores
la aproximación es esencialmente perfecta, pero es la primera vez en la
fase que se toca, aunque sea indirectamente, un índice ANN real.

**Metadata y filtros combinados.** Se añadió `extraer_metadata` (jornada +
tipo_pregunta, parseado del propio `pregunta_id`) guardada junto a cada
vector, habilitando el parámetro `where` de Chroma: combinar búsqueda
semántica con un filtro exacto en una sola consulta, algo que FAISS puro
no ofrece sin programarlo aparte. Con datos reales (`"¿Quién recibe la
primera tarjeta amarilla?"`, texto repetido entre jornadas): sin filtro,
los vecinos más cercanos son copias idénticas de *otras* jornadas
(similitud 1.0000); con `where={"jornada": "J03"}`, el resultado deja de
ser una copia exacta y aparece `J03_P03_ROJA` (0.8894) — la primera vez en
la fase que un resultado combina cercanía semántica y una condición
exacta a la vez.

**Comparación final (entregable de la fase).** Los tres métodos (fuerza
bruta, FAISS, ChromaDB) devuelven exactamente los mismos 5 vecinos y los
mismos scores para `J01_P01_MARCADOR` — confirma que, con este tamaño de
corpus, la diferencia entre ellos es de arquitectura (persistencia,
interfaz, filtros por metadata, escalabilidad), no de precisión.

## Resumen del usuario

**Fase 3:
**El concepto central: por qué la fuerza bruta deja de alcanzar**

Buscar el vecino más cercano exacto es, en el peor caso, **O(n·d)** — n vectores, d dimensiones cada uno. Con n=442 no lo notas. Pero:

- Con n=1 millón de vectores (por ejemplo, todos los mensajes de un chat de soporte de una empresa grande), cada búsqueda recorre 1 millón de comparaciones.
- En sistemas reales (buscadores, recomendadores, RAG a escala) el corpus puede tener cientos de millones o miles de millones de vectores, y las búsquedas ocurren muchas veces por segundo.
- mayor data es mas dificil de encontra con fuerza bruta
- Indice FAISS:  _interfaz_ de FAISS (crear índice → añadir vectores → buscar)
- -**FAISS es, literalmente, el "buscador de puntos cercanos en el mapa".** Tú le entregas todas las coordenadas (los 442 vectores), él las guarda organizadas internamente, y cuando le preguntas "¿qué está más cerca de este punto?", te devuelve los vecinos más próximos — rápido, incluso si en vez de 442 puntos tuvieras 100 millones.
- -En una frase: **FAISS es el motor de búsqueda por cercanía; ChromaDB es la base de datos completa (con ese tipo de motor por dentro) que además guarda, persiste y organiza todo lo demás alrededor.** Es la diferencia entre tener solo el algoritmo, y tener el sistema completo listo para usarse en una aplicación real
- -A diferencia de FAISS (solo guarda vectores, requiere mapear ids a mano),
ChromaDB guarda vector + texto + id juntos en una coleccion persistida en
disco (data/processed/chroma_db/).
- ChromaDB es similar a FAISS pero con superpoderes

## Validación del tutor

**Qué entendiste bien:**

- El motivo de fondo de la fase — que la fuerza bruta es O(n·d) y se vuelve
  costosa a escala, no con 442 vectores sino con corpus reales de millones —
  está capturado con precisión, incluyendo la intuición correcta de que
  "mayor data es más difícil de encontrar con fuerza bruta".
- La analogía del "buscador de puntos cercanos en el mapa" para FAISS: la
  repetiste completa y en el lugar correcto, lo cual indica que entendiste
  la idea, no solo la memorizaste — FAISS no entiende contenido, solo
  compara coordenadas ya calculadas por otro sistema (MiniLM en Fase 2).
- La distinción FAISS vs. ChromaDB en la frase resumen ("motor de búsqueda"
  vs. "base de datos completa que además guarda, persiste y organiza") es
  exactamente la distinción correcta, y la reforzaste de forma independiente
  con el detalle técnico real (FAISS solo guarda vectores, requiere mapear
  ids a mano vs. Chroma guarda vector+texto+id juntos y persiste en disco).
- La interfaz de tres pasos de FAISS (crear índice → añadir vectores →
  buscar) — es el patrón que se repetirá en herramientas similares más
  adelante (incluida la propia Chroma), vale la pena tenerlo interiorizado.

**Imprecisión a corregir:** "ChromaDB es similar a FAISS pero con
superpoderes" simplifica de más una relación que en realidad es de
**contención**, no de "mismo tipo de cosa pero mejor". Chroma no es una
versión superior de FAISS — usa un algoritmo de la misma familia (HNSW,
similar en espíritu a lo que hace FAISS) **por dentro**, pero además
resuelve un problema distinto: persistencia en disco, guardar texto y
metadata junto al vector, y filtros exactos combinados con búsqueda
semántica (`where={...}`). Son piezas de responsabilidad distinta: FAISS es
una librería de indexación (bajo nivel, tú manejas todo lo demás); Chroma es
una base de datos (alto nivel, ella maneja todo lo demás). La consecuencia
de quedarte con "Chroma = FAISS con superpoderes" es que en Fase 5 (RAG),
donde vas a necesitar justamente esa persistencia + metadata + filtros para
construir el pipeline real, podrías subestimar por qué se eligió Chroma ahí
específicamente en vez de FAISS puro — no es "porque es mejor en general",
es porque el problema de esa fase (recuperar contexto real para inyectarlo
en el prompt del LLM) necesita las piezas que solo Chroma da out-of-the-box.

**Para profundizar (conecta con Fase 5):**

1. El bug real que encontramos en `buscar()` de FAISS (excluir el ancla
   después de truncar a `k`, en vez de antes) no quedó en tu resumen — vale
   la pena que lo repases en `docs/fase-3-vector-database.md` sección 2,
   porque es un patrón de bug genérico ("filtrar antes vs. después de
   truncar") que reaparece en cualquier pipeline de retrieval, incluido el
   RAG de Fase 5.
2. El ejemplo de metadata + `where` (vecinos de una tarjeta amarilla,
   filtrados por jornada) tampoco aparece en tu resumen — es la pieza más
   directamente reutilizable en Fase 5, donde vas a necesitar filtrar
   resultados de retrieval por algún criterio exacto además de la similitud.
3. Si quieres cerrar del todo la intuición de "por qué O(n·d) importa",
   vale la pena en algún momento ver de cerca (aunque sea conceptualmente,
   sin implementarlo) cómo HNSW logra evitar comparar contra todo el
   corpus — la idea de "capas de saltos grandes a saltos finos" que
   mencionamos de pasada, para que la próxima vez que veas `IndexHNSWFlat`
   o el índice de Chroma no sea una caja negra.
