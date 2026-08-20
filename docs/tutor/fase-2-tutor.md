# Tutor — Fase 2: Embeddings

## Explicación del tutor

**Qué es un embedding y por qué no es lo mismo que un token-id.** Un
token-id (Fase 1) es una etiqueta arbitraria — un número de fila sin
relación matemática con sus vecinos. Un embedding es un vector de números
reales (384 dimensiones, con `all-MiniLM-L6-v2`) donde la distancia
geométrica (medida con similitud coseno) refleja similitud semántica entre
los textos que representa.

**De dónde sale la cercanía: pesos + entrenamiento.** La tabla de
embeddings es una tabla de **pesos** — inicializada al azar, igual que
cualquier tabla de pesos de una red neuronal. El entrenamiento la ajusta: se
muestran al modelo pares de oraciones con relación *implícita* en la
estructura del dato (post + respuesta de Reddit, título + resumen de un
paper, la misma oración en dos idiomas — nadie etiqueta "gana≈vence" a
mano), se mide el error entre la similitud calculada y la esperada, y se
ajustan los vectores en la dirección que reduce ese error (descenso de
gradiente). Repetido millones de veces sobre datos reales, palabras que
aparecen en contextos parecidos convergen a zonas cercanas del espacio — un
efecto estadístico emergente, no una regla programada.

Dato histórico: `word2vec` (Mikolov et al., 2013) hizo famosa esta idea con
`vector("Rey") - vector("Hombre") + vector("Mujer") ≈ vector("Reina")`.

**MiniLM se consume, no se entrena en este proyecto.** El entrenamiento
(pesos + corpus de ellos) ya ocurrió, antes y en otro lugar, hecho por el
equipo que publicó el modelo. Fase 2 es consulta pura: se le da texto plano,
tokeniza internamente con su propio BPE (sin relación con el tokenizer de
Fase 1 — vocabularios e ids completamente independientes), busca cada token
en su tabla ya fija, y combina los vectores de los tokens de una oración en
un único vector vía **mean pooling** (promedio componente a componente) — el
*sentence embedding*, la unidad real que se compara en esta fase.

**Cómo se mide la cercanía, en código.** `similitud_coseno(a, b)` =
`dot(a,b) / (norma(a) * norma(b))`, implementada a mano para que la fórmula
quede explícita. `top_k_similares(idx, embeddings, k)` compara una fila
contra **todas** las demás de una sola vez (`embeddings @ vector_objetivo`,
producto matriz-vector) y devuelve las `k` más altas — búsqueda por fuerza
bruta (k-NN exacto), sin ningún índice; la motivación directa de por qué
Fase 3 (vector databases, FAISS/HNSW) hace falta cuando el corpus crece.

**Hallazgo real de datos, no solo teoría.** El corpus tiene 6582 filas pero
solo 442 `pregunta_id` únicos (una fila por cada `(pregunta, participante)`).
Comparar sobre las 6582 filas crudas da similitud 1.0 con copias exactas de
la misma pregunta, sin aportar nada — se resolvió deduplicando por
`pregunta_id` antes de generar embeddings. Incluso deduplicado, hay texto
**idéntico carácter por carácter** repetido entre jornadas distintas
(`"¿Quién gana Partido 1?"` en `J01`, `J02`, `J04`, `J05`, `J06`) — caso
límite real, no un bug: el texto no menciona la jornada, así que el
embedding no tiene forma de distinguirlas. Relevante para Fase 5 (RAG): el
identificador de recuperación no puede depender solo del texto de la
pregunta.

## Resumen del usuario

- Debemos dar significados a los tokens, vamos hacer: **vector de números
  reales** donde la _distancia geométrica_ entre vectores refleja
  _similitud semántica_ entre textos.
- Un embedding es un vector, el significado se puede medir con geometria,
  pregunta, ¿Como se que lo que esta mas cerca geometricamente es lo que da
  significado, los tokens no fueron con ids aleatorios?

O sea, yo entiendo que cada token tiene su id, que estos son numeros
aleatorios que salieron de la fase 1, ahora, para la fase 2, vamos a usar un
termino qeu es embedding, que es un vector, ahora, entiendo que en tu
ejmplo colocas id y luego [0.1,0.4] para un id, por ejemplo, eso es
aleatoriamente, hasta este momento todo es aleatorio, entonces, no se si
esto tenga qe ver con los pesos o algo asi, he escuchado ese termino, pero
no nos desviemos, entonces tenemos id seguido de [0.1,0.4], esto como dices
puedes ser la palabra gana, ahora, todos los tokens esta representados de
esta forma, ahora, ¿como saber que el token gana esta cerca del token
Colombia?

La tabla de embeddings — esa matriz de 250 filas × 2 columnas — **es una
tabla de pesos**

Acá está el punto clave que hay que aclarar: **"gana" y "Colombia" NO
deberían quedar cerca**. Son palabras sin relación semántica fuerte (una es
un verbo genérico, la otra un país). Lo que sí esperamos que quede cerca es,
por ejemplo, `"gana"` y `"vence"` (sinónimos funcionales), o `"Colombia"` y
`"Argentina"` (mismo tipo de entidad: países/selecciones).

La pregunta de fondo entonces es: **¿qué mecanismo decide, número por
número, si dos vectores se acercan o se alejan?** Ahí es donde entra el
entrenamiento

Tomemos una pregunta real de tu corpus: `"¿Quién gana Partido 1?"`.

1. **Tokenización interna de MiniLM**: parte el texto en subwords con su
   propio BPE. Algo como: `["¿", "Quién", "gana", "Partido", "1", "?"]` →
   cada uno se convierte en **su** id interno (ej.
   `[204, 891, 4821, 77, 12, 55]` — números inventados para el ejemplo).

2. **Lookup en la tabla ya entrenada**: por cada id, va a la fila
   correspondiente de la tabla (la que quedó fija después del Momento A) y
   saca su vector de 384 números. No hay comparación acá — es leer la fila
   `4821`, la fila `77`, etc. Ya sabe qué hay en cada fila porque quedó
   grabado del entrenamiento.

```
id 204 (¿)       → [0.02, -0.31, ..., 0.18]   (384 números)
id 891 (Quién)   → [0.44,  0.09, ..., -0.02]
id 4821 (gana)   → [0.13, -0.67, ..., 0.55]
id 77 (Partido)  → [0.71,  0.02, ..., 0.19]
id 12 (1)        → [-0.10, 0.88, ..., 0.03]
id 55 (?)        → [0.05, -0.12, ..., 0.44]
```

3. **Combinación en UN solo vector de la oración**: tenés 6 vectores (uno
   por token), pero querés **uno solo** que represente toda la pregunta.
   MiniLM tiene una capa interna que combina esos 6 vectores en 1 (a grandes
   rasgos, un promedio ponderado — esto se llama "pooling"). Resultado:
   **un solo vector de 384 números**, que representa
   `"¿Quién gana Partido 1?"` completa.

```
"¿Quién gana Partido 1?" → [0.21, 0.05, -0.18, ..., 0.09]   (384 números, UN vector)
```

- _sentence embedding_, frase o pregunta que resume en un vector - id con
  su vector

## Validación del tutor

**Lo que entendiste bien, y sólido:** la distinción entre id (arbitrario,
de Fase 1) y embedding (vector con significado geométrico, de Fase 2).
Identificaste solo, sin pista previa, que la tabla de embeddings "tiene que
ver con los pesos" — correcto y buen instinto. El cierre de tus apuntes,
"sentence embedding, frase o pregunta que resume en un vector", es la
definición correcta y precisa del concepto central de la fase.

**Matiz a reforzar:** en tus apuntes la pregunta *"¿cómo saber que el
token gana está cerca del token Colombia?"* queda sin la respuesta directa
escrita al lado. La respuesta central: **no se sabe de antemano ni se
decide** — es una consecuencia observable *después* del entrenamiento. Solo
se mide (con similitud coseno) el resultado ya entrenado; no se predice qué
va a quedar cerca de qué antes de que el entrenamiento ocurra. Importante
para no asumir en Fase 3/5 que uno puede "razonar" de antemano la geometría
de un modelo pre-entrenado — a veces el resultado empírico sorprende (como
pasó con "México" en el BPE de Fase 1, casi sin fusionar contra la
intuición).

**Para conectar con lo que viene:** los apuntes no incluyen el hallazgo de
deduplicación (6582 filas → 442 preguntas únicas) ni el caso de texto
idéntico entre jornadas distintas. Relevante para Fase 3, donde conviene
indexar por `pregunta_id` y no por texto crudo, por este mismo motivo.
