# Tutor — Fase 0: Setup y datos

## Explicación del tutor

**¿Es necesario tener data para un LLM+RAG, aunque sea "mini"?**

Sí, pero en este proyecto la palabra "data" cumple dos roles distintos que conviene no mezclar:

- **Data de entrenamiento del LLM** (Fase 4): el corpus enseña al modelo *patrones del idioma*. Con solo 6582 filas, un LLM entrenado desde cero producirá texto con "sabor" al corpus, pero no lenguaje fluido general — limitación esperada y aceptada en `PROYECTO_BASE.md` (modelo de 1-10M parámetros, no se busca fluidez tipo GPT).
- **Data del RAG** (Fase 5): el corpus actúa como *base documental* consultada en el momento (retrieval), sin que el modelo la memorice en sus pesos. Aquí el volumen bajo importa menos — no se trata de "aprender idioma" sino de indexar y recuperar hechos.

La analogía de "enseñar un idioma sin mostrar ninguna palabra" aplica de lleno al primer rol, pero no tanto al segundo. Distinción importante antes de Fase 4, para no interpretar un LLM con generación pobre como un fallo del proyecto — es el resultado esperado dado el tamaño del corpus, y es justo lo que el RAG compensa.

**Dato curioso / industria:** GPT-3 se entrenó con ~300 mil millones de tokens; este corpus tiene ~6582 registros cortos. Cuando el corpus propio es pequeño, la técnica estándar en la industria es exactamente el enfoque de este proyecto: (1) usar RAG para compensar lo que el modelo no puede memorizar, y (2) aceptar que un LLM entrenado desde cero con poco dato será de nicho/estilo, no generalista. Coherente con que Fase 2 use un modelo de embeddings *pre-entrenado* (`all-MiniLM-L6-v2`) en vez de entrenar embeddings desde cero, que sí requeriría un corpus enorme.

**Sobre el corpus como merge de hojas:** el diseño es `PUNTUACIONES` (base, ya tiene respuesta + resultado oficial + puntos por fila) enriquecida vía merge (`how="left"`) con `PREGUNTAS` (por `pregunta_id`), `PARTICIPANTES` (por `email`, no por `id` — detalle importante del esquema real) y `JORNADAS` (por `jornada_id`), implementado en `construir_corpus()` de [`src/data_prep.py`](../../src/data_prep.py).

## Data para el LLM vs. data para el RAG (explicado sin tecnicismos)

Piensa en dos personas distintas ayudándote con la Polla Mundial:

**Persona 1: el "aprendiz de idioma"** (esto es el LLM)
Es alguien que nunca ha visto cómo se habla de fútbol, y tú le vas a enseñar
leyéndole tus 6582 filas una y otra vez, para que agarre el estilo: cómo se
formulan las preguntas, cómo suenan los nombres, qué palabras van juntas.
Con el tiempo, esta persona empieza a "sonar" como tu polla — pero el
conocimiento se le queda grabado en la cabeza, mezclado, no como una lista
que puede repasar. El problema es que 6582 filas es muy poquito para que
alguien aprenda a hablar bien un idioma completo — es como si alguien
intentara aprender español leyendo solo 6582 frases sueltas: va a captar
un poco el ritmo y el vocabulario, pero no va a hablar con fluidez ni te
va a poder citar datos exactos de memoria, porque nunca fue su trabajo
memorizar, sino "agarrar la onda" del idioma.

**Persona 2: el "bibliotecario"** (esto es el RAG)
Es alguien que no necesita aprender a hablar como tu polla. Su trabajo es
mucho más simple: tiene todas tus 6582 filas ordenadas en fichas, y cuando
le preguntas algo, va corriendo, busca la ficha exacta que responde tu
pregunta, y te la trae. No necesita "sonar" a nada ni haber memorizado
nada — solo necesita que sus fichas estén bien organizadas para
encontrarlas rápido. Por eso, aunque tengas pocas fichas (poca data), el
bibliotecario igual puede hacer bien su trabajo: mientras la ficha exista
y esté bien archivada, la va a encontrar y te la va a traer sin problema.

**Por qué importa la diferencia:**
Si solo pensaras en "necesito data" como una sola cosa, cuando llegues a
construir a la Persona 1 (el LLM, en la Fase 4) y notes que no habla muy
bien o se le olvidan cosas, podrías pensar que el proyecto salió mal. Pero
en realidad eso es totalmente esperado — es lo normal cuando el "aprendiz"
tuvo pocas frases para practicar. La jugada de este proyecto es justamente
no depender solo del aprendiz: cuando el aprendiz (LLM) no recuerde bien
algo, el bibliotecario (RAG) le va a pasar la ficha correcta para que la
use como apoyo al responder. Por eso el RAG "rescata" al LLM pequeño: uno
sabe poco de memoria, pero el otro tiene todo bien guardado y listo para
consultar.

## Resumen del usuario

Fase 0:
En este caso tenemos una fuente de información que es un excel con una baja data de volumen de información, aqui viene una pregunta, para lo que queremos hacer, que es un mini LLM con un RAG, ¿es necesario tener data?.
Ahora, con respecto a la data, nosotros creamos un corpus.

Un poco de teoria:

Un **corpus** es simplemente una colección grande de textos que se usa para estudiar o entrenar algo con lenguaje. Es como una "biblioteca" organizada de ejemplos de texto real.
Por ejemplo: imagina que quieres enseñarle a alguien a reconocer cómo habla la gente de un país. Le darías miles de cartas, noticias y conversaciones reales de ese país. Esa colección completa de textos es el corpus.
En inteligencia artificial, un corpus es el conjunto de textos (libros, artículos, conversaciones, etc.) que se usa para que un modelo "aprenda" el idioma y sus patrones.
Un corpus puede ser de un solo tema (por ejemplo, solo noticias de deportes) o puede mezclar muchos temas distintos (noticias, cuentos, conversaciones, recetas, etc.).
Depende de para qué lo quieras usar:
- Si quieres que algo "hable de todo un poco" (como un asistente general), necesitas un corpus **variado**, con muchos temas mezclados.
- Si quieres que algo sea experto en un tema específico (por ejemplo, solo medicina, o solo derecho), usas un corpus **especializado**, enfocado solo en ese tema.
Ejemplo cotidiano: si quieres formar a alguien como "cultura general", le das libros de historia, ciencia, cocina y deportes. Pero si quieres que sea especialista en cocina, le das solo libros de cocina. Ambos son corpus válidos, solo cambia el propósito.

Ahora, para nuestro caso, tenemos diferentes hojas en el Excel: PARTICIPANTES, JORNADAS, PREGUNTAS, RESPUESTAS, PUNTUACIONES, TABLA_GENERAL, RESULTADOS_OFICIALES, por lo que podemos unificar y crear un corpus central, por lo que: la hoja PUNTUACIONES ya tiene, por fila, la respuesta de un participante a una pregunta junto con el resultado oficial y los puntos obtenidos. Le faltan textos legibles (el enunciado de la pregunta, el nombre del participante, el nombre de la jornada), que viven en otras hojas. Se cruzan (merge) por sus respectivos ids para completar esos
textos. Con esto creamos un corpus robusto que simplifica y resumen nuestra data del torneo.

Respondiendo a la pregunta: ¿Es necesario tener data?, respuesta:
Sí, es indispensable — un LLM (grande o mini) **no se puede crear desde cero sin datos**, porque justamente lo que hace el modelo es "aprender" patrones de lenguaje a partir de ejemplos reales de texto (eso es el corpus del que hablábamos). Es como querer enseñarle a alguien a hablar un idioma sin nunca haberle mostrado ni una palabra de ese idioma: es imposible, no tiene de dónde aprender.

## Validación del tutor

**Lo que entendiste bien:**
- La definición de corpus (variado vs. especializado) es correcta y bien ejemplificada.
- El mecanismo técnico del merge (`PUNTUACIONES` como base + textos legibles de las otras hojas cruzados por id/email) es preciso y coincide con el pipeline real.
- La respuesta "sí, es indispensable" a la pregunta de si se necesita data está bien fundamentada con la analogía del idioma.

**Lo que falta o es impreciso:**
- El resumen trata "data para el LLM" y "data para el RAG" como la misma necesidad con la misma exigencia de volumen. No lo es: el LLM entrenado desde cero con 6582 filas sí tendrá limitaciones notorias de fluidez (bajo volumen), mientras que el RAG puede funcionar razonablemente bien con ese mismo volumen porque no depende de "aprender idioma" sino de indexar y recuperar. Sin esta distinción, en Fase 4 se podría interpretar un LLM con generación pobre como un fallo del proyecto, cuando es el resultado esperado — el RAG es justamente lo que compensa esa debilidad.
- No se menciona el volumen concreto (6582 registros, 442 preguntas, 18 participantes) como referencia numérica para evaluar resultados en Fase 4.

**Para profundizar:**
- En Fase 1 (tokens), revisitar la pregunta "¿alcanza este volumen?" en términos de *cantidad de tokens totales*, no de "filas" — esa es la unidad real que importa para entrenar.
