# Tutor — Fase 5: RAG completo

## Explicación del tutor

### ¿Qué es RAG?

El LLM de Fase 4 (`TransformerLM`) es un generador de texto puro: aprendió
a predecir "qué palabra sigue probablemente" a partir de los patrones que
vio en 6582 oraciones del corpus durante el entrenamiento. Con 864K
parámetros y 5 épocas, memorizar 442 pares pregunta→respuesta específicos
es prácticamente imposible — el modelo aprendió gramática superficial
("¿Quién gana Partido N?"), no los datos reales.

RAG resuelve esto sin tocar el modelo: antes de generar, se *busca* el dato
correcto en el corpus (retrieval) y se lo escribe explícitamente en el
texto de entrada (el "contexto"). Es la diferencia entre examen a libro
cerrado (Fase 4 sola) y examen a libro abierto (RAG): el modelo ya no tiene
que recordar el dato, solo leerlo de lo que se le puso adelante y
continuar el texto de forma coherente.

**RAG = Retrieval (buscar el dato correcto) + Augmented Generation
(generación aumentada con ese dato inyectado).**

### Por qué MiniLM y no el `TransformerLM` de Fase 4, para buscar

Búsqueda y generación son tareas distintas, resueltas por redes distintas:

- **MiniLM** (Fase 2) fue entrenado para que oraciones *parecidas en
  significado* queden *cerca* en un espacio vectorial de 384 dimensiones —
  su trabajo es comparar, no generar.
- **`TransformerLM`** (Fase 4) fue entrenado para predecir el siguiente
  token — su trabajo es generar, no comparar.

Además, el corpus ya está indexado en ChromaDB con vectores de MiniLM
(Fase 3): si la query se buscara con otro modelo, viviría en un espacio
vectorial distinto al del corpus indexado, y la comparación no significaría
nada (como buscar en el índice de un libro con el índice de otro).

### Por qué el LLM sigue aportando algo, aunque el buscador ya trajo el dato

Para preguntas literales de una sola fila ("¿quién ganó el Partido 11?"),
el buscador solo casi alcanza — el LLM aporta poco. El valor real aparece
cuando hay que **combinar** varios hechos recuperados en una respuesta
coherente (ej. "¿qué tal jugó Colombia en la jornada 2?", que necesita
tejer marcador + tarjetas + corners). El buscador da los materiales; el
LLM redacta.

Ejemplo de industria (soporte técnico con RAG, patrón real de
Intercom/Notion AI): 4 fragmentos de documentación dispersos sobre
expiración de API keys, ninguno responde solo la pregunta completa — un
LLM grande (GPT-4/Claude) los combina, infiere relaciones no explícitas
("el grace period evita el downtime si renovás a tiempo") y redacta una
respuesta nueva. Con el `TransformerLM` de este proyecto (unos pocos
millones de parámetros), ese paso de síntesis es real pero mucho más
modesto — el valor del RAG en este proyecto está sobre todo en el
**retrieval preciso**, no en la elocuencia de la generación.

### El límite real que apareció al construirlo

Con `modelo_fase4.pt` conectado directo a RAG, el LLM con frecuencia
**ignoraba el contexto inyectado** y continuaba hacia otra plantilla del
corpus sin relación. Se comprobó (con greedy y con sampling a varias
temperaturas) que no era ruido de aleatoriedad: el modelo aprendió, con
alta confianza en sus pesos, a continuar hacia ciertas plantillas porque
**nunca vio, durante el entrenamiento, la forma "contexto + pregunta"** —
Fase 4 entrenó con oraciones sueltas, no con este patrón de entrada.

La solución fue reentrenar desde cero (`train_rag.py`) con un dataset
sintético armado con la forma exacta de RAG (usando el propio retrieval de
Chroma para construir el contexto de cada ejemplo — texto 100% real, solo
reensamblado), con dos ejes de variedad (`k` de contexto variable, orden
barajado) y más capacidad (874K → 2.79M parámetros, dentro del rango
1-10M del proyecto). Mejoró de forma medible (loss_val de 1.06 a 0.30, y
el modelo dejó de saltar a plantillas completamente ajenas), pero **no
resolvió del todo** discriminar cuál línea exacta del contexto copiar
cuando hay varias con estructura casi idéntica — límite de arquitectura
(falta de un mecanismo de atención dirigida/copia tipo cross-attention),
no de datos o capacidad.

### Overfitting (por qué se agregó early stopping)

Overfitting es cuando el modelo deja de aprender el patrón general y
empieza a memorizar detalles específicos del set de entrenamiento — como
un estudiante que memoriza las respuestas exactas de un examen de práctica
en vez de entender el tema: brilla en lo que ya vio, no mejora en lo
nuevo. Se detecta comparando `loss_train` (lo que el modelo ve) contra
`loss_val` (datos que nunca ve): si `loss_val` deja de bajar o sube
mientras `loss_train` sigue bajando, es la señal.

Se dio en la corrida con más capacidad (2.79M parámetros) y 20 épocas
sobre el mismo corpus base de 442 preguntas: `loss_val` tocó mínimo
alrededor de la época 16-18 y después subió levemente. La causa: más
parámetros dan más "espacio" para memorizar, y sin proporcionalmente más
datos genuinos ese espacio extra se llena memorizando en vez de
generalizando. La solución (patrón estándar de la industria, *early
stopping*) fue guardar en cada época una copia del `state_dict` si
`loss_val` mejoraba, y persistir esa copia al final en vez de los pesos de
la última época.

## Resumen del usuario

(No se pegó un resumen de apuntes en esta sesión — se guió la fase en
modo GUIAR de principio a fin, con preguntas de aclaración conceptual en
el camino: qué es RAG, por qué MiniLM y no el LLM propio para buscar, un
ejemplo de industria para ver el valor real de la generación, qué es
overfitting y por qué apareció.)

## Validación del tutor

No aplica en esta sesión (no se cerró con modo CERRAR ni se pegó un
resumen para validar). Si en una sesión futura se quiere cerrar la fase
formalmente, invocar de nuevo `/tutor-fase` en modo CERRAR con un resumen
de lo aprendido.
