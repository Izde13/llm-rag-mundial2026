# Tutor — Fase 4: LLM propio desde cero

Fase construida en Modo GUIAR, muy despacio y paso a paso, con verificación
en código real contra el corpus del mundial en cada pieza — esta es la fase
más densa del proyecto (arquitectura transformer completa desde cero en
PyTorch), así que la explicación de cada paso quedó en su propio archivo en
vez de condensarse, para no perder las analogías, los ejemplos numéricos
calculados a mano y las dudas puntuales que se resolvieron en el camino.

## Índice de la sesión, paso a paso

| Archivo | Contenido |
|---|---|
| [fase-4/00-panorama.md](fase-4/00-panorama.md) | Qué es un Transformer, por qué reemplazó a RNN/LSTM, el flujo de datos completo, qué es un tensor, qué significa "vector" con 128 componentes |
| [fase-4/01-embedding-y-posicion.md](fase-4/01-embedding-y-posicion.md) | Token embedding aprendido (vs. pre-entrenado de Fase 2), positional encoding, por qué self-attention necesita ayuda para distinguir orden, verificado con `"Roja Partido"` |
| [fase-4/02-self-attention.md](fase-4/02-self-attention.md) | La analogía Q/K/V, por qué el entrenamiento hace emerger relaciones tipo "tarjeta↔amarilla", la fórmula completa verificada a mano sobre `"tarjeta amarilla"` |
| [fase-4/03-causal-masking.md](fase-4/03-causal-masking.md) | Por qué un modelo generativo no puede "espiar" el futuro durante el entrenamiento, la máscara triangular, verificado con y sin máscara sobre `"la tarjeta amarilla"` |
| [fase-4/04-multi-head-attention.md](fase-4/04-multi-head-attention.md) | Por qué una sola cabeza no basta, varias vistas en paralelo, verificado con 2 cabezas mostrando repartos de atención distintos |
| [fase-4/05-feedforward-residual-layernorm.md](fase-4/05-feedforward-residual-layernorm.md) | Por qué attention solo "mezcla" y feed-forward "transforma", conexión residual (no perder información al apilar capas), layer norm (estabilidad numérica) |
| [fase-4/06-transformer-block.md](fase-4/06-transformer-block.md) | Ensamblado del bloque completo y apilado de 4 capas, verificado con shapes y conteo de parámetros sobre el corpus real |
| [fase-4/07-entrenamiento.md](fase-4/07-entrenamiento.md) | Cabeza de salida, cross-entropy loss, backpropagation, hiperparámetros decididos y resultado real del entrenamiento (loss 2.29→0.35 en 5 épocas) |
| [fase-4/08-generacion.md](fase-4/08-generacion.md) | Generación autoregresiva, greedy vs. sampling con temperatura, resultado real de texto generado por el modelo entrenado |

## Resumen del usuario

- Transformer
- Un LLM, en el fondo, hace una sola cosa: dada una secuencia de tokens, predice **qué token viene después**.
- El problema de fondo es: para predecir el siguiente token, el modelo necesita "entender" el contexto
- **Dato curioso:** antes de 2017, el estado del arte para esto eran las RNN/LSTM — redes que procesan la secuencia palabra por palabra, en orden, arrastrando un "estado" de memoria. El problema: son lentas (no se paralelizan, porque el paso _t_ depende del paso _t-1_) y les cuesta recordar dependencias lejanas (la palabra 1 y la palabra 200 casi no se "ven"). El paper _"Attention Is All You Need"_ (Google, 2017) propuso eliminar la recurrencia por completo y reemplazarla con **self-attention**: cada token mira directamente a todos los demás tokens de la secuencia en un solo paso, sin cadena secuencial. Eso permitió paralelizar el entrenamiento en GPU y es, literalmente, el origen de la "T" en GPT (_Generative Pre-trained **Transformer**_).
- En Fase 2 usaste sentence-transformers con el modelo all-MiniLM-L6-v2: una red neuronal que alguien más ya entrenó (con billones de palabras de internet) y que tú solo usaste — le diste texto, te devolvió vectores. Los pesos de esa red están congelados; tú nunca los tocaste ni los ajustaste. Es como usar una calculadora ya hecha.
- En Fase 4, el "embedding" va a ser una simple tabla de números inicializados al azar (nn.Embedding en PyTorch — literalmente una matriz [vocab_size, dim]). Al principio esos vectores no significan nada — son ruido. La diferencia está en lo que pasa después: durante el entrenamiento, cuando el modelo se equivoca prediciendo el siguiente token, el algoritmo de backpropagation ajusta esa misma tabla (además de todos los demás pesos del transformer) para que, poco a poco, vectores de tokens que aparecen en contextos parecidos terminen pareciéndose entre sí — igual que en Fase 2, pero en vez de heredarlo ya hecho, lo vas a construir tú desde cero, entrenándolo con tu propio corpus.
- Tensor: **"arreglo de números de cualquier número de dimensiones"**
- Lo que entiendo hasta el momento, tenemos un corpus, luego se hace una tokenizacion de ese corpus con BPE donde vamos a fusionar, con lo de los pares y empezar a crear token de acuerdo a lo que se vaya repitiendo, hasta este momento, entonces cada token tendra su id y cada oracion se arma con los id de los tokens y se rellena con PAD, qeu es id 0
- Lo que entiendo, la tabla de posiciones la creaste teniendo en cuenta la oracion mas larga, , luego pasamos la oracion, que al vector del token le suma la de posicion
- Para que el modelo entienda bien qué significa "amarilla" aquí, necesita relacionarla con "tarjeta" (no es un color cualquiera, es parte de una tarjeta) y con "primera" (importa el orden temporal). Self-attention es el mecanismo que le permite a cada palabra **"mirar" a las demás palabras de la oración y decidir cuánto tomar de cada una**.
- Causal masking es tapar la palabra siguiente
- multi-head attention, son varios self-attention pero orientados de diferentes temas
- feed-forward: **cada palabra, por separado, "piensa" un rato sobre la información que ya tiene**, sin mirar a las demás palabras (a diferencia de attention, que sí mira a todas).
- La pieza que hace falta en el medio: ReLU: Entre "estirar" y "comprimir" hay un paso simple llamado ReLU. Es una regla mínima: "si un número es negativo, cámbialo a 0; si es positivo, déjalo igual".
- Conexión residual: En vez de "la salida de esta capa reemplaza la entrada", se hace: **"la salida de esta capa se le suma a la entrada"**.
- Layer normalization hace exactamente eso con el vector de cada palabra, después de cada suma residual: toma los números tal como están y los reescala para que, en promedio, ronden 0 y tengan una dispersión estándar — sin cambiar la relación entre ellos (cuál es más grande que cuál), solo llevándolos a un rango manejable.
- logits?

## Validación del tutor

**Qué entendiste bien:**

- Transformer y por qué reemplazó a RNN/LSTM — el punto de fondo (predecir
  el siguiente token, el cuello de botella de la recurrencia secuencial,
  self-attention como solución paralelizable) está bien capturado.
- Embedding Fase 2 vs. Fase 4 — la distinción "calculadora ya hecha"
  (MiniLM, congelado) vs. "tabla que empieza en ruido y se ajusta con
  backpropagation" es precisa; fue el matiz que más costó asentar en la
  conversación y quedó bien resuelto.
- Tensor — "arreglo de números de cualquier número de dimensiones" es la
  definición correcta, sin el error común de pensar que un vector debe
  tener 3 componentes.
- Self-attention — el ejemplo de "amarilla" relacionándose con "tarjeta" y
  "primera" es el mismo usado en la sesión, y la idea de "mirar a las demás
  palabras y decidir cuánto tomar de cada una" está bien capturada.
- Feed-forward y ReLU — ambas definiciones correctas y con las palabras
  justas ("piensa sobre la información que ya tiene, sin mirar a las
  demás"; la regla de ReLU tal cual).
- Conexión residual y layer norm — ambas correctas y precisas, sin perder
  matices (layer norm: "reescala sin cambiar la relación entre los
  números").
- BPE → ids → PAD — el flujo de armar cada oración con los ids de los
  tokens y rellenar con PAD (id 0) hasta igualar largo está bien entendido.

**Imprecisiones a corregir:**

1. *"La tabla de posiciones la creaste teniendo en cuenta la oración más
   larga"* — no quedó resuelto del todo. La tabla se crea **una sola vez**,
   con un tamaño máximo fijo del proyecto (`max_seq_len=64`, decidido de
   antemano), no en función de ninguna oración particular — cada oración
   usa tantas filas de esa tabla como tokens tenga. Consecuencia si queda
   así: en Fase 5 (RAG), con contexto recuperado de largo variable
   inyectado en el prompt, podría pensarse que hace falta "recrear" algo
   cada vez — no es así, la tabla ya está fija y lista para cualquier largo
   hasta 64. Ver [`fase-4/01-embedding-y-posicion.md`](fase-4/01-embedding-y-posicion.md),
   sección "tabla fija vs. lo que se usa por oración".
2. *"Causal masking es tapar la palabra siguiente"* — simplificación que se
   queda corta: no tapa solo la palabra inmediatamente siguiente, tapa
   **todas** las posiciones futuras a la vez (el triángulo completo).
   Consecuencia: al razonar sobre por qué la primera palabra de una oración
   tiene "100% de atención a sí misma" (visto en el ejemplo real), "tapar
   solo la siguiente" no cuadraría — "tapar todo el futuro" sí. Ver
   [`fase-4/03-causal-masking.md`](fase-4/03-causal-masking.md).
3. *"Multi-head attention son varios self-attention orientados a diferentes
   temas"* — dirección correcta, matiz importante: nadie orienta las
   cabezas a un tema de antemano. Cada cabeza tiene pesos aleatorios
   independientes, y la especialización emerge sola durante el
   entrenamiento, por el mismo mecanismo de "reduce el error" ya entendido
   para self-attention simple. Consecuencia: podría buscarse cómo asignar
   un tema a mano a cada cabeza más adelante, cuando la arquitectura solo
   da la *libertad* de especializarse, no una instrucción. Ver
   [`fase-4/04-multi-head-attention.md`](fase-4/04-multi-head-attention.md).

**Pregunta pendiente resuelta: "¿logits?"** — números crudos que salen de
la última capa lineal (`cabeza_salida`), antes de softmax, uno por cada uno
de los 254 símbolos del vocabulario. Pueden ser negativos, no suman 1, no
son probabilidades todavía — softmax los convierte en porcentajes reales.
Mismo patrón visto dos veces en la fase: en self-attention (`scores` antes
de softmax) y aquí en la salida final (`logits` antes de softmax). Ver
[`fase-4/07-entrenamiento.md`](fase-4/07-entrenamiento.md), Parte A.

**Para profundizar (conecta con Fase 5):**

1. Backpropagation + `loss.backward()` + `optimizador.step()`
   ([`fase-4/07-entrenamiento.md`](fase-4/07-entrenamiento.md), Parte C) no
   aparece en el resumen — es el mecanismo que da sentido a todo lo demás
   (por qué las cabezas se especializan, por qué tarjeta↔amarilla emerge,
   por qué el embedding deja de ser ruido).
2. El resultado real de generación (greedy cayendo en bucles, sampling con
   temperatura variando coherencia,
   [`fase-4/08-generacion.md`](fase-4/08-generacion.md)) tampoco aparece —
   relevante para decidir en Fase 5 con qué estrategia de generación
   combinar el retrieval.

## Resultado real de esta sesión

Arquitectura completa implementada y verificada en
[`src/transformer_model.py`](../../src/transformer_model.py)
(`EmbeddingConPosicion`, `SelfAttention` con causal masking opcional,
`MultiHeadAttention`, `FeedForward`, `TransformerBlock`, `TransformerLM`),
entrenada con [`src/train.py`](../../src/train.py) (864,510 parámetros,
5 épocas, loss_train 2.29→0.35, loss_val 1.12→0.33, sin señal de
overfitting) y usada para generar texto con
[`src/generar.py`](../../src/generar.py) (greedy y sampling con
temperatura), reconociendo las plantillas del corpus y generando
variaciones nuevas sin memorizar preguntas exactas. Detalle técnico
completo (decisiones de diseño, comandos, resultados) en
[`docs/fase-4-llm-propio.md`](../fase-4-llm-propio.md); el recorrido
pedagógico completo, paso a paso, en la carpeta [`fase-4/`](fase-4/)
listada arriba.
