# Fase 4 — Panorama: qué es un Transformer

## La pregunta que resuelve toda la arquitectura

Un LLM, en el fondo, hace una sola cosa: dada una secuencia de tokens,
predecir **qué token viene después**. Todo el diseño del transformer es
maquinaria para hacer bien esa predicción.

El problema de fondo es: para predecir el siguiente token, el modelo
necesita "entender" el contexto — no solo qué palabras hay, sino cómo se
relacionan entre sí. "El equipo que **ganó** la jornada 3" — para predecir
bien lo que sigue, el modelo necesita que la palabra "ganó" preste atención
a "equipo" y a "jornada 3", no tratarlas como una bolsa de palabras sueltas.

**Dato curioso:** antes de 2017, el estado del arte para esto eran las
RNN/LSTM — redes que procesan la secuencia palabra por palabra, en orden,
arrastrando un "estado" de memoria. El problema: son lentas (no se
paralelizan, porque el paso *t* depende del paso *t-1*) y les cuesta
recordar dependencias lejanas (la palabra 1 y la palabra 200 casi no se
"ven"). El paper *"Attention Is All You Need"* (Google, 2017) propuso
eliminar la recurrencia por completo y reemplazarla con **self-attention**:
cada token mira directamente a todos los demás tokens de la secuencia en un
solo paso, sin cadena secuencial. Eso permitió paralelizar el entrenamiento
en GPU y es, literalmente, el origen de la "T" en GPT (*Generative
Pre-trained **Transformer***).

## El flujo de datos, de principio a fin

```
texto → [tokenizer BPE, Fase 1]  → IDs de tokens (enteros)
      → [token embedding]         → vector por token (aprendido)
      → [+ positional encoding]   → vector que ahora sabe "dónde" está
      → [N × TransformerBlock]    → vectores enriquecidos con contexto
      → [cabeza de salida]        → distribución de probabilidad sobre el vocabulario
      → siguiente token
```

Cada `TransformerBlock` (se apilan varios — 4-6 capas según el roadmap, por
las restricciones de hardware del proyecto) tiene esta forma interna:

```
entrada
  │
  ├──────────────┐
  ▼              │  (conexión residual)
Multi-Head        │
Self-Attention     │
  │              │
  + ◄────────────┘
  │
LayerNorm
  │
  ├──────────────┐
  ▼              │  (conexión residual)
Feed-Forward       │
(MLP)              │
  │              │
  + ◄────────────┘
  │
LayerNorm
  │
  ▼
salida (misma forma que la entrada)
```

Piezas y su rol, en una frase cada una:

| Pieza | Rol |
|---|---|
| **Token embedding** | Convierte cada ID de token en un vector que la red puede procesar y **aprender a ajustar** durante el entrenamiento (a diferencia de MiniLM en Fase 2, que ya venía entrenado). |
| **Positional encoding** | El self-attention por sí solo no distingue orden — "Colombia venció a Brasil" y "Brasil venció a Colombia" verían el mismo conjunto de tokens. Esto inyecta la posición. |
| **Self-attention** | Cada token calcula cuánto debe "atender" (prestar atención) a cada otro token, y mezcla información en proporción a esa atención. Es el corazón del modelo. |
| **Causal masking** | Para generar texto token a token, un token no puede mirar tokens *futuros* — si pudiera, estaría haciendo trampa (viendo la respuesta antes de "predecirla"). |
| **Multi-head attention** | En vez de una sola forma de atender, varias "cabezas" en paralelo aprenden a fijarse en relaciones distintas (sintaxis, correferencia, etc.). |
| **Feed-forward (MLP)** | Después de mezclar información entre tokens (attention), cada token procesa esa información individualmente con una red densa — la parte que añade capacidad de transformación no lineal. |
| **LayerNorm + residuales** | No aprenden "significado" — estabilizan el entrenamiento. Sin ellos, redes de varias capas son muy difíciles de entrenar (gradientes que explotan o desaparecen). |

**Cómo lo hace la industria vs. lo que se hizo aquí:** GPT-3 tiene 96
capas, 96 cabezas de atención, embeddings de 12,288 dimensiones, 175B
parámetros, entrenado en miles de GPUs durante semanas. Este modelo tiene 4
capas, 4 cabezas, embeddings de 128 dimensiones, ~864K parámetros,
entrenado en CPU en minutos. La arquitectura es **exactamente la misma
fórmula matemática** — lo único que cambia es la escala. Es quizás el punto
pedagógico más valioso de la fase: GPT-3, GPT-4, Llama, etc. no son "otra
cosa" — son este mismo bloque, repetido y agrandado.

## Qué es un tensor

Un **tensor** es la misma idea de escalar/vector/matriz, generalizada a
cualquier número de dimensiones:

```
0 dimensiones → un número solo         → escalar         5
1 dimensión   → una lista de números   → vector          [5, 12, 3]
2 dimensiones → una tabla de números   → matriz          [[5, 12, 3],
                                                            [8,  1, 9]]
3+ dimensiones → una tabla de tablas   → tensor (n-D)    [[[...],[...]], [[...],[...]]]
```

"Tensor" es el nombre genérico que usan las librerías de deep learning
(PyTorch, TensorFlow) para "arreglo de números de cualquier número de
dimensiones". Un escalar es un tensor de 0 dimensiones, un vector es un
tensor de 1 dimensión, una matriz es un tensor de 2 dimensiones.

**Con datos propios:** un lote de 3 preguntas rellenadas con `<PAD>` hasta
el mismo largo es un tensor 2D (una matriz):

```python
torch.tensor([
    [45,  12,  89,  0,   0 ],
    [66,  71,  13, 253, 64],
    [201,  0,   0,  0,   0 ],
]).shape   # torch.Size([3, 5])  →  3 filas, 5 columnas
```

`shape` (o "forma") dice el tamaño en cada dimensión. Cuando ese tensor de
IDs pase por `nn.Embedding`, cada número se reemplaza por su **vector** de
`d_model` dimensiones, y el tensor gana una dimensión más:

```
Antes (IDs):        shape [3, 5]        →  3 preguntas, 5 posiciones, 1 id cada una
Después (vectores):  shape [3, 5, 128]  →  3 preguntas, 5 posiciones, 128 números por posición
```

Entender qué shape entra y qué shape sale de cada pieza es, en la práctica,
la mitad de "entender" una arquitectura de deep learning — es el hilo que
se siguió en cada paso de esta fase.

**Por qué existen los tensores (y no simplemente "listas de listas" de
Python):** un tensor de PyTorch es una estructura optimizada en C++/CUDA
que permite operaciones matemáticas sobre millones de números en paralelo,
en CPU o GPU. Además, PyTorch rastrea automáticamente las operaciones que
se le aplican a un tensor para poder calcular gradientes después — la base
de todo el entrenamiento (paso 7).

## Qué significa "vector" cuando no son 3 componentes (x, y, z)

Un vector en física (x, y, z) tiene 3 componentes porque el espacio físico
tiene 3 dimensiones — se puede dibujar como una flecha. En este proyecto no
se describen posiciones en el espacio físico, sino **significado de una
palabra** — algo mucho más rico, así que se usan muchos más números: 128 en
este caso (dentro del rango 128-256 del roadmap).

```
vector en 2D:      (x, y)                    → 2 componentes
vector en 3D:      (x, y, z)                 → 3 componentes
vector "Colombia": (n1, n2, ..., n128)       → 128 componentes
```

No hay ninguna regla que diga "un vector debe tener 3 componentes" — es
solo el caso particular del espacio físico. Cada uno de los 128 números es,
conceptualmente, una "dimensión de significado" abstracta — durante el
entrenamiento, el modelo decide por sí solo qué representa cada una, sin
que se le diga explícitamente. Con pocas dimensiones no habría suficiente
"espacio" para capturar los matices de significado entre miles de palabras
distintas (Fase 2, con MiniLM, usaba 384; aquí se usan 128 por las
restricciones de hardware del proyecto).

## Siguiente paso

[01-embedding-y-posicion.md](01-embedding-y-posicion.md) — cómo un id de
token se convierte en un vector, y cómo se le inyecta la noción de
posición.
