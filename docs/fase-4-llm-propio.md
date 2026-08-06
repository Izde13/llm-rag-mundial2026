# Fase 4 — LLM propio desde cero

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): construir
la arquitectura transformer completa (self-attention, multi-head attention,
positional encoding, layer norm, feed-forward, causal masking) en PyTorch
puro, sin la librería `transformers` de HuggingFace, y entrenarla desde cero
sobre el corpus del mundial hasta que genere texto con "sabor" al corpus.

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/bpe_vocab.py`](../src/bpe_vocab.py) | Vocabulario BPE (Fase 1) → ids enteros, con tokens `<PAD>`/`<UNK>` |
| [`src/transformer_model.py`](../src/transformer_model.py) | Arquitectura completa: embedding+posición, self-attention, causal masking, multi-head attention, feed-forward, `TransformerBlock`, `TransformerLM` |
| [`src/train.py`](../src/train.py) | Preparación de datos (entrada/objetivo desplazados) y loop de entrenamiento |
| [`src/generar.py`](../src/generar.py) | Generación de texto autoregresiva (greedy y sampling con temperatura) |

## 1. De símbolos BPE a ids enteros (`src/bpe_vocab.py`)

El tokenizer BPE de Fase 1 (`tokenizer_bpe.py`) tokeniza texto a una lista de
**símbolos-string** (`"Marcador</w>"`, `"vs</w>"`, ...), pero `nn.Embedding` de
PyTorch necesita **ids enteros** para indexar su tabla de vectores. Faltaba
la capa de mapeo string↔id que Fase 1 nunca necesitó (solo comparaba
símbolos, nunca los pasó por una red neuronal).

`construir_vocab_ids(merges, textos)` recorre el corpus tokenizado con BPE y
asigna un id a cada símbolo distinto, reservando dos ids especiales:

- **`<PAD>` (id 0):** relleno para igualar la longitud de secuencias dentro
  de un mismo lote — el entrenamiento procesa varias oraciones a la vez como
  un tensor rectangular, y oraciones de largo distinto necesitan rellenarse
  hasta el máximo del lote.
- **`<UNK>` (id 1):** salvaguarda para un símbolo no visto al construir el
  vocabulario (no debería activarse con el propio corpus, pero es la
  práctica estándar de la industria).

`codificar_ids`/`decodificar_ids` son el camino ida y vuelta, reutilizando
`tokenizar_con_merges` de Fase 1 (los merges se cargan desde
`tokenizer_bpe_stats.json`, sin reentrenar BPE).

**Resultado:** 254 ids (252 símbolos BPE + 2 especiales).

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/bpe_vocab.py
```

## 2. Arquitectura transformer (`src/transformer_model.py`)

Construida pieza por pieza, en el mismo orden en que aparecen en un bloque
transformer real. Todas las piezas preservan la forma `[batch, seq_len,
d_model]` de entrada a salida, lo que permite encadenarlas sin reajustar
tamaños.

### `EmbeddingConPosicion`

Dos tablas `nn.Embedding` (lookup: la fila *i* es el vector del índice *i*),
sumadas elemento a elemento:

- `token_embedding[id]` — qué token es (aprendido durante el entrenamiento,
  a diferencia de MiniLM en Fase 2, que ya venía entrenado).
- `position_embedding[i]` — en qué posición de la secuencia está.

Se eligió posicional **aprendido** (tabla indexada por posición, mismo
mecanismo que el embedding de tokens) en vez de la fórmula sinusoidal fija
del paper original de 2017 — es el patrón usado por GPT-2 en adelante, y
reutiliza un concepto ya construido en vez de introducir trigonometría.
Necesario porque self-attention, por diseño, es *permutation-invariant*: sin
esta suma, "Colombia venció a Brasil" y "Brasil venció a Colombia" serían
indistinguibles para el modelo.

### `SelfAttention`

```
Attention(Q,K,V) = softmax( Q·Kᵀ / √d_model ) · V
```

Tres transformaciones lineales (`W_query`, `W_key`, `W_value`) sobre el
mismo vector de entrada producen la "pregunta", el "cartel" y el
"contenido" de cada token. `Q·Kᵀ` compara cada pregunta contra cada cartel;
dividir por `√d_model` es estabilidad numérica (evita que softmax se sature
con `d_model` grande); softmax convierte los puntajes, fila por fila, en un
reparto de atención que suma 1.0; ese reparto pondera el promedio de los
`V` — Q y K solo deciden el reparto, nunca aparecen en el resultado directo.

Verificado a mano: el cálculo de `Attention("tarjeta", "amarilla")` con
`d_model=4` coincide número por número entre el cálculo manual y la salida
del módulo.

### Causal masking (parámetro `causal` de `SelfAttention`)

A cada posición se le prohíbe mirar posiciones futuras: antes del softmax,
se rellenan con `-inf` las celdas de `scores` por encima de la diagonal
(`torch.triu(..., diagonal=1)`) — softmax les asigna 0% de atención sin
tocar la fórmula. Necesario para entrenar un modelo generativo: sin esto, la
predicción de la posición *i* podría "espiar" tokens que en la generación
real todavía no existirían — la diferencia entre un modelo autoregresivo
(GPT) y uno bidireccional (BERT).

### `MultiHeadAttention`

Varias cabezas de atención en paralelo, cada una con su propia porción de
dimensiones (`d_k = d_model / num_heads`), concatenadas al final y
mezcladas con una capa lineal de salida. Implementado con una sola matriz
grande por Q/K/V "cortada" en pedazos por cabeza (`.view().transpose()`) —
matemáticamente equivalente a `num_heads` instancias independientes de
`SelfAttention`, pero una sola operación vectorizada en vez de un bucle.

Cada cabeza, al tener pesos propios, tiene libertad de especializarse en un
tipo de relación distinto durante el entrenamiento (gramatical, temática,
posicional) — una sola cabeza fuerza a todas esas señales a competir en un
solo reparto.

### `FeedForward`

Estira el vector a un espacio más grande (`d_ff`, 4× `d_model`, la misma
proporción del paper original y de GPT), aplica ReLU, y comprime de vuelta.
Aplicado a cada posición por separado — nunca mezcla información entre
posiciones, a diferencia de attention. Sin ReLU, dos capas lineales
seguidas colapsan matemáticamente en una sola transformación lineal; ReLU
rompe esa linealidad y es lo que le da al modelo capacidad de aprender
relaciones que no son un simple promedio ponderado (el límite de lo que
attention, por sí solo, puede producir).

### `TransformerBlock`

```
x = LayerNorm(x + MultiHeadAttention(x))
x = LayerNorm(x + FeedForward(x))
```

**Conexión residual** (`x + submódulo(x)`, en vez de reemplazar): si una
capa no tiene nada útil que aportar para cierta posición, puede aprender a
producir ~0 y dejar pasar la información original casi intacta — necesario
para apilar varias capas sin perder información capa tras capa (idea
tomada de ResNet, 2015, adoptada por el paper de transformers de 2017).

**LayerNorm** tras cada suma: reescala los números (promedio ~0, dispersión
estándar) sin cambiar su orden relativo — mantenimiento de estabilidad
numérica para que el entrenamiento no se vuelva errático con varias capas
apiladas, no le agrega significado al vector.

### `TransformerLM`

Embedding + `num_layers` bloques apilados + cabeza de salida (`nn.Linear`
final que convierte el vector de cada posición en logits sobre los 254
símbolos del vocabulario).

**Cómo correrlo** (corre pruebas de cada pieza individual, con shapes y
parámetros sobre datos reales del corpus):

```bash
./.venv/Scripts/uv.exe run python src/transformer_model.py
```

## 3. Entrenamiento (`src/train.py`)

Tarea de entrenamiento: dado un fragmento de oración, predecir la siguiente
palabra (autoregresivo). El objetivo es la misma oración desplazada una
posición:

```
oración:   id0  id1  id2  id3
entrada:   id0  id1  id2        (x)
objetivo:       id1  id2  id3   (y)
```

Causal masking garantiza que la predicción de la posición *i* solo pudo
usar `id0..idi`, nunca el objetivo.

**Loop:** `logits = modelo(entrada)` → `CrossEntropyLoss(logits, objetivo)`
(con `ignore_index` en `<PAD>`, para que el relleno mecánico no cuente como
error) → `loss.backward()` (backpropagation: PyTorch recorre la cadena de
operaciones hacia atrás y calcula cuánto contribuyó cada uno de los ~864K
parámetros al error) → `optimizador.step()` (Adam mueve cada peso un
pasito en la dirección que reduce el error).

**Split train/val 90/10**, barajado, para poder distinguir aprendizaje real
de memorización.

**Hiperparámetros de esta corrida:**

```
vocab_size=254, d_model=128, num_heads=4, d_ff=512, num_layers=4
max_len=64, batch_size=32, épocas=5, lr=3e-4
parámetros totales: 864,510
```

**Resultado real** (`data/processed/train_log.txt`):

```
epoca 1/5  loss_train=2.2868  loss_val=1.1156
epoca 2/5  loss_train=0.7517  loss_val=0.5481
epoca 3/5  loss_train=0.4544  loss_val=0.3971
epoca 4/5  loss_train=0.3748  loss_val=0.3508
epoca 5/5  loss_train=0.3486  loss_val=0.3326
```

Referencia: `-ln(1/254) ≈ 5.54` es el loss esperado de un modelo sin
entrenar (probabilidad uniforme sobre el vocabulario). El loss final de
0.33 confirma aprendizaje real, no ruido. `loss_val` se mantuvo por debajo
de `loss_train` en todas las épocas — sin señal de overfitting (que se
vería como `loss_val` subiendo mientras `loss_train` sigue bajando),
esperable dado lo repetitivo/templado del corpus (generado por plantillas
en Fase 0).

Modelo entrenado guardado en `data/processed/modelo_fase4.pt` (pesos +
vocabulario + hiperparámetros, para poder recargarlo sin reentrenar).

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/train.py
```

## 4. Generación de texto (`src/generar.py`)

Generación autoregresiva: predecir el siguiente token → agregarlo al texto
→ volver a predecir usando el texto ya extendido, repetido hasta un límite
de tokens. En cada iteración se recorta la entrada a los últimos
`max_seq_len` tokens (la tabla de posiciones del modelo no cubre más).

Dos estrategias de elección sobre la distribución de probabilidad que
devuelve el modelo:

- **Greedy** (`torch.argmax`): siempre el token más probable — determinista,
  pero propenso a repeticiones/bucles.
- **Sampling con temperatura** (`torch.multinomial` sobre
  `softmax(logits / temperatura)`): sorteo ponderado por probabilidad;
  temperatura baja (0.5) se acerca a greedy, temperatura alta (1.0+) da más
  variedad a costa de coherencia.

**Resultado real** (`Marcador Partido`, `¿Quién`, `Roja` como inicios):

```
Marcador Partido  (greedy):        Marcador Partido 11 — España vs Cabo Verde Marfil vs Cabo To
¿Quién            (greedy):        ¿Quién gana Partido 11? ? ? ? (parte del combo Top 4
Roja              (sampling, 0.5): Roja en orden eón? (pón ita de penales) vs Catar 4:
```

El modelo reconoce y reproduce la estructura de las plantillas del corpus
(`"Marcador Partido N — Equipo1 vs Equipo2"`, `"¿Quién gana Partido N?"`) y
genera nombres de países reales del corpus, sin memorizar preguntas
exactas — combina patrones aprendidos en variaciones nuevas, con errores
típicos de un modelo pequeño entrenado pocas épocas (mezcla de plantillas
distintas, bucles con greedy). Confirma el entregable de la fase: "modelo
que genera texto con sabor al corpus".

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/generar.py
```

## 5. Verificación

No se agregaron tests automatizados nuevos en esta fase (a diferencia de
Fases 1-3) — la verificación se hizo de forma interactiva, pieza por pieza,
comparando cálculos manuales contra la salida real del código (self-attention
sobre `"tarjeta amarilla"`, causal masking sobre `"la tarjeta amarilla"`,
multi-head attention con 2 cabezas) antes de ensamblar el modelo completo.

La suite de tests existente (Fases 0-3) se confirmó intacta tras los
cambios de esta fase:

```bash
./.venv/Scripts/uv.exe run pytest -q
```

```
22 passed in 29.03s
```

## 6. Pendiente

- No hay tests automatizados (`tests/test_transformer_model.py`) para la
  arquitectura de esta fase — quedaría bien para Fase 6 (portafolio),
  formalizando las verificaciones manuales ya hechas (shapes, equivalencia
  causal/no-causal, reversibilidad de `bpe_vocab`).
- Generación es ineficiente a propósito: recalcula toda la secuencia en
  cada token nuevo en vez de usar KV-caching (técnica estándar de la
  industria para no reprocesar lo ya calculado) — fuera de alcance para
  este proyecto educativo, pero vale mencionarlo si se compara con
  inferencia real de producción.
- Fase 5 del roadmap: RAG completo — integrar el vector store de Fase 3
  (retrieval) con este LLM propio (generación), inyectando contexto
  recuperado en el prompt antes de generar (ver
  [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
