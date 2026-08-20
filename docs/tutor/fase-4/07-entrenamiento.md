# Paso 7 — Cabeza de salida y loop de entrenamiento

Esta es la fase donde, por fin, todos los pesos aleatorios construidos en
los pasos 1-6 empiezan a tener significado real.

## Parte A — La cabeza de salida: de "vector" a "qué palabra sigue"

`TransformerLM` (paso 6) da un vector de 128 números por posición
(`[batch, seq_len, 128]`). Lo que se quiere es: para cada posición, ¿cuál
de los 254 símbolos del vocabulario debería venir después?

Se agrega **otra capa lineal más**, que convierte el vector de 128 números
en un vector de 254 números — uno por cada símbolo posible:

```python
logits = nn.Linear(d_model, vocab_size)(vector_final)   # 128 -> 254
```

A esos 254 números crudos se les llama **logits** — "puntajes sin
normalizar" de qué tan probable es cada símbolo como siguiente token.
Todavía no son probabilidades (pueden ser negativos, no suman 1) — se les
aplica **softmax**, igual que en self-attention, convirtiéndolos en 254
porcentajes que suman 100%:

```
"¿Quién recibe la primera tarjeta" → vector final (128) → logits (254) → softmax → probabilidades (254)
                                                                              ↓
                                                probabilidad_de("amarilla") = 34%
                                                probabilidad_de("roja")     = 22%
                                                probabilidad_de("azul")     = 0.01%
                                                ... (251 más)
```

## Parte B — Medir el error: cross-entropy loss

Se sabe qué palabra *debería* haber salido (se saca del propio corpus — la
palabra real que sigue en el texto). **Cross-entropy loss** compara "qué
probabilidad le asignó el modelo a la respuesta correcta" y lo convierte en
un número de error:

- Alta probabilidad a la palabra correcta (ej. 90% a "amarilla") → loss
  **bajo**.
- Baja probabilidad a la palabra correcta (ej. 1% a "amarilla", 80% a
  "roja") → loss **alto**.

No se implementó a mano — PyTorch la trae lista (`nn.CrossEntropyLoss`).

## Parte C — El loop de entrenamiento: backpropagation en código real

El "predecir-fallar-ajustar" del paso 2 (por qué "tarjeta" y "amarilla"
terminan relacionándose) se convierte en 4 líneas de código, repetidas
miles de veces:

```python
for lote in datos_de_entrenamiento:
    logits = modelo(lote_entrada)                  # 1. predecir
    loss = funcion_perdida(logits, lote_objetivo)   # 2. medir el error
    loss.backward()                                  # 3. calcular en qué dirección ajustar cada peso
    optimizador.step()                               # 4. mover cada peso un pasito en esa dirección
```

- **`loss.backward()`**: como cada tensor "recuerda" las operaciones que se
  le aplicaron, PyTorch recorre toda la cadena hacia atrás (embedding →
  attention → feed-forward → ... → logits → loss) y calcula automáticamente
  cuánto contribuyó cada uno de los ~864,000 parámetros al error final.
  Esto es backpropagation.
- **`optimizador.step()`**: usa esos cálculos para efectivamente mover cada
  peso. El algoritmo usado es **Adam**, una versión mejorada del "descenso
  de gradiente" que ajusta automáticamente qué tan grande debe ser cada
  paso.
- **`optimizador.zero_grad()`** (antes de `backward()`): "limpia" los
  cálculos de ajuste del lote anterior — si no, PyTorch los iría acumulando
  en vez de reemplazarlos.

## Parte D — Entrada y objetivo desplazados

Detalle específico de un modelo generativo tipo GPT: **el "objetivo" es la
misma oración, desplazada una posición**.

```
oración real:   "¿"  "Quién"  "recibe"  "la"  "primera"  "tarjeta"  "amarilla"
entrada  (x):   "¿"  "Quién"  "recibe"  "la"  "primera"  "tarjeta"
objetivo (y):        "Quién"  "recibe"  "la"  "primera"  "tarjeta"  "amarilla"
```

En la posición 0 (`"¿"`), el modelo debe predecir `"Quién"`. En la posición
1 (`"¿ Quién"`), debe predecir `"recibe"`. Cada posición predice la palabra
que sigue inmediatamente después, usando causal masking (paso 3) para no
hacer trampa mirando el futuro — la razón exacta por la que se construyó
causal masking antes: sin él, esta forma de entrenar sería imposible de
forma honesta.

## Código real (`src/transformer_model.py` y `src/train.py`)

**Cabeza de salida agregada a `TransformerLM`:**

```python
class TransformerLM(nn.Module):
    def __init__(self, ...):
        ...
        self.cabeza_salida = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.cabeza_salida(x)  # [batch, seq_len, vocab_size] -- logits
```

**Preparación de datos (`preparar_lote`):**

```python
def preparar_lote(textos, merges, simbolo_a_id, max_len):
    id_pad = simbolo_a_id[TOKEN_PAD]
    entradas, objetivos = [], []
    for texto in textos:
        ids = codificar_ids(texto, merges, simbolo_a_id)
        ids = ids[: max_len + 1]
        ids = ids + [id_pad] * (max_len + 1 - len(ids))
        entradas.append(ids[:-1])   # x
        objetivos.append(ids[1:])   # y, desplazado una posición
    return torch.tensor(entradas), torch.tensor(objetivos)
```

Se trunca oraciones más largas que `max_len` (además de rellenar las
cortas) — necesario porque el roadmap fija un contexto corto y fijo, y
algunas preguntas del corpus superan ese largo.

**Un paso de entrenamiento:**

```python
def un_paso_de_entrenamiento(modelo, optimizador, funcion_perdida, entrada, objetivo):
    logits = modelo(entrada)
    loss = funcion_perdida(logits.reshape(-1, vocab_size), objetivo.reshape(-1))
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()
    return loss.item()
```

**Dos detalles de esta implementación:**
- **`ignore_index=simbolo_a_id[TOKEN_PAD]`** en `CrossEntropyLoss`: le dice
  a la función de pérdida que **ignore** las posiciones `<PAD>` al calcular
  el error, porque son relleno mecánico sin contenido real.
- **`evaluar(...)`** (con `@torch.no_grad()`): mismo cálculo de loss sobre
  el set de validación, pero sin `backward()` ni `optimizador.step()` — solo
  mide, no ajusta pesos. Sirve para detectar si el modelo memoriza en vez
  de generalizar.

## Prueba de un solo paso (verificación antes de entrenar de verdad)

Antes de lanzar el entrenamiento completo, se corrió **un único** paso
sobre 8 preguntas reales, solo para confirmar que el mecanismo está
correctamente conectado:

```
Shape entrada:  (8, 32)
Shape objetivo: (8, 32)

Loss tras un paso: 5.6900
(referencia: -ln(1/254) = 5.5373 -- loss esperado con pesos al azar, sin haber aprendido nada)
```

`5.69` está muy cerca del valor teórico esperado con pesos completamente al
azar (`5.54`) — exactamente lo esperado tras un solo paso: el modelo sigue
prácticamente "no sabiendo nada", pero el mecanismo (predecir → medir error
→ ajustar) está correctamente conectado de principio a fin.

## Hiperparámetros decididos para el entrenamiento real

Antes de lanzar el entrenamiento completo, se repasó qué controla cada
hiperparámetro:

- **Arquitectura:** `d_model=128, num_heads=4, d_ff=512, num_layers=4` (ya
  usados en pruebas anteriores), `max_seq_len=64` (el roadmap sugiere
  64-128; con datos de Fase 1 —promedio 11 tokens BPE por pregunta, máximo
  85— cubre casi todas las preguntas sin truncar).
- **`batch_size=32`:** tensión entre lotes grandes (gradientes más suaves,
  mejor uso de hardware) y lotes chicos (más rápidos por paso, más
  ruidosos). Razonable para CPU y un corpus de 6582 preguntas.
- **`num_epocas=5`:** tensión entre muy pocas épocas (el modelo no llega a
  aprender nada útil) y demasiadas (riesgo de *overfitting* — memorizar el
  corpus en vez de generalizar, más probable con un corpus chico y
  repetitivo como este).
- **`lr=3e-4`:** valor estándar de la industria para Adam con modelos de
  este tamaño.
- **Train/validation split 90/10, barajado:** para poder distinguir
  aprendizaje real de memorización — el set de validación nunca se usa
  para entrenar, solo para medir el loss al final de cada época.

## Resultado real del entrenamiento

```
=== Entrenamiento TransformerLM (Fase 4) ===
Corpus: 6582 preguntas -> 5923 train / 659 val
vocab_size=254, d_model=128, num_layers=4, parametros=864,510
max_len=64, batch_size=32, epocas=5, lr=0.0003

epoca 1/5  loss_train=2.2868  loss_val=1.1156
epoca 2/5  loss_train=0.7517  loss_val=0.5481
epoca 3/5  loss_train=0.4544  loss_val=0.3971
epoca 4/5  loss_train=0.3748  loss_val=0.3508
epoca 5/5  loss_train=0.3486  loss_val=0.3326
```

**Cómo leer esto:** el loss inicial de referencia (sin entrenar) era
`5.54`. Tras 5 épocas, bajó a `0.33` — una caída enorme, señal clara de
aprendizaje real, no ruido.

**Detalle a mirar con cuidado: `loss_val` menor que `loss_train` en cada
época.** Esto es inusual (normalmente se esperaría lo contrario). Dos
explicaciones probables, no excluyentes:

1. El corpus es muy repetitivo y estructurado (preguntas generadas con
   plantillas fijas en Fase 0) — el set de validación probablemente
   comparte patrones casi idénticos con el de entrenamiento, así que
   "generalizar" aquí es más fácil de lo normal.
2. `loss_train` se promedia usando pesos que fueron cambiando a lo largo de
   la época (los primeros lotes se miden con pesos "más viejos"), mientras
   `loss_val` siempre se mide **al final**, con los pesos ya más ajustados
   de esa época — eso sesga `loss_train` ligeramente hacia arriba de forma
   artificial.

**Lo importante:** no se ve la señal de alarma de overfitting (`loss_val`
subiendo mientras `loss_train` sigue bajando) — ambos bajan de forma
consistente y terminan cerca uno del otro. Señal saludable para esta
primera corrida.

Modelo guardado en `data/processed/modelo_fase4.pt` (pesos + vocabulario +
hiperparámetros, para poder recargarlo sin reentrenar).

## Siguiente paso

[08-generacion.md](08-generacion.md) — usar el modelo ya entrenado para
generar texto nuevo, token a token.
