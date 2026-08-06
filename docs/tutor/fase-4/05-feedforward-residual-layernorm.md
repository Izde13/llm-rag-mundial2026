# Paso 5 — Feed-forward, conexión residual y layer norm

Tres piezas juntas porque, en un transformer, siempre aparecen combinadas
de la misma forma.

## 1. Feed-forward (FFN) — "pensar" cada palabra por su cuenta

### La analogía

Multi-head attention (pasos 2-4) es puro **mezclado** — cada palabra
termina siendo un promedio ponderado de otras palabras. Es como mezclar
pintura: si mezclas 70% azul + 30% amarillo, obtienes verde, pero **nunca
un color que no sea alguna combinación de los colores que ya tenías**.
Attention está matemáticamente limitado de la misma forma: por mucho que
mezcle, sigue siendo una combinación de lo que ya existía.

Piensa en una persona que sale de una reunión de equipo (attention juntó
opiniones de sus compañeros) y ahora, sola en su escritorio, **necesita
tiempo para procesar lo que escuchó y sacar sus propias conclusiones**, sin
hablar con nadie más en ese momento. Eso hace feed-forward: cada palabra,
por separado, "piensa" sobre la información que ya tiene, sin mirar a las
demás (a diferencia de attention).

### El mecanismo: estirar, filtrar, comprimir

```
número de entrada (128)  →  se estira a  (512)  →  se vuelve a comprimir a  (128)
```

**Por qué estirar primero:** más espacio de trabajo temporal para combinar
y transformar la información de formas complejas, antes de comprimirla de
vuelta — como resolver un problema en una hoja grande en vez de una
diminuta. El factor 4x (128→512→128) lo usó el paper original de 2017 y se
sigue usando en GPT y la mayoría de LLMs modernos.

**ReLU en el medio:** `ReLU(x) = max(0, x)` — "si es negativo, cámbialo a
0; si es positivo, déjalo igual". Sin ella, `Linear2(Linear1(x))` sería
matemáticamente equivalente a **una sola** transformación lineal (dos
transformaciones lineales seguidas colapsan en una) — no se ganaría nada
por tener dos capas. ReLU rompe esa linealidad.

### El valor real de FFN (no solo la mecánica)

Attention decide **de quién** tomar información. Feed-forward decide **qué
hacer** con esa información una vez reunida — puede aprender relaciones que
no son una simple mezcla: reglas del tipo "si esta combinación específica
de números aparece, entonces produce esta otra cosa distinta", incluyendo
relaciones no proporcionales ni lineales.

Ejemplo del dominio: "si el vector tiene rasgos de *tarjeta* Y rasgos de
*segunda amonestación en el partido*, entonces significa *expulsión*". Eso
no es "un poco de tarjeta más un poco de segunda vez" — es una combinación
específica que dispara un concepto distinto. Attention por sí solo (puro
mezclado/promediado) no puede aprender ese tipo de "si pasa esto Y esto,
entonces esto otro" — necesita la no-linealidad de ReLU dentro de
feed-forward.

**En una frase: feed-forward es lo que le da al modelo la capacidad de
aprender patrones que no son simples promedios — sin él, todo el modelo,
sin importar cuántas capas de attention tenga, seguiría siendo
matemáticamente equivalente a un puñado de promedios ponderados
encadenados, y eso no alcanza para capturar el lenguaje real.**

### La matemática exacta detrás (misma operación de siempre)

```
1. Linear1:  salida = (entrada × pesos1) + sesgo1     ← multiplicar y sumar (4 → 8 números, ej.)
2. ReLU:     salida = max(0, entrada)                  ← comparar con 0 (sin cambiar tamaño)
3. Linear2:  salida = (entrada × pesos2) + sesgo2     ← multiplicar y sumar otra vez (8 → 4)
```

No hay matemática nueva — son las mismas dos operaciones de Q/K/V
(multiplicar-y-sumar) más una comparación simple (ReLU) en el medio. Lo
único distinto es el **tamaño** de las matrices de pesos.

### Ejemplo real, verificado: vector de "amarilla"

Con `d_model=4`, `d_ff=8`, sobre `vector_amarilla = [0.58, -0.16, -0.31, -0.37]`
(el mismo vector que salió de self-attention en el paso 2):

```
entrada:                          [0.58, -0.16, -0.31, -0.37]

paso 1 -- estirado a 8 (Linear1): [-0.11, -0.51, -0.31, -0.64, -0.04, 0.36, 0.2, -0.02]

paso 2 -- ReLU (negativos a 0):   [0.0, 0.0, 0.0, 0.0, 0.0, 0.36, 0.2, 0.0]
                                    (6 de 8 números eran negativos y se apagaron)

paso 3 -- comprimido a 4 (Linear2): [0.12, 0.36, 0.32, 0.38]
```

Con datos reales del corpus (`d_model=6`, `d_ff=24`, sobre `"la tarjeta
amarilla"`): cada palabra procesada de forma completamente independiente
(el vector de `'la'` cambió sin que `'tarjeta'` ni `'amarilla'`
influyeran); shape se preservó `(1, 3, 6)` → `(1, 3, 6)`; 318 parámetros
entrenables con ese `d_model` chico (`(6×24+24) + (24×6+6)`). Con
`d_model=128` real y `d_ff=512`: `(128×512+512) + (512×128+128) ≈ 131,712`
parámetros — feed-forward suele ser la pieza con **más** parámetros del
bloque completo, más que attention.

### Código real (`src/transformer_model.py`)

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
```

## 2. Conexión residual — "no perder lo que ya tenías"

### El problema

Imagina un "teléfono descompuesto": si cada persona **reemplaza por
completo** lo que escuchó con su propia versión, después de 6-8 personas el
mensaje original puede quedar irreconocible. El modelo tendrá varias capas
apiladas (4-6) — si cada capa **reemplaza por completo** el vector de la
palabra con su propia versión nueva, corre el mismo riesgo: información
útil (ej. "qué palabra es literalmente esta") se puede perder o distorsionar
capa tras capa, sin que el modelo pueda recuperarla si la necesita después.

### La solución: sumar, no reemplazar

```
sin conexión residual:  vector_nuevo = Attention(vector_viejo)
con conexión residual:  vector_nuevo = vector_viejo + Attention(vector_viejo)
```

Es la misma "suma elemento a elemento" ya vista (token+posición en el paso
1). Analogía: editar un documento con control de cambios activado — en vez
de borrar una frase y escribir una nueva, se añade una anotación al margen
sin borrar el texto original. El texto original sigue disponible, y la
anotación se suma como información adicional.

### El beneficio práctico

Si una capa "no tiene nada útil que aportar" para cierta palabra, puede
aprender a producir un resultado cercano a 0 — y como se suma (no
reemplaza), el vector pasa prácticamente intacto a la siguiente capa. Sin
la conexión residual, esa misma capa "inútil" estaría obligada a reemplazar
el vector con algo, arriesgando distorsionarlo aunque no tuviera nada que
aportar — eso hace mucho más difícil y lento entrenar redes con muchas
capas apiladas.

**Dato curioso:** las conexiones residuales no nacieron con los
transformers — vienen de ResNet (Microsoft Research, 2015), arquitectura de
visión por computadora que ganó ImageNet ese año. Fue tan efectiva para
entrenar redes muy profundas que el paper de transformers de 2017 la
adoptó directamente.

## 3. Layer normalization — mantener los números en un rango estable

### El problema

Con conexiones residuales sumando capa tras capa, los números se pueden ir
haciendo cada vez más grandes (o más chicos), acumulándose capa sobre
capa — como una bola de nieve que crece al rodar. El entrenamiento (los
ajustes de backpropagation) funciona mejor cuando los números están en un
rango razonable y predecible; con números muy grandes o dispersos, los
ajustes se vuelven erráticos, como afinar un instrumento con clavijas
demasiado sueltas o tensas: pequeños giros producen cambios exagerados.

### La analogía: reescalar una receta

Una receta para 40 personas ("3kg de harina", "40 huevos") se reescala
(dividiendo todo por 40) para cocinarla para uno solo, sin cambiar la
esencia — sigue siendo la misma combinación de ingredientes, en una escala
manejable.

Layer normalization hace esto con el vector de cada palabra, tras cada
suma residual: reescala los números para que, en promedio, ronden 0 y
tengan una dispersión estándar — sin cambiar la relación entre ellos.

```
vector antes de normalizar:  [50, -30, 80, 10]     ← números grandes, dispersos
                                    ↓ layer norm
vector después:              [0.3, -1.2, 1.5, -0.6] ← mismo "patrón" relativo, escala manejable
```

El número más alto (80) sigue siendo el más alto después (1.5); el más
bajo (-30) sigue siendo el más bajo (-1.2) — el orden relativo no cambia,
solo la escala.

**En una frase: layer norm no le agrega ningún significado nuevo al
vector — es puro mantenimiento técnico, como reescalar la receta.**

### Código: se reutiliza la de PyTorch

No se implementó a mano — `nn.LayerNorm(d_model)` es pura aritmética de
reescalado, sin nada conceptualmente nuevo que aprender (aunque tiene un
par de parámetros ajustables que afinan la escala final, detalle menor). Se
usa directamente, igual que ya se usaban `nn.Linear` y `nn.Embedding` sin
reimplementarlas.

## Cómo se ensamblan las tres piezas

```
x
│
├──────────────┐
▼              │
MultiHeadAttn   │ (conexión residual)
│              │
+ ◄────────────┘
│
LayerNorm
│
├──────────────┐
▼              │
FFN             │ (conexión residual)
│              │
+ ◄────────────┘
│
LayerNorm
│
▼
salida
```

Este patrón se ensambla en código en el paso 6 (`TransformerBlock`).

## Siguiente paso

[06-transformer-block.md](06-transformer-block.md) — ensamblar todas las
piezas en un bloque completo y apilar varios.
