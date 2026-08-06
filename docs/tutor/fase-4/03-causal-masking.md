# Paso 3 — Causal masking

## El problema

Imagina al modelo, ya entrenado, generando texto palabra por palabra —
hasta ahora escribió `"¿Quién recibe la primera tarjeta"` y tiene que
adivinar la siguiente palabra. Para eso usa self-attention entre las
palabras que **ya escribió**. Pero self-attention (paso 2), tal como se
construyó, deja que **cualquier palabra mire a cualquier otra sin
restricción** — incluidas las que vienen después.

El problema aparece durante el **entrenamiento**: se le da al modelo la
oración completa de una vez (`"¿Quién recibe la primera tarjeta
amarilla?"`) para que aprenda a predecir cada palabra a partir de las
anteriores. Si al calcular la predicción para "tarjeta", el mecanismo de
atención le permite mirar también "amarilla" (que viene *después*, en la
misma oración de entrenamiento), el modelo estaría **haciendo trampa** —
vería la respuesta antes de tener que adivinarla. Aprendería a "copiar" en
vez de predecir de verdad, y en el momento real de generar texto (donde las
palabras futuras todavía no existen) fallaría completamente.

## La solución: tapar las posiciones futuras

**Causal masking:** al calcular el reparto de atención de la palabra en la
posición *i*, se le prohíbe mirar cualquier posición *mayor* a *i* — solo
puede ver posiciones iguales o anteriores a la suya.

Con 4 palabras ("Colombia venció a Brasil"), marcando con `X` lo permitido
y `-` lo tapado:

```
                Colombia  venció   a     Brasil
Colombia:          X        -      -       -      ← solo puede verse a sí misma
venció:            X        X      -       -      ← puede ver Colombia y a sí misma
a:                 X        X      X       -      ← puede ver Colombia, venció, y a sí misma
Brasil:            X        X      X       X      ← puede ver toda la oración anterior
```

Es una forma de **triángulo** — cada fila ve su propia posición y todas las
anteriores, nunca las posteriores.

**Dato curioso:** por esta forma, en el código de PyTorch/papers a veces se
llama *"lower triangular mask"* (máscara triangular inferior).

## Cómo se implementa: antes del softmax, no después

No se "borra" nada después — se pone un número extremadamente negativo
(`-infinito`, en la práctica) en las celdas prohibidas **antes** de aplicar
softmax. Softmax convierte números altos en porcentajes altos y números muy
bajos en porcentajes casi 0 — un puntaje `-infinito` recibe **0% de
atención**, sin necesidad de tocar la fórmula:

```
scores originales:        [0.30, -1.01, 2.55, 1.03]
+ máscara (tapar futuro):  [0.30, -1.01, -inf, -inf]   ← ejemplo para la 2da palabra, solo ve las 2 primeras
softmax(...):              [62%,   38%,   0%,    0%]   ← las tapadas quedan en 0% automáticamente
```

**Comparación con la industria:** este es el mecanismo que distingue un
modelo *generativo autoregresivo* (GPT) de uno tipo BERT (que sí deja ver
la oración completa en ambas direcciones, porque no genera texto
secuencialmente — se usa para clasificación o rellenar huecos). La "G" de
GPT es justamente por esto: *Generative* implica predecir hacia adelante,
palabra por palabra, sin espiar el futuro.

## Diseño de código: parámetro en la misma clase

En vez de una clase nueva, `SelfAttention` recibe un parámetro opcional
`causal: bool`. Cuando está activo, construye la máscara triangular y la
suma a los `scores` antes del softmax:

```python
def __init__(self, d_model, causal=False):
    ...
    self.causal = causal

def forward(self, x):
    ...
    scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)

    if self.causal:
        seq_len = x.shape[1]
        mascara = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mascara, float("-inf"))

    pesos_atencion = torch.softmax(scores, dim=-1)
    return pesos_atencion @ V
```

`torch.triu(..., diagonal=1)` genera el triángulo de "posiciones futuras"
(todo lo que está por encima de la diagonal principal, sin incluirla).
`masked_fill` reemplaza esas posiciones por `-inf` antes del softmax.

## Ejemplo real, verificado: `"la tarjeta amarilla"`

Comparando la misma matriz de pesos (mismos Q, K) con y sin la máscara:

**Sin causal masking** (cada palabra reparte libremente):

```
                    la</w>   tarjeta</w>   amarilla</w>
'la' atiende a:      50%        28%           22%     ← "la" ya mira "amarilla", que viene DESPUÉS
'tarjeta' atiende a: 60%        31%            9%     ← "tarjeta" también mira "amarilla" (futuro)
'amarilla' atiende a:17%        62%           21%
```

**Con causal masking** (la misma matriz de pesos original, con las
posiciones futuras tapadas antes del softmax):

```
                    la</w>   tarjeta</w>   amarilla</w>
'la' atiende a:      100%        0%            0%      ← ahora SOLO puede verse a sí misma (primera palabra)
'tarjeta' atiende a: 66%        34%            0%      ← puede ver 'la' y a sí misma, NO 'amarilla'
'amarilla' atiende a:17%        62%           21%      ← es la última, puede ver todo (nada que tapar)
```

**Lo que cambió:**
- `'la'` (primera palabra): antes repartía 50/28/22%; ahora el 100% se
  queda con ella misma — no hay ninguna palabra anterior que pueda mirar.
- `'tarjeta'` (segunda palabra): antes tenía 9% hacia `'amarilla'` (el
  futuro); con la máscara ese 9% desaparece y el resto (66/34%) se
  redistribuye entre `'la'` y sí misma — softmax siempre reparte 100% entre
  lo que sí puede ver.
- `'amarilla'` (última palabra): no cambia — al ser la última, no tiene
  ningún "futuro" que tapar.

Esa forma triangular garantiza que, al entrenar la predicción de
`"amarilla"` a partir de `"la tarjeta"`, el modelo nunca pudo haber mirado
`"amarilla"` para hacerla — la única forma de acertar es haber aprendido de
verdad el patrón del lenguaje, no copiar la respuesta.

## Siguiente paso

[04-multi-head-attention.md](04-multi-head-attention.md) — por qué una sola
"cabeza" de atención no basta y cómo se combinan varias en paralelo.
