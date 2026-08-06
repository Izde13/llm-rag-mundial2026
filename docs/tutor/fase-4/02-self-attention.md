# Paso 2 — Self-attention (una cabeza)

## El problema que resuelve

Tras el paso 1, cada palabra tiene su propio vector (token + posición) —
pero vive **aislado**, sin saber nada de las demás palabras de la oración.
El vector de "venció" no sabe que "Colombia" está cerca, ni que "Brasil"
viene después.

Ejemplo con el corpus propio: `"¿Quién recibe la primera tarjeta
amarilla?"`. Para que el modelo entienda bien "amarilla" aquí, necesita
relacionarla con "tarjeta" (no es un color cualquiera) y con "primera"
(importa el orden temporal). Self-attention es el mecanismo que le permite
a cada palabra **"mirar" a las demás palabras y decidir cuánto tomar de
cada una**.

## La analogía: una reunión donde cada palabra hace una pregunta

Cada palabra de la oración es una persona en una mesa de reunión:

1. **Hace una pregunta** ("¿quién tiene información relevante para mí?") —
   su **Query** (consulta).
2. **Cada otra palabra tiene un "cartel"** que resume qué información
   ofrece — su **Key** (llave).
3. Se compara la pregunta de la palabra 1 contra el cartel de cada una de
   las demás — cuanto más "calzan", más atención le presta.
4. Cada palabra tiene también **el contenido real** que aportaría si le
   prestan atención — su **Value** (valor).
5. La palabra 1 se lleva una **mezcla** de los "values" de todas las demás,
   ponderada según cuánta atención le prestó a cada una.

Esto pasa para las 4 palabras a la vez, simultáneamente — cada una hace su
propia pregunta, mira a todas las demás (incluida ella misma), y se lleva
su propia mezcla.

Q, K, V no son "otra cosa mágica" — son el mismo vector de la palabra (el
que salió del paso 1), pasado por **tres transformaciones lineales
distintas** (tres matrices de pesos aprendibles, una por cada uno). Al
inicio esas matrices son aleatorias — otra vez, la maquinaria antes que el
significado.

## Cómo se ajusta esto durante el entrenamiento (la pregunta de fondo)

**Nadie le dice al modelo "tarjeta va con amarilla".** Eso se descubre
solo, indirectamente, por prueba y error masivo:

1. Se le da al modelo, del corpus real: entrada `"¿Quién recibe la primera
   tarjeta"`, objetivo `"amarilla"` (la palabra real que sigue).
2. Con las matrices Q/K/V aún aleatorias, el modelo predice mal.
3. Se mide el error (la "loss").
4. **Backpropagation + descenso de gradiente**: cada uno de los millones de
   números dentro del modelo tuvo algún grado de responsabilidad en el
   error. Backpropagation calcula, para cada uno, en qué dirección y cuánto
   moverlo para reducir el error si se repitiera este mismo ejemplo. Se
   mueve cada número un pasito en esa dirección.
5. Se repite con miles de ejemplos distintos del corpus, varias vueltas
   (épocas).

**Por qué termina emergiendo justo "amarilla ↔ tarjeta":** la palabra
"tarjeta" aparece, en el corpus, en muchísimas oraciones donde la palabra
siguiente es "amarilla" o "roja" — nunca "gol" o "corner". Cada vez que el
modelo falla en predecir el color correcto, el ajuste empuja — un poquito
cada vez — a que la matriz Query de "amarilla" y la matriz Key de "tarjeta"
produzcan vectores que, al compararlos (producto punto), den un número
alto. La atención fuerte entre ambas palabras **no se programa** — se
vuelve la forma más eficiente de reducir el error, tras verlo repetirse
cientos de veces. El signo `¿`, en cambio, aparece indistintamente antes de
palabras completamente distintas — nunca ayuda a predecir nada específico,
así que nunca desarrolla atención fuerte hacia él.

**La idea central: el "saber" no vive en un solo lugar (ni en Q, ni en K,
ni en el embedding) — vive en la combinación de todos esos pesos, ajustados
juntos, muchas veces, porque juntos determinan si el modelo acierta o falla
en predecir la siguiente palabra.**

## La fórmula completa

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```

1. **`Q · Kᵀ`** — comparar cada pregunta contra cada cartel. Produce una
   matriz `seq_len × seq_len` de "puntajes de compatibilidad" — la celda
   `[i,j]` dice cuánto calza la pregunta de la palabra *i* con el cartel de
   la palabra *j*. Es el mismo producto punto usado para similitud en
   Fase 2/3, con propósito distinto.
2. **`/ √d_k`** — ajuste de escala. Con `d_model` grande, los productos
   punto tienden a salir muy grandes, lo que vuelve inestable el softmax
   siguiente (una sola palabra "gana" toda la atención de forma extrema).
   Dividir mantiene los números en un rango razonable — estabilidad
   numérica, no significado.
3. **`softmax(...)`** — convierte puntajes en porcentajes de atención que
   suman 100% por fila. Ej.: puntajes `5, 2, 8, 1` → reparto `30%, 10%,
   55%, 5%`.
4. **`· V`** — mezcla final: promedio ponderado de los "contenidos" (Value)
   de todas las palabras, según el reparto de atención.

```
entrada:      [batch, seq_len, d_model]
Q, K, V:      [batch, seq_len, d_model]      (misma forma, tres transformaciones distintas)
Q · Kᵀ:       [batch, seq_len, seq_len]      (matriz de puntajes: cada palabra vs. cada palabra)
softmax(...): [batch, seq_len, seq_len]      (mismos puntajes, como porcentajes)
resultado:    [batch, seq_len, d_model]      (de vuelta a la forma original)
```

## Cómo se usa el vector de una palabra para producir su Q — a mano

`Q_tarjeta = W_query · vector_de_tarjeta`: **cada fila de la matriz
`W_query` se multiplica número por número contra el vector de entrada, y se
suman los resultados** (multiplicar y sumar — la misma operación de
siempre, con un paso extra de multiplicar antes de sumar).

Con `vector_de_tarjeta = [-1.00, -1.70, 1.94, -1.41]` y la primera fila de
`W_query = [0.24, -0.28, 0.11, 0.06]`:

```
(0.24 × -1.00) + (-0.28 × -1.70) + (0.11 × 1.94) + (0.06 × -1.41)
=  -0.24        +   0.48         +   0.21         +  -0.08
=   0.37
```

Ese `0.37` es el primer número de `Q_tarjeta`. Se repite con cada fila de
`W_query` para los demás números del vector. **La misma matriz `W_query` se
usa para todas las palabras** (no es una transformación distinta por
palabra) — lo que cambia es el vector de entrada, no la regla de
transformación. Es como una fotocopiadora con lentes desajustados: la
distorsión (aunque mal calibrada al inicio) se aplica siempre igual a
cualquier cosa que se le meta.

Lo mismo se repite con otras dos matrices (`W_key`, `W_value`) para obtener
K y V.

## Qué se hace con Q, K, V una vez calculados

```
Q y K  →  se comparan entre sí (producto punto)  →  producen el reparto de atención (%)
V      →  se mezcla usando ese reparto            →  produce el vector final de cada palabra
```

Q y K son solo "calculadoras del reparto" — nunca aparecen en el resultado
final directamente. V es el único que efectivamente compone el vector de
salida, ponderado según lo que Q y K decidieron.

## Ejemplo real, de punta a punta: `"tarjeta amarilla"`

Con `d_model=4`, entrada del paso 1:

```
'tarjeta'  : [-1.00, -1.70,  1.94, -1.41]
'amarilla' : [-2.02,  1.65,  1.90,  0.65]
```

**Q, K, V** (con pesos aleatorios sin entrenar):

```
'tarjeta'  → Q=[0.07,-1.72,0.32,-0.73]  K=[1.30,-0.52,1.94,1.40]  V=[0.86,-0.32,-0.24,-0.55]
'amarilla' → Q=[1.17,-0.90,0.18,1.98]   K=[-0.53,0.41,-0.42,1.58] V=[-0.68,0.56,-0.59,0.46]
```

**Paso 1 — `Q · Kᵀ`:**

```
                cartel de 'tarjeta'   cartel de 'amarilla'
pregunta 'tarjeta':    0.59              -2.02
pregunta 'amarilla':   5.10               2.06
```

Verificación a mano de la celda `[tarjeta, tarjeta] = 0.59`:

```
Q_tarjeta = [0.07, -1.72, 0.32, -0.73]
K_tarjeta = [1.30, -0.52, 1.94, 1.40]

(0.07×1.30) + (-1.72×-0.52) + (0.32×1.94) + (-0.73×1.40)
=  0.09      +   0.89        +   0.62       +  -1.02
=  0.59   ✓
```

**Paso 2 — dividir entre `√4 = 2`:**

```
0.30   -1.01
2.55    1.03
```

**Paso 3 — softmax, fila por fila:**

```
'tarjeta' reparte:   79% a 'tarjeta', 21% a 'amarilla'
'amarilla' reparte:  82% a 'tarjeta', 18% a 'amarilla'
```

**Paso 4 — mezclar V según ese reparto:**

```
V_tarjeta  = [0.86, -0.32, -0.24, -0.55]
V_amarilla = [-0.68, 0.56, -0.59, 0.46]

nuevo_tarjeta = 0.79 × V_tarjeta + 0.21 × V_amarilla

primer número: 0.79×0.86 + 0.21×(-0.68) = 0.68 - 0.14 = 0.53
```

**Resultado final:**

```
nuevo vector de 'tarjeta' : [0.53, -0.13, -0.32, -0.33]
nuevo vector de 'amarilla': [0.58, -0.16, -0.31, -0.37]
```

Con pesos sin entrenar, este reparto es esencialmente ruido — no significa
que "amarilla realmente le presta 82% de atención a tarjeta" en ningún
sentido semántico todavía. Es la mecánica funcionando correctamente sobre
números al azar; el significado real llega con el entrenamiento (paso 7).

## Código real (`src/transformer_model.py`)

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, causal=False):
        super().__init__()
        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model
        self.causal = causal

    def forward(self, x):
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        pesos_atencion = torch.softmax(scores, dim=-1)
        return pesos_atencion @ V
```

**Verificado:** correr este código sobre `"tarjeta amarilla"` con
`torch.manual_seed(0)` produce exactamente `[0.53, -0.13, -0.32, -0.33]` y
`[0.58, -0.16, -0.31, -0.37]` — coincide número por número con el cálculo
manual de arriba.

Shape de entrada `(1, 2, 4)` → shape de salida `(1, 2, 4)`: self-attention
**no cambia la forma**, solo mezcla contenido entre posiciones — necesario
para poder encadenar con más piezas sin ajustar tamaños.

## Siguiente paso

[03-causal-masking.md](03-causal-masking.md) — por qué, para entrenar un
modelo generativo, cada palabra no puede mirar libremente hacia el futuro.
