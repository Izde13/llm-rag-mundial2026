# Paso 1 — Token embedding y positional encoding

## Fase 2 vs. Fase 4: dos tipos de embedding completamente distintos

En **Fase 2** se usó `sentence-transformers` con `all-MiniLM-L6-v2`: una red
neuronal que **alguien más ya entrenó** (con billones de palabras de
internet) y que solo se *usó* — se le dio texto, devolvió vectores. Los
pesos de esa red están congelados; nunca se tocaron ni se ajustaron. Es como
usar una calculadora ya hecha.

En **Fase 4**, el "embedding" es una simple tabla de números
**inicializados al azar** (`nn.Embedding` en PyTorch — literalmente una
matriz `[vocab_size, dim]`). Al principio esos vectores no significan
nada — son ruido. La diferencia está en lo que pasa después: durante el
entrenamiento, cuando el modelo se equivoca prediciendo el siguiente token,
backpropagation ajusta **esa misma tabla** (además de todos los demás pesos
del transformer) para que, poco a poco, vectores de tokens que aparecen en
contextos parecidos terminen pareciéndose entre sí — igual que en Fase 2,
pero construido y entrenado desde cero con el propio corpus.

Dos matices:

1. El tokenizer que alimenta esto **no** es el de HuggingFace/MiniLM — es el
   propio tokenizer BPE de Fase 1, entrenado sobre el corpus del mundial.
2. Nada de esto tiene significado hasta que se entrena (paso 7). Los pasos
   1-6 son pura arquitectura — construir la "maquinaria vacía". El
   significado emerge recién cuando esa maquinaria ve datos y se ajusta con
   gradiente descendente.

## `nn.Embedding` — la idea, sin código

`nn.Embedding` es una tabla. Con 3 tokens posibles (`gol=0, corner=1,
roja=2`) y 2 dimensiones por vector:

```
        dim0    dim1
id 0:  [ 0.5,  -1.2 ]     ← el vector de "gol"
id 1:  [ 0.1,   0.9 ]     ← el vector de "corner"
id 2:  [-0.7,   0.3 ]     ← el vector de "roja"
```

Si se pide el vector del token con `id=0`, simplemente se devuelve **la
fila 0** — un lookup, como buscar una palabra en un diccionario y sacar su
definición. `id` entra, vector de esa fila sale. No hay matemática compleja.

**La única parte "mágica": esos números empiezan al azar y luego cambian.**
Al crear `nn.Embedding(3, 2)`, PyTorch rellena la tabla con números
aleatorios sin significado — un "cajón vacío con compartimentos". Durante
el entrenamiento, cada vez que el modelo se equivoca, PyTorch ajusta un
poquito esos números — así, con miles de ejemplos, la fila de "gol" termina
pareciéndose más a la fila de "anotación" que a la de "roja", porque el
entrenamiento los fue empujando en esa dirección.

## Por qué hace falta la posición

### El problema

```
"Colombia venció a Brasil"
"Brasil venció a Colombia"
```

Significan cosas **opuestas** (distinto ganador), pero usan las mismas 4
palabras en orden distinto.

Self-attention (paso 2) procesa un conjunto de vectores mirando "quién se
relaciona con quién", pero por diseño matemático **no sabe cuál vector vino
primero o segundo**. Si se le pasan los vectores de `[Colombia, venció, a,
Brasil]` o de `[Brasil, venció, a, Colombia]`, sin ayuda extra los trataría
como el mismo conjunto reordenado — perdiendo justo la información que
distingue quién le ganó a quién.

### La solución: sumar una etiqueta de "dónde estoy parado"

```
"Colombia" en posición 0  →  vector("Colombia") + vector("posición 0")
"venció"   en posición 1  →  vector("venció")   + vector("posición 1")
"a"        en posición 2  →  vector("a")        + vector("posición 2")
"Brasil"   en posición 3  →  vector("Brasil")   + vector("posición 3")
```

En la oración invertida, "Colombia" recibiría el vector de "posición 3" en
vez de "posición 0" — un resultado final distinto, que es justo lo que
permite al modelo distinguir el orden.

La tabla de posiciones es **otra tabla `nn.Embedding`**, idéntica en
mecánica a la de tokens: indexada por "en qué posición estoy" en vez de
"qué palabra es", con la misma inicialización aleatoria y el mismo proceso
de aprendizaje durante el entrenamiento.

### Dos formas de codificar posición, y la decisión tomada

| Enfoque | Cómo funciona | Quién lo usa |
|---|---|---|
| **Sinusoidal (fijo)** | Fórmula fija con senos/cosenos de distinta frecuencia por dimensión — no se entrena, se calcula una vez. | Transformer original (2017) |
| **Aprendido (`nn.Embedding` de posiciones)** | Tabla `[max_seq_len, d_model]`, igual que el embedding de tokens. Se entrena junto con todo lo demás. | GPT-2, GPT-3, la mayoría de LLMs modernos |

**Dato curioso:** el paper de 2017 eligió sinusoidal argumentando que
permitiría extrapolar a secuencias más largas que las vistas en
entrenamiento. En la práctica esa ventaja resultó débil, y GPT-2 en
adelante usa una tabla de posiciones aprendida — más simple y funciona
igual de bien. Se eligió posicional aprendido para este proyecto: reutiliza
el mismo concepto de `nn.Embedding` ya visto, en vez de introducir fórmulas
trigonométricas nuevas.

## "Suma elemento a elemento", la operación de fondo

```
Lista A: [1.5, -0.3]
Lista B: [0.6,  0.3]

posición 1:  1.5  +  0.6  =  2.1
posición 2: -0.3  +  0.3  =  0.0

Resultado: [2.1, 0.0]
```

Primero con primero, segundo con segundo — como sumar dos filas de una
hoja de cálculo, columna por columna. Cuando se suma "vector del token" +
"vector de la posición", es exactamente esta misma operación — solo que
ahora las dos listas tienen nombre.

## Aclarando una duda real de la sesión: "¿esos números realmente saben que están en la posición 0?"

**No, todavía no.** La fila 0 de la tabla de posiciones es solo un par de
números al azar — no tiene ningún significado de "primer lugar" grabado.

Lo que da el significado no son los números en sí, sino la **regla de uso
consistente**: siempre, sin excepción, se usa la fila 0 para el primer
lugar de cualquier oración (analogía: un casillero de correo #0 no tiene
nada especial en la puerta, pero si siempre se mete ahí la primera carta
del día, cualquiera que lo revise sabrá qué llegó primero — no porque el
casillero lo supiera, sino porque la regla de uso es consistente).

El significado real aparece recién durante el **entrenamiento** (paso 7):
como la fila 0 se usa consistentemente, miles de veces, siempre para "lo
que va primero", cuando el modelo se equivoca y ajusta sus pesos, ajusta
específicamente la fila 0 cada vez que estaba razonando sobre "lo que va
primero". Con el tiempo, esa fila desarrolla un patrón de números útil para
esa tarea — no porque alguien lo programó, sino porque es la única fila que
consistentemente participó en ella.

**En este punto del proyecto, ambas tablas (tokens y posiciones) siguen
siendo ruido aleatorio.** Lo único "real" hasta ahora es la arquitectura —
la tubería y la regla consistente de qué se suma con qué.

## Aclarando otra duda real: tabla fija vs. lo que se usa por oración

La tabla de posiciones **no se crea pensando en la oración más larga de un
lote en particular**. Se crea una sola vez, con un tamaño máximo fijo del
proyecto (`max_seq_len`, ej. 64), decidido de antemano. Cada oración, sin
importar su largo, solo consulta tantas filas de esa tabla como tokens
tenga — el resto de la tabla simplemente no se toca esa vez, pero sigue
disponible por si otra oración es más larga.

## Ejemplo real, de punta a punta

Con `texto = 'Roja Partido'` (nota: "Roja" no se fusionó como token
completo en los 300 merges de BPE — quedó en caracteres sueltos, así que el
ejemplo usa `'R'` y `'o'`), `d_model=4` (para que quepa en pantalla):

```
=== texto real -> simbolos BPE -> ids ===
texto:    'Roja Partido'
simbolos: ['R', 'o']
ids:      [72, 168]

=== LIBRETA DE TOKENS (solo las 2 filas usadas) ===
  id 72  ('R') -> [0.56, 0.79, -0.18, -0.73]
  id 168 ('o') -> [0.95, 0.73, -1.46, -2.05]

=== LIBRETA DE POSICIONES (las 2 filas que existen) ===
  posición 0 -> [-0.72, -1.63,  1.41, -0.51]
  posición 1 -> [-2.52,  0.77,  1.51, -0.81]

=== SUMA, palabra por palabra ===
'R' (posición 0):
   vector token     : [ 0.56,  0.79, -0.18, -0.73]
 + vector posición   : [-0.72, -1.63,  1.41, -0.51]
 = vector final       : [-0.16, -0.84,  1.22, -1.24]

'o' (posición 1):
   vector token     : [ 0.95,  0.73, -1.46, -2.05]
 + vector posición   : [-2.52,  0.77,  1.51, -0.81]
 = vector final       : [-1.56,  1.51,  0.06, -2.85]
```

Resultado: una matriz de 2 filas × 4 columnas — cada fila es "una letra +
su lugar", lista para entrar a la siguiente pieza del transformer. Esto es
exactamente lo que corre `EmbeddingConPosicion` sobre el corpus real, solo
que con `d_model=128` y lotes enteros en vez de una palabra a la vez.

## Con datos completos del corpus (`d_model=128`)

```python
lote_textos = ["Marcador Partido 1 — México vs Sudáfrica",
               "¿Quién gana Partido 1?",
               "Marcador Partido 2 — Corea del Sur vs República Checa"]
# shape ids de entrada:  (3, 23)   (batch, seq_len)
# shape salida embedding: (3, 23, 128)   (batch, seq_len, d_model)
# parámetros entrenables: 35,456   →  254×128 (tabla tokens) + 23×128 (tabla posiciones)
```

## Código real (`src/transformer_model.py`)

```python
class EmbeddingConPosicion(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, ids):
        _, seq_len = ids.shape
        posiciones = torch.arange(seq_len, device=ids.device)
        return self.token_embedding(ids) + self.position_embedding(posiciones)
```

## Qué es "el embedding", en la cadena completa

```
"Marcador"  →  BPE  →  "Marcador</w>"  →  id (66)  →  EMBEDDING  →  [0.5, -1.2, ..., 0.3]  (128 números)
   texto      tokeniza    símbolo         número      la libreta      el vector final
```

"Embedding" nombra dos cosas relacionadas: **la libreta completa** (la tabla
`nn.Embedding`, el mecanismo) y **el vector resultado** de consultar una
página (lo que se obtiene). Por qué no basta con quedarse con el id solo:
el id `66` es una etiqueta arbitraria sin relación útil con otros ids (el
id 67 no es necesariamente parecido en significado); el embedding, tras
entrenar, sí captura parecido de significado — la misma idea de Fase 2 con
similitud coseno, pero el vector se construye y ajusta dentro del propio
modelo.

## Siguiente paso

[02-self-attention.md](02-self-attention.md) — cómo cada token "mira" a los
demás tokens de la secuencia para decidir qué información tomar de ellos.
