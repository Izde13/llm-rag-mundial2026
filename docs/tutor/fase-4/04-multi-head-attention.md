# Paso 4 — Multi-head attention

## El problema: una sola "cabeza" solo puede fijarse en un tipo de relación a la vez

En el paso 2 se construyó **una** self-attention: una sola `W_query`, una
sola `W_key`, una sola `W_value`. Con eso, cada palabra calcula **un solo**
reparto de atención hacia las demás.

Pero el lenguaje tiene *varios tipos de relación* ocurriendo a la vez entre
las mismas palabras. Con `"la tarjeta amarilla"`:

- **Concordancia gramatical** (¿"amarilla" concuerda en género/número con
  "tarjeta"?).
- **Significado/tema** ("tarjeta" + "amarilla" evocan "sanción en un
  partido").
- **Posición/cercanía** (palabras vecinas suelen relacionarse más que las
  lejanas).

Con una sola cabeza, todas esas señales tienen que competir y mezclarse en
el mismo reparto de porcentajes — el modelo se ve forzado a promediar todo
en una sola vista, perdiendo matices.

## La solución: varias cabezas en paralelo

**Multi-head attention** repite el mecanismo de self-attention varias
veces, en paralelo, con matrices `W_query`/`W_key`/`W_value` distintas e
independientes para cada copia — y al final junta los resultados:

```
                    ┌─── Cabeza 1 (sus propias W_q, W_k, W_v) ───┐
entrada  ──────────►├─── Cabeza 2 (sus propias W_q, W_k, W_v) ───┤──► concatenar ──► mezclar
                    └─── Cabeza 3 (sus propias W_q, W_k, W_v) ───┘
```

Cada cabeza, al tener sus propias matrices (inicializadas distinto, al
azar), tiene libertad de **especializarse en un patrón distinto** durante
el entrenamiento — una podría terminar prestando más atención a
concordancia gramatical, otra a relaciones temáticas, etc. Nadie asigna esa
especialización a mano — emerge del mismo proceso de "reducir el error"
del paso 2, solo que ahora hay varias "vistas" compitiendo y
complementándose.

**No se usa el `d_model` completo en cada cabeza.** Si `d_model=128` y hay
4 cabezas, cada cabeza trabaja con `128/4 = 32` dimensiones — se **reparte**
el espacio total entre las cabezas, no se multiplica el costo por 4:

```
d_model = 128, num_heads = 4  →  cada cabeza usa d_k = 128/4 = 32

Cabeza 1: [batch, seq_len, 32]  ┐
Cabeza 2: [batch, seq_len, 32]  │
Cabeza 3: [batch, seq_len, 32]  ├─ concatenar ──► [batch, seq_len, 128]  ──► una capa lineal final
Cabeza 4: [batch, seq_len, 32]  ┘
```

La capa lineal final (`W_output`) tras concatenar existe para que el modelo
pueda **mezclar** la información de las distintas cabezas entre sí, en vez
de dejarlas como bloques separados sin comunicación.

**Dato curioso:** el paper de 2017 usó 8 cabezas con `d_model=512` (64 dims
por cabeza). GPT-3 usa 96 cabezas con `d_model=12,288`. Con el `d_model`
pequeño de este proyecto (128), 4 cabezas es razonable — suficiente para
ver el concepto sin dejar a cada cabeza con muy pocas dimensiones.

## Diseño de código: una matriz grande, cortada por cabeza

En vez de crear cabezas como instancias separadas de `SelfAttention` (lo
que recrearía innecesariamente matrices de tamaño completo por cabeza), la
forma estándar de la industria es: **una sola matriz grande** que produce
Q, K, V para *todas* las cabezas de una vez, y luego se "corta" en pedazos
por cabeza.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, causal=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)
        self.W_output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # partir d_model en (num_heads, d_k) y mover num_heads antes de seq_len
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if self.causal:
            mascara = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mascara, float("-inf"))

        pesos_atencion = torch.softmax(scores, dim=-1)
        salida = pesos_atencion @ V

        salida = salida.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_output(salida)
```

**Por qué esto es matemáticamente igual a "varias `SelfAttention`
separadas":** dividir una matriz grande `W_query` de `d_model×d_model` en
pedazos de `d_model×d_k` (uno por cabeza) es equivalente a tener
`num_heads` matrices pequeñas independientes — solo que calculadas en una
sola operación de PyTorch por eficiencia, en vez de un bucle `for`. El
resultado numérico es el mismo; cambia solo la implementación.

`.view(...).transpose(1,2)` "reparte" las 128 columnas en 4 grupos de 32 y
los reordena para que cada cabeza opere de forma independiente en su propia
porción — mecánica de tensores (reorganizar shape), no una operación
matemática nueva.

## Ejemplo real, verificado: `"la tarjeta amarilla"`, 2 cabezas

Con `d_model=8`, `num_heads=2` (`d_k=4`), causal activo en ambas:

```
=== Cabeza 0 (causal) ===
  la</w>          100%      0%      0%
  tarjeta</w       46%     54%      0%
  amarilla</       37%     23%     40%

=== Cabeza 1 (causal) ===
  la</w>          100%      0%      0%
  tarjeta</w        6%     94%      0%
  amarilla</       20%     54%     26%
```

**Lo que demuestra:**
- Shapes correctos: entrada `(1, 3, 8)` → salida `(1, 3, 8)` — multi-head
  no cambia la forma final, solo lo que pasa por dentro.
- Causal masking se respeta en ambas cabezas — ninguna fila tiene atención
  hacia el futuro.
- Las dos cabezas "opinan distinto" sobre la misma oración: en la fila de
  `'tarjeta'`, la Cabeza 0 reparte casi parejo (46%/54%) entre `'la'` y sí
  misma, mientras la Cabeza 1 se concentra fuertemente en sí misma (94%).
  Con pesos sin entrenar esa diferencia es aleatoria, pero es la prueba de
  que la arquitectura sí da espacio a que las cabezas se especialicen de
  forma distinta una vez que el entrenamiento las ajuste.
- 256 parámetros entrenables con `d_model=8`: 4 matrices de `8×8`
  (`W_query, W_key, W_value, W_output`) = `4 × 64 = 256`. Con `d_model=128`
  real, serían `4 × 128×128 = 65,536` parámetros solo en esta pieza.

## Siguiente paso

[05-feedforward-residual-layernorm.md](05-feedforward-residual-layernorm.md)
— feed-forward (procesar cada palabra por su cuenta), conexión residual (no
perder información al apilar capas) y layer norm (estabilidad numérica).
