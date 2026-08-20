# Paso 6 — Ensamblar TransformerBlock y apilarlo

## Diseño

Junta todo lo construido en los pasos 2-5, en el orden exacto del diagrama
del panorama inicial:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, causal=True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, causal=causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))   # residual + layer norm, tras attention
        x = self.norm2(x + self.ffn(x))          # residual + layer norm, tras feed-forward
        return x
```

Cada línea del `forward`, contra lo ya explicado:
- `x + self.attention(x)` → la conexión residual: "no reemplaces, suma".
- `self.norm1(...)` → layer norm, reescala los números tras la suma.
- Se repite el mismo patrón para feed-forward.

`causal=True` por defecto, porque el modelo es generativo (predice palabra
por palabra) — cada bloque necesita el causal masking activo dentro de su
`MultiHeadAttention`.

## Apilar varios bloques

Un modelo real no usa un solo bloque — usa varios apilados en secuencia: la
salida de un bloque es la entrada del siguiente.

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.embedding = EmbeddingConPosicion(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return x
```

Esto, en su forma inicial, todavía no incluye la "cabeza de salida"
(convertir el vector final en una predicción de próxima palabra) — eso es
el paso 7. `TransformerLM` es la "columna vertebral" del modelo: de ids a
vectores enriquecidos con todo el contexto, pasando por `num_layers`
bloques (4-6 según el roadmap).

## Verificación con datos reales

Con `vocab_size=254`, `d_model=128`, `num_heads=4`, `d_ff=512`,
`num_layers=4`, sobre 4 preguntas reales del corpus:

```
Shape ids entrada: (4, 23)   (batch, seq_len)
Shape salida:      (4, 23, 128)   (batch, seq_len, d_model) -- misma forma, tras 4 bloques

Parametros embedding:      35,456
Parametros por bloque:     197,760
Parametros 4 bloques:      791,040
TOTAL parametros modelo:  826,496
```

(El conteo final tras agregar la cabeza de salida en el paso 7 sube a
864,510.)

**Lo que demuestra:**
- Shape se preserva de principio a fin: `(4, 23)` ids entran, `(4, 23,
  128)` vectores salen — 4 oraciones reales del corpus, pasando íntegramente
  por 4 bloques transformer apilados, sin errores de dimensión.
- 826,496 parámetros — dentro del rango del roadmap (~1-10M para hardware
  básico), con margen para crecer si se quisiera más capas o dimensiones.
- Feed-forward domina el conteo por bloque (como se anticipó en el paso 5):
  de los ~198K parámetros por bloque, la mayoría son de `FeedForward`, no
  de `MultiHeadAttention`.

## Resumen de la arquitectura completa hasta aquí

```
texto → BPE → ids → EmbeddingConPosicion → 4× TransformerBlock
                                             (MultiHeadAttention causal
                                              + residual + LayerNorm
                                              + FeedForward
                                              + residual + LayerNorm)
      → vectores finales de 128 dimensiones por token
```

Lo que falta para que esto haga algo útil: convertir esos vectores finales
en una predicción real de "cuál es la siguiente palabra" (paso 7, cabeza de
salida), y luego el loop de entrenamiento que le da significado a todos
estos pesos que hasta ahora son puro ruido aleatorio.

## Siguiente paso

[07-entrenamiento.md](07-entrenamiento.md) — cabeza de salida, cross-entropy
loss, y el loop de entrenamiento real sobre el corpus.
