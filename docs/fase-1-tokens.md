# Fase 1 — Tokens

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): entender
y construir, desde cero, el proceso de tokenización — convertir el texto del
corpus en secuencias de números que un modelo pueda procesar. Se implementan
dos niveles: char-level (base conceptual) y BPE (subword, el usado en la
industria), y se comparan sobre el corpus completo generado en Fase 0.

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/tokenizer_char.py`](../src/tokenizer_char.py) | Tokenizador char-level: vocabulario, codificar, decodificar |
| [`src/tokenizer_bpe.py`](../src/tokenizer_bpe.py) | Tokenizador BPE: entrenamiento por fusión de pares, tokenización con merges aprendidos |
| [`src/tokenizer_stats.py`](../src/tokenizer_stats.py) | Compara ambos tokenizadores sobre el corpus completo (entregable de la fase) |
| [`tests/test_tokenizers.py`](../tests/test_tokenizers.py) | Pruebas de reversibilidad y consistencia de ambos tokenizadores |

## 1. Qué es un token y por qué existe

Una red neuronal solo procesa números — un **token** es la unidad mínima en
la que se parte un texto para asignarle un id entero, formando un
**vocabulario** (todas las piezas distintas que el modelo puede llegar a ver).
Hay tres granularidades posibles:

- **Char-level**: cada carácter es un token. Vocabulario pequeño, secuencias
  largas.
- **Word-level**: cada palabra es un token. Secuencias cortas, pero el
  vocabulario explota y no hay forma de representar una palabra nueva
  (requiere un token `<UNK>`, con pérdida de información).
- **Subword-level (BPE)**: punto intermedio, usado en la industria moderna
  (GPT, LLaMA). Nunca hay una palabra "imposible de tokenizar", porque en el
  peor caso siempre se puede descomponer hasta el carácter.

Se implementaron char-level y BPE, en ese orden: char-level primero por ser
trivial de razonar (texto → ids → texto), BPE después para entender el
algoritmo de fusión por dentro, sin depender de librerías como `tiktoken` o
`sentencepiece`.

## 2. Tokenizador char-level (`src/tokenizer_char.py`)

- `construir_vocabulario(textos)`: recorre todos los caracteres únicos del
  corpus y asigna un id a cada uno. Devuelve `char_a_id` y `id_a_char` (se
  necesitan ambos: codificar y decodificar).
- `codificar(texto, char_a_id)` / `decodificar(ids, id_a_char)`: conversión
  en ambas direcciones. Propiedad exigida y verificada:
  `decodificar(codificar(x)) == x`.

Se tokenizó el campo `pregunta` del corpus (texto en lenguaje natural más
representativo, a diferencia de ids o nombres propios repetidos).

**Resultado sobre el corpus completo (6582 preguntas):**

```
tamaño_vocabulario: 78 caracteres
total_caracteres_en_corpus: 322,317
longitud_promedio_pregunta: 48.97
longitud_máxima_pregunta: 190
```

Los 78 caracteres son coherentes con lo detectado en Fase 0: alfabeto
español + puntuación + los caracteres especiales verificados como UTF-8
correcto (`° ¿ Ó á é í ñ ó ú — ⁷`).

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/tokenizer_char.py
```

Salida guardada en `data/processed/tokenizer_char_stats.json` (vocabulario
completo + estadísticas sobre el corpus completo + 3 ejemplos ilustrativos
de codificación/decodificación).

## 3. Tokenizador BPE (`src/tokenizer_bpe.py`)

BPE nació en 1994 como algoritmo de compresión de datos (Philip Gage), y se
reaprovechó para tokenización de lenguaje en 2015 (Sennrich et al.). El
algoritmo, implementado a mano:

1. **`preparar_palabras(textos)`**: parte cada palabra en caracteres +
   marcador de fin de palabra `</w>` (evita fusionar el final de una palabra
   con el inicio de la siguiente). Cuenta frecuencia de palabras únicas, no
   recorre el corpus completo en cada iteración (optimización necesaria para
   miles de iteraciones).
2. **`contar_pares(vocab_palabras)`**: frecuencia de cada par de símbolos
   adyacentes, ponderada por la frecuencia de la palabra.
3. **`fusionar_par(par, vocab_palabras)`**: reemplaza el par ganador por su
   símbolo fusionado en todas las palabras.
4. **`entrenar_bpe(textos, num_merges)`**: repite contar → fusionar
   `num_merges` veces, guardando la lista **ordenada** de merges.
5. **`tokenizar_con_merges(texto, merges)`**: aplica las reglas aprendidas,
   en el mismo orden en que se entrenaron, a texto nuevo.

**Decisiones de diseño explicadas:**
- Se implementó sobre caracteres Unicode, no sobre bytes crudos (a
  diferencia de GPT-2+, que usa byte-level BPE para garantizar cobertura
  universal de cualquier alfabeto/emoji). Para este corpus, 100% texto en
  español con encoding ya verificado limpio en Fase 0, la capa de bytes solo
  agregaría complejidad accidental sin aportar aprendizaje.
- `num_merges = 300`: punto de partida razonable para un corpus de este
  tamaño (6582 preguntas). Cada merge es una regla de reescritura de dos
  símbolos adyacentes en uno — el orden importa porque una merge posterior
  puede operar sobre el resultado de una merge anterior (fusiones en
  cascada, ej. `(i,c)` → `ic`, luego `(ic, a</w>)` → `ica</w>`).
- Una merge no es un evento único: es una regla reutilizable que se
  descubre una vez durante el entrenamiento y se reaplica, en el mismo
  orden, a cualquier texto nuevo — por eso se guardan `merges` (el proceso)
  y `vocabulario_final` (el resultado) por separado.

**Resultado sobre el corpus completo:**

```
num_merges_aplicados: 300
tamaño_vocabulario_bpe: 252 símbolos
```

Ejemplo real (`"Marcador Partido 1 — México vs Sudáfrica"`):

```
['Marcador</w>', 'Partido</w>', '1</w>', '—</w>', 'M', 'é', 'x', 'i', 'c',
 'o</w>', 'vs</w>', 'Su', 'd', 'á', 'f', 'r', 'ica</w>']
```

Palabras muy frecuentes en el corpus (`"Marcador"`, `"Partido"`, `"vs"`)
quedaron colapsadas en un solo token. `"México"` quedó casi sin fusionar
pese a mencionarse seguido — sus pares de caracteres específicos (`x-i`,
`é-x`) compitieron cada ronda contra terminaciones mucho más comunes en
español (`-ado`, `ar</w>`, etc.) en todo el corpus, y perdieron las 300
veces. Ilustra que BPE es pura estadística de frecuencia, sin ninguna
noción de qué palabras son "importantes" semánticamente.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/tokenizer_bpe.py
```

Salida guardada en `data/processed/tokenizer_bpe_stats.json` (merges en
orden, vocabulario final, ejemplos).

## 4. Comparación final (`src/tokenizer_stats.py`)

Entregable de la fase: corre ambos tokenizadores sobre el corpus completo
(6582 preguntas, sin muestreo) reutilizando las funciones de los dos
scripts anteriores, y cuantifica el beneficio de BPE.

| Métrica | Char-level | BPE |
|---|---|---|
| Tamaño vocabulario | 78 | 252 |
| Longitud promedio (tokens/pregunta) | 48.97 | 11.07 |
| Longitud máxima | 190 | 85 |
| Total tokens en corpus | 322,317 | 72,873 |

**Factor de compresión BPE vs. char-level: 4.42x**

Este resultado importa para Fase 4: el transformer tiene costo computacional
que crece cuadráticamente con la longitud de secuencia (self-attention
compara cada token con todos los demás). Secuencias ~4.4x más cortas con
BPE son directamente relevantes dado el contexto corto objetivo (64-128
tokens) y el hardware limitado del proyecto.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/tokenizer_stats.py
```

Salida guardada en `data/processed/tokenizer_comparacion.json`.

## 5. Verificación (tests)

[`tests/test_tokenizers.py`](../tests/test_tokenizers.py) verifica la
propiedad no-negociable de cualquier tokenizador — reversibilidad — y
consistencia del proceso de fusión de BPE:

- `test_char_level_es_reversible`: `decodificar(codificar(x)) == x`.
- `test_char_level_vocabulario_cubre_todos_los_caracteres`: el vocabulario
  construido incluye todos los caracteres presentes en el texto de origen.
- `test_bpe_no_produce_tokens_vacios`: ninguna fusión genera un token `""`.
- `test_bpe_conserva_todos_los_caracteres_originales`: reconstruir el texto
  a partir de los tokens BPE (quitando marcadores `</w>`) recupera el
  original.
- `test_bpe_reduce_o_mantiene_longitud_frente_a_char_level`: BPE nunca
  produce más tokens que caracteres tiene el texto.

**Cómo correr las pruebas:**

```bash
./.venv/Scripts/uv.exe run pytest tests/test_tokenizers.py -v
```

```
tests/test_tokenizers.py::test_char_level_es_reversible PASSED
tests/test_tokenizers.py::test_char_level_vocabulario_cubre_todos_los_caracteres PASSED
tests/test_tokenizers.py::test_bpe_no_produce_tokens_vacios PASSED
tests/test_tokenizers.py::test_bpe_conserva_todos_los_caracteres_originales PASSED
tests/test_tokenizers.py::test_bpe_reduce_o_mantiene_longitud_frente_a_char_level PASSED
```

## 6. Pendiente

- Fase 2 del roadmap: embeddings — de tokens a vectores semánticos, usando
  un modelo pre-entrenado liviano (`all-MiniLM-L6-v2`) (ver
  [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
- Nota para cuando se entrene el LLM en Fase 4: la unidad real para evaluar
  si el corpus "alcanza" es la cantidad de tokens (72,873 en BPE), no la
  cantidad de filas (6582) — dato que quedó pendiente de Fase 0.
