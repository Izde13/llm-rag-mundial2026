# Fase 0 — Setup y datos

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): dejar
el repo estructurado, un entorno Python reproducible, y el corpus de datos
del Mundial 2026 exportado a un JSON limpio, listo para usarse en las fases
siguientes (tokenización, embeddings, etc.).

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/explorar_excel.py`](../src/explorar_excel.py) | Diagnóstico: lista hojas, columnas, caracteres especiales, verifica encoding |
| [`src/data_prep.py`](../src/data_prep.py) | Pipeline: cruza las hojas del Excel y genera `data/processed/corpus.json` |
| [`tests/test_data_prep.py`](../tests/test_data_prep.py) | Pruebas automatizadas del pipeline (`pytest`) |

## 1. Estructura del repositorio

```
IA-tech/
├── data/
│   ├── raw/          # datos originales, intocables
│   └── processed/    # datos generados por scripts, se puede borrar y regenerar
├── src/               # código fuente del proyecto
├── notebooks/         # exploración interactiva (Jupyter), para fases futuras
├── tests/             # pruebas automatizadas
├── docs/              # documentación por fase (este archivo vive aquí)
├── pyproject.toml      # definición del proyecto y sus dependencias
├── uv.lock             # versiones exactas de cada dependencia (reproducibilidad)
└── .venv/              # entorno virtual (no se versiona en git)
```

**Por qué esta separación:**
- `data/raw/` nunca se edita a mano. Si algo sale mal en el procesamiento,
  siempre se puede volver al origen y regenerar todo desde cero.
- `data/processed/` es 100% derivable de `data/raw/` vía código en `src/`.
  Nunca contiene información que no se pueda reconstruir.

## 2. Entorno virtual y gestor de dependencias

Python instala paquetes de forma global por defecto, lo cual es riesgoso:
si dos proyectos en la misma máquina necesitan versiones distintas de una
librería, chocan entre sí. Un **entorno virtual** (`venv`) es una carpeta
aislada con su propia copia de Python y sus propias librerías.

Pasos que se siguieron:

```bash
# 1. Crear el entorno virtual del proyecto
python -m venv .venv

# 2. Instalar uv DENTRO del entorno virtual (no globalmente)
./.venv/Scripts/python.exe -m pip install --upgrade pip uv

# 3. Inicializar el proyecto con uv (genera pyproject.toml, .gitignore, git init)
./.venv/Scripts/uv.exe init --no-workspace --python 3.13 --name ia-tech

# 4. Agregar las dependencias necesarias para leer Excel/CSV
./.venv/Scripts/uv.exe add pandas openpyxl
```

**Por qué `uv` y no solo `pip`:** `uv` genera un *lockfile* (`uv.lock`) que
fija las versiones exactas de cada dependencia y sus sub-dependencias. Esto
significa que cualquier persona que clone el repo y corra `uv sync` obtiene
exactamente el mismo entorno, sin sorpresas de "en mi máquina sí funciona".

**Nota de higiene importante:** en un primer intento se instaló `uv` con
`pip install uv` directo, sin darse cuenta de que el Python activo en ese
momento era el global del sistema (no un venv). Se corrigió desinstalando
`uv` del Python global (`pip uninstall uv`) y reinstalándolo dentro de
`.venv/`. Lección: siempre confirmar en qué entorno se está parado antes de
instalar algo (`python -c "import sys; print(sys.prefix)"` — si coincide con
la instalación base de Python, no es un venv).

`uv init` generó por su cuenta `.gitignore` (ya excluye `.venv/`,
`__pycache__/`, etc.) e inicializó el repositorio git.

## 3. Origen de los datos

El corpus proviene del proyecto ["Polla Mundial 2026"](../../API%20POLLA%20MUNDIAL),
un sistema propio (n8n + Google Sheets + Apps Script) que gestiona un pool
de predicciones entre amigos para el Mundial 2026. La base de datos real
vive en Google Sheets; para este proyecto se usó una exportación manual a
Excel: `Polla Mundial 2026.xlsx`, copiada a `data/raw/` sin modificar.

El libro de Excel tiene 9 hojas, reflejando el esquema documentado en el
README de ese proyecto:

| Hoja | Filas | Contenido |
|---|---|---|
| `PARTICIPANTES` | 18 | id, nombre, email de cada jugador |
| `JORNADAS` | 15 | fechas del torneo (J00 a J14) |
| `PREGUNTAS` | 442 | catálogo de preguntas y reglas de puntos |
| `RESPUESTAS` | 6582 | respuesta cruda de cada participante por pregunta |
| `PUNTUACIONES` | 6582 | respuesta + resultado oficial + puntos ya evaluados |
| `TABLA_GENERAL` | 17 | ranking final |
| `RESULTADOS_OFICIALES` | 442 | resultado real de cada pregunta |
| `LOG_ERRORES` | 0 | vacía |
| `LOG_AUDITORIA` | 1 | prácticamente vacía |

`PUNTUACIONES` es la hoja más rica para el corpus: ya cruza participante,
pregunta, respuesta, resultado oficial y puntos en una sola fila.

## 4. Investigación: ¿el texto está corrupto?

Al inspeccionar los datos por primera vez en la terminal, los acentos y la
`¿` se veían corruptos (`�Qui�n pasa 1� del Grupo A?`). Antes de "arreglar"
nada, se investigó la causa con evidencia, en vez de asumir un fix genérico.
Todo ese diagnóstico quedó convertido en código reutilizable en
[`src/explorar_excel.py`](../src/explorar_excel.py), en vez de vivir solo
en el historial de una conversación:

- `listar_hojas()` / `describir_hojas()`: qué pestañas tiene el libro y qué
  columnas trae cada una.
- `caracteres_no_ascii()`: recorre todas las hojas y todas las columnas de
  texto y devuelve el conjunto de caracteres fuera del rango ASCII básico
  (`ord(ch) > 127`). Se usó para acotar el universo de caracteres
  especiales antes de decidir si había un problema real de encoding.
- `verificar_encoding_correcto()`: comprueba que el texto no contenga el
  carácter de reemplazo Unicode (`U+FFFD`, `�`), que es la señal real de
  pérdida de bytes — a diferencia de un carácter que simplemente se ve raro
  en la consola.

Pasos del diagnóstico:

1. Se probó re-interpretar los bytes con `texto.encode('latin-1').decode('utf-8')`
   y variantes (`cp1252`, `mac_roman`) — todas fallaron con
   `invalid start byte`, indicando que no era un simple problema de
   doble-decodificación reversible.
2. Se inspeccionaron los *code points* exactos de los caracteres rotos
   (`ord(ch)`) y se confirmó que correspondían a bytes Latin-1 reales
   (ej. `0xbf` = `¿`, `0xe9` = `é`), no al carácter de reemplazo Unicode.
   Esto sugería pérdida real de bytes, no solo mala visualización.
3. Se corrió `caracteres_no_ascii()` sobre las 9 hojas: solo 11 caracteres
   distintos en total (`° ¿ Ó á é í ñ ó ú — ⁷`), un universo pequeño y
   manejable.
4. **Prueba decisiva** (`verificar_encoding_correcto()`): se escribió uno
   de los textos "rotos" a un archivo `.txt` en UTF-8 y se leyó de vuelta.
   El texto apareció perfectamente correcto: `¿Quién pasa 1° del Grupo A?`.

**Conclusión:** el archivo Excel siempre tuvo el texto en UTF-8 correcto.
El "mojibake" que se veía era un artefacto de la terminal de Windows
(que usa `cp1252` para mostrar texto), no un defecto real de los datos.
No se aplicó ninguna transformación de encoding al corpus — hubiera sido
un arreglo innecesario y potencialmente dañino sobre datos que ya estaban
bien.

**Lección general:** nunca confiar en cómo se ve el texto en la consola
para diagnosticar encoding. Verificar escribiendo a un archivo o
inspeccionando los *code points* (`ord(ch)`) directamente. Por esa misma
razón, `explorar_excel.py` fuerza `sys.stdout.reconfigure(encoding="utf-8")`
en su función `main()` — sin eso, el propio script de diagnóstico fallaría
al imprimir caracteres como `⁷` en una consola Windows con `cp1252`.

**Cómo correr el diagnóstico completo:**

```bash
./.venv/Scripts/uv.exe run python src/explorar_excel.py
```

## 5. Script de normalización: `src/data_prep.py`

[`src/data_prep.py`](../src/data_prep.py) cruza (`merge`, equivalente a
`JOIN` en SQL) las hojas `PUNTUACIONES` + `PREGUNTAS` + `PARTICIPANTES` +
`JORNADAS` para producir un registro plano y legible por fila, y lo
exporta a `data/processed/corpus.json`. Sus funciones principales:

- `cargar_hojas()`: lee las 4 hojas necesarias del Excel y las devuelve
  como un diccionario de DataFrames.
- `construir_corpus(hojas)`: hace los 3 merges (contra `PREGUNTAS` por
  `pregunta_id`, contra `PARTICIPANTES` por email, contra `JORNADAS` por
  `jornada_id`) y selecciona/renombra las columnas finales.
- `main()`: orquesta todo y escribe `data/processed/corpus.json`.

**Decisiones de diseño explicadas:**
- `how="left"` en los merges: se conservan todas las filas de `PUNTUACIONES`
  aunque no hubiera match (en la práctica sí hubo match en el 100% de los
  casos, verificado con los tests — ver sección 6).
- `participante_id` en `PUNTUACIONES` resultó ser en realidad el email del
  participante, no el `id` de `PARTICIPANTES` — por eso el cruce se hace
  contra la columna `email`, no `id`.
- `ensure_ascii=False` en `json.dump`: evita que acentos/ñ se guarden como
  escapes tipo `ñ`; el JSON queda legible como texto plano.
- `indent=2`: formato legible por humanos, apropiado para un repo educativo
  (a costa de un archivo más grande que el mínimo).
- `default=str`: red de seguridad para columnas de tipo fecha
  (`pandas.Timestamp`) que el `json` estándar no sabe serializar solo.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/data_prep.py
```

## 6. Resultado y verificación

```
Corpus generado: data/processed/corpus.json (6582 registros)
```

La verificación de que ningún registro quedara con campos vacíos por falta
de match en los merges quedó convertida en pruebas automatizadas en
[`tests/test_data_prep.py`](../tests/test_data_prep.py) (con `pytest`):

- `test_corpus_no_tiene_campos_vacios`: para cada columna obligatoria
  (`jornada`, `participante`, `pregunta`), cuenta cuántos registros
  quedaron vacíos (`NaN`) después del merge y falla si hay al menos uno.
- `test_corpus_tiene_las_columnas_esperadas`: confirma que el DataFrame
  final tenga todas las columnas que espera el resto del pipeline.
- `test_corpus_tiene_registros`: chequeo mínimo de que el corpus no salió
  vacío.

**Cómo correr las pruebas:**

```bash
./.venv/Scripts/uv.exe run pytest tests/ -v
```

```
tests/test_data_prep.py::test_corpus_no_tiene_campos_vacios PASSED
tests/test_data_prep.py::test_corpus_tiene_las_columnas_esperadas PASSED
tests/test_data_prep.py::test_corpus_tiene_registros PASSED
```

Cada registro del corpus final tiene esta forma:

```json
{
  "jornada_id": "J01",
  "jornada": "Jornada 1 - Fase de Grupos",
  "participante": "Jugador Ejemplo",
  "pregunta_id": "J01_P01_RESULTADO",
  "pregunta": "¿Quién gana Partido 1?",
  "respuesta_participante": "México",
  "resultado_oficial": "México",
  "puntos": 2
}
```

## 7. Pendiente

- Descargar y colocar en `data/raw/` cualquier hoja adicional de Google
  Sheets que no esté ya incluida en `Polla Mundial 2026.xlsx`, si aplica.
- Fase 1 del roadmap: tokenización del corpus (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
