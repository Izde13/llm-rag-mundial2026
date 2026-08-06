# Paso 8 — Generación de texto

## La idea central: repetir "predecir el siguiente token" una y otra vez

Durante el entrenamiento (paso 7), el modelo solo hizo una cosa: dado un
pedazo de texto, predecir qué símbolo viene después. **Generar texto es
literalmente usar esa misma habilidad, en un bucle**: se le da un inicio,
predice el siguiente token, se lo agrega al texto, y se le vuelve a
preguntar "¿y ahora qué sigue?" — usando el texto ya extendido como nueva
entrada.

```
entrada: "Marcador Partido"
paso 1: modelo predice el siguiente token → " 1"
        texto: "Marcador Partido 1"
paso 2: modelo predice el siguiente token → "</w>"
        texto: "Marcador Partido 1 "
paso 3: modelo predice el siguiente token → "—"
        texto: "Marcador Partido 1 —"
...(se repite hasta un límite de tokens o un token de "fin")
```

Esto se llama generación **autoregresiva** — cada nuevo token depende de
todos los anteriores, incluidos los que el propio modelo generó, no solo
los originales.

## Cómo elegir "el" token entre 254 probabilidades

La salida del modelo, tras softmax, es una distribución de probabilidad
sobre los 254 símbolos — no "un" token, sino 254 porcentajes. Tres
estrategias, de menos a más aleatoria:

| Estrategia | Cómo funciona | Efecto |
|---|---|---|
| **Greedy (voraz)** | Siempre elige el token con mayor probabilidad | Determinista (mismo input → mismo output siempre), pero puede caer en repeticiones ("el el el el...") |
| **Sampling con temperatura** | Elige al azar, ponderado por las probabilidades, con un parámetro que "aplana" o "afila" la distribución | Más variedad, texto menos robótico |
| **Top-k / top-p** | Como sampling, pero restringido a los k tokens más probables (o al conjunto mínimo que suma p% de probabilidad) | Evita elegir tokens absurdos de baja probabilidad, manteniendo variedad |

**Temperatura, sin fórmula:**
- Baja (ej. 0.5) → el modelo se vuelve más "conservador", casi como greedy
  — elige casi siempre lo más probable.
- Alta (ej. 1.5) → el modelo se vuelve más "arriesgado" — más chance a
  opciones menos probables, texto más variado pero con más riesgo de
  generar cosas sin sentido.
- 1.0 → usa las probabilidades tal cual salieron del softmax, sin
  modificar.

**Decisión de diseño:** dado que es un modelo pequeño entrenado sobre un
corpus chico y repetitivo (plantillas), se implementaron **greedy** y
**sampling con temperatura**, ambas en la misma función, para poder
comparar.

## Código real (`src/generar.py`)

```python
def generar(modelo, texto_inicial, merges, simbolo_a_id, id_a_simbolo,
            max_seq_len, max_nuevos_tokens=20, temperatura=1.0, greedy=False):
    ids = codificar_ids(texto_inicial, merges, simbolo_a_id)

    with torch.no_grad():
        for _ in range(max_nuevos_tokens):
            entrada = torch.tensor([ids[-max_seq_len:]])   # recorta al contexto maximo
            logits = modelo(entrada)
            ultimo_logit = logits[0, -1]                    # solo interesa la ultima posicion

            if greedy:
                siguiente_id = torch.argmax(ultimo_logit).item()
            else:
                probs = torch.softmax(ultimo_logit / temperatura, dim=-1)
                siguiente_id = torch.multinomial(probs, num_samples=1).item()

            ids.append(siguiente_id)

    return decodificar_ids(ids, id_a_simbolo)
```

**Puntos del diseño:**
- **`entrada = ids[-max_seq_len:]`**: recorta a los últimos `max_seq_len`
  tokens generados (64 en este proyecto). Necesario porque el modelo se
  entrenó con secuencias de máximo 64 posiciones (`position_embedding`
  solo tiene 64 filas) — no puede procesar una entrada más larga.
- **`logits[0, -1]`**: de todos los logits que devuelve el modelo (uno por
  cada posición de la secuencia), solo importa el de la **última**
  posición: la predicción de "qué sigue después de todo lo que ya tengo".
- **`torch.argmax(...)`**: implementación literal de "greedy" — elige el
  índice con el valor más alto, sin aleatoriedad.
- **`torch.multinomial(probs, ...)`**: implementación de "sampling" — elige
  un índice al azar, con probabilidad proporcional a `probs` (los tokens
  con mayor probabilidad tienen más chance, pero no es garantizado).
- **Ineficiencia intencional a notar:** en cada iteración se reprocesa
  **toda** la secuencia desde el principio (no solo el token nuevo) — la
  forma más simple de implementarlo, aunque no la más eficiente (la
  industria usa "KV-caching" para no recalcular lo ya procesado; queda
  fuera de alcance de este proyecto educativo).

## Resultado real, con el modelo ya entrenado

Con `"Marcador Partido"`, `"¿Quién"`, `"Roja"` como inicios:

```
Marcador Partido  (greedy):        Marcador Partido 11 — España vs Cabo Verde Marfil vs Cabo To
¿Quién            (greedy):        ¿Quién gana Partido 11? ? ? ? (parte del combo Top 4
Roja              (sampling, 0.5): Roja en orden eón? (pón ita de penales) vs Catar 4:
```

**Lo que aprendió bien, claramente:**
- Reconoce y reproduce la estructura de las plantillas del corpus:
  `"Marcador Partido N — Equipo1 vs Equipo2"`, `"¿Quién gana Partido N?"`,
  `"(parte del combo Top ..."`.
- Genera nombres de países reales del corpus (España, Cabo Verde, México,
  Corea del Sur, Senegal, Alemania, Argentina, Francia) — aprendió el
  "vocabulario temático" del dominio.
- `"Roja"` (que nunca se fusionó como token BPE completo en Fase 1) el
  modelo sigue asociándola correctamente con contexto de tarjetas/faltas.

**Lo que se nota poco entrenado / limitado:**
- Repite países sin sentido dentro de la misma oración (`"España vs Cabo
  Verde Marfil vs Cabo"`) — mezcla plantillas de distintas preguntas en vez
  de completar una sola de forma coherente.
- Con `greedy`, tiende a caer en bucles (`"? ? ?"`) — exactamente el
  problema anticipado al hablar de esa estrategia.
- `temperatura=1.0` genera texto visiblemente más "alocado" que
  `temperatura=0.5` (mezcla más países, saltos más raros de tema) — la
  temperatura hace justo lo esperado: más "riesgo" = más variedad, pero
  menos coherencia.

**Conclusión pedagógica, la más importante:** el modelo **no memorizó el
corpus al pie de la letra** (no devuelve preguntas exactas ya vistas) —
genera variaciones nuevas combinando patrones aprendidos, con errores
típicos de un modelo pequeño (864,510 parámetros, corpus chico) entrenado
pocas épocas. Es exactamente lo esperado del roadmap: "modelo que genera
texto con sabor al corpus", ni más ni menos.

## Cierre de la fase

Con este paso queda completo el recorrido de la Fase 4: de un id de token
suelto (paso 1) a un modelo que genera texto reconociblemente parecido al
corpus del mundial (paso 8), pasando por cada pieza de la arquitectura
transformer construida y verificada a mano, número por número, contra
datos reales.

Lo que sigue en el roadmap es la Fase 5 (RAG completo): integrar el vector
store de Fase 3 (retrieval) con este LLM propio (generación), inyectando
contexto recuperado en el prompt antes de generar.
