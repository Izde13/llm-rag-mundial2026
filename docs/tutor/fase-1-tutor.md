# Tutor — Fase 1: Tokens

## Explicación del tutor

**Por qué el modelo necesita números, no texto**

Una red neuronal —incluido el transformer de Fase 4— es, en el fondo, una
cadena de multiplicaciones de matrices. No hay ninguna operación en ese
pipeline que sepa "leer" una letra; todo input debe ser numérico antes de
entrar. Por eso la tokenización no es un paso opcional de limpieza, es el
puente obligatorio entre "texto que un humano entiende" y "números que una
matriz puede multiplicar".

**Token y vocabulario**

Un token es la unidad mínima en la que se parte el texto para asignarle un
id entero. El vocabulario es la lista completa de piezas distintas que el
modelo puede llegar a ver.

**Las tres granularidades, y por qué BPE gana**

- Char-level: vocabulario chico, secuencias largas.
- Word-level: secuencias cortas, pero vocabulario que explota y no puede
  representar palabras nuevas (necesita `<UNK>`, con pérdida de
  información).
- BPE (subword): parte de caracteres y fusiona iterativamente los pares
  más frecuentes hasta formar un vocabulario de subpalabras. Nunca hay una
  palabra "imposible de tokenizar" porque en el peor caso siempre se puede
  caer hasta el carácter individual.

Ejemplo de simulación manual (mini-corpus de juguete con "gana Colombia" x2
y "gana México"): las palabras frecuentes ("gana", con 3 apariciones) se
comprimen rápido; las raras ("México", 1 aparición) casi no cambian. El
orden de las fusiones queda "grabado" como una lista ordenada de reglas
(merges) que se reaplican, en el mismo orden, a texto nuevo.

**Dato curioso:** BPE nació en 1994 como algoritmo de compresión de datos
(Philip Gage), sin relación con IA. El paper de Sennrich et al. (2015) lo
reaprovechó para tokenización de lenguaje, y desde entonces es estándar de
facto.

**Comparación con la industria:** GPT-2 en adelante usa *byte-level* BPE
(sobre bytes crudos 0-255, no caracteres Unicode) para garantizar
cobertura universal de cualquier alfabeto o emoji, con ~50,000 merges
entrenados sobre corpus de cientos de gigabytes. En este proyecto se
usaron 300 merges sobre caracteres Unicode, suficiente para un corpus de
6582 preguntas en un solo idioma con encoding ya verificado limpio en
Fase 0.

**Merges: proceso vs. resultado**

Una merge es una regla de reescritura ("cuando veas X seguido de Y,
fusiónalos en XY") descubierta una vez durante el entrenamiento y
reaplicada, en el mismo orden exacto, a cualquier texto nuevo. Las
fusiones son en cascada: una merge posterior opera sobre el resultado de
una anterior, nunca sobre el texto original (ej. `(i,c)→ic`, luego
`(ic,a</w>)→ica</w>`). Por eso el JSON de salida guarda `merges`
(el proceso) y `vocabulario_final` (el resultado) por separado.

**Resultado real sobre el corpus completo (6582 preguntas):**

| Métrica | Char-level | BPE |
|---|---|---|
| Tamaño vocabulario | 78 | 252 |
| Longitud promedio (tokens/pregunta) | 48.97 | 11.07 |
| Total tokens en corpus | 322,317 | 72,873 |

Factor de compresión BPE vs. char-level: **4.42x**. Importa concretamente
porque el costo de self-attention en el transformer (Fase 4) crece
cuadráticamente con la longitud de secuencia — con contexto corto
(64-128 tokens) y hardware limitado, esta compresión es relevante, no un
detalle cosmético.

**Caso real observado:** "México" quedó casi sin fusionar en el corpus
pese a mencionarse seguido — sus pares de caracteres específicos (`x-i`,
`é-x`) compitieron cada ronda contra terminaciones mucho más comunes en
español (`-ado`, `ar</w>`) en todo el corpus, y perdieron las 300 veces.
Ilustra que BPE es pura estadística de frecuencia, sin ninguna noción de
qué palabras son "importantes" semánticamente.

## Resumen del usuario

- Un modelo matematico procesa números, es logico, pero es importante la anotación, dado que el corpus deben ser números
- Multiplicaciones de matrices
- Un token es la unidad mínima en la que se parte un texto para poder mapearla a un número (un "id"). Tokenizar es el proceso de partir "¿Quién gana Partido 1?" en piezas y asignarle un id entero a cada pieza única, formando un vocabulario (la lista de todas las piezas posibles que el modelo puede llegar a ver).
- Granularidades para partir texto: por caracter, por palabra y por sub palabra que es BPE (Byte pair encoding) donde va fusionando pares que mas se repiten, itera hasta formar un vocabulario de subpalabras frecuente y esas subpalabras quedan como un token, ejemplo:
	Al inicio: C-o-l-o-m-b-i-a (8 tokens, uno por carácter)
	Tras varias fusiones, algo como: Col-om-bia o incluso Colombia completo como un solo token, porque el algoritmo "aprendió" que esa secuencia se repite tanto que vale la pena darle su propio id.
- Primero creamos un char level, donde extrajimos cada caracterer, le dimos un id (número) y luego cada parte del corpus quedo representado por esos numeros (ids)
- Luego creamos el BPE donde vamos a fusionar, con lo de los pares y empezar a crear token de acuerdo a lo que se vaya repitiendo
- Los dos son procesos diferentes, solo son de comparacion, en este caso nos sirve mas BPE

## Validación del tutor

**Lo que entendiste bien:**
- La necesidad de números para el modelo, y la razón técnica correcta
  ("multiplicaciones de matrices") — vas al fondo, no te quedaste en
  "porque sí".
- La definición de token y vocabulario está exacta.
- El ejemplo de `Colombia` (char por char → subpalabras → palabra
  completa) demuestra que entendiste el efecto de la fusión progresiva.
- La distinción de que char-level y BPE son "procesos diferentes, solo de
  comparación" y que BPE es superior para este caso: correcto y es justo
  la conclusión que buscaba la fase.

**Lo que falta o es impreciso:**
- Describes qué hace la fusión de BPE pero no el cómo se decide y guarda
  ese proceso: falta la idea de que cada fusión es una regla ordenada que
  luego se reaplica a texto nuevo (no solo al corpus de entrenamiento).
  Sin esto, en Fase 4 se podría pensar que "tokenizar" es solo mirar un
  diccionario fijo, cuando en realidad para BPE es correr la cadena
  completa de reglas cada vez.
- No se menciona por qué BPE es mejor que char-level en términos
  cuantificables para el proyecto: la compresión de secuencia (4.42x)
  importa porque el costo de self-attention en el transformer de Fase 4
  crece cuadráticamente con la longitud de la secuencia.
- No se menciona el caso "México": BPE es pura estadística de frecuencia
  sin ninguna noción semántica — una palabra puede aparecer mucho y aun
  así quedar poco fusionada si sus pares de caracteres pierden la
  competencia de frecuencia contra terminaciones más comunes del idioma.

**Para profundizar (conectando con Fase 2):**
- En Fase 2 (embeddings), el id numérico de un token por sí solo no tiene
  significado semántico — dos ids consecutivos no son "parecidos" en
  ningún sentido. Ese es el problema que resuelven los embeddings: mapear
  cada token a un vector donde la cercanía sí tiene significado.
