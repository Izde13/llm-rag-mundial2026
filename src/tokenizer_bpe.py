"""
Fase 1 — Tokenizador BPE (Byte Pair Encoding) simple.

Segundo nivel de tokenizacion, construido sobre la base del char-level
(tokenizer_char.py). En vez de que cada caracter sea un token, BPE fusiona
iterativamente los pares de simbolos mas frecuentes del corpus, formando
subpalabras. El algoritmo:

1. Parte cada palabra en caracteres + marcador de fin de palabra `</w>`.
2. Cuenta la frecuencia de cada par de simbolos adyacentes.
3. Fusiona el par mas frecuente en un simbolo nuevo.
4. Repite `num_merges` veces, guardando el orden de las fusiones
   (ese orden es necesario para tokenizar texto nuevo mas adelante).

Se entrena sobre el campo `pregunta` del corpus, igual que el char-level.
"""

import json
import sys
from collections import Counter
from pathlib import Path

CORPUS_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
OUTPUT_JSON = Path(__file__).parent.parent / "data" / "processed" / "tokenizer_bpe_stats.json"
NUM_MERGES = 300
FIN_PALABRA = "</w>"


def preparar_palabras(textos: list[str]) -> dict[tuple[str, ...], int]:
    """Parte cada texto en palabras, y cada palabra en simbolos (caracteres + </w>).

    Devuelve un diccionario {palabra_partida: frecuencia}, contando las
    palabras unicas en vez de recorrer todo el corpus en cada iteracion.
    """
    frecuencia_palabras: Counter[str] = Counter()
    for texto in textos:
        for palabra in texto.split():
            frecuencia_palabras[palabra] += 1

    vocab_palabras: dict[tuple[str, ...], int] = {}
    for palabra, frecuencia in frecuencia_palabras.items():
        simbolos = tuple(list(palabra) + [FIN_PALABRA])
        vocab_palabras[simbolos] = frecuencia
    return vocab_palabras


def contar_pares(vocab_palabras: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    pares: Counter[tuple[str, str]] = Counter()
    for simbolos, frecuencia in vocab_palabras.items():
        for i in range(len(simbolos) - 1):
            par = (simbolos[i], simbolos[i + 1])
            pares[par] += frecuencia
    return pares


def fusionar_par(
    par: tuple[str, str], vocab_palabras: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    simbolo_fusionado = "".join(par)
    nuevo_vocab: dict[tuple[str, ...], int] = {}

    for simbolos, frecuencia in vocab_palabras.items():
        nuevos_simbolos = []
        i = 0
        while i < len(simbolos):
            if i < len(simbolos) - 1 and (simbolos[i], simbolos[i + 1]) == par:
                nuevos_simbolos.append(simbolo_fusionado)
                i += 2
            else:
                nuevos_simbolos.append(simbolos[i])
                i += 1
        nuevo_vocab[tuple(nuevos_simbolos)] = frecuencia

    return nuevo_vocab


def entrenar_bpe(
    textos: list[str], num_merges: int
) -> tuple[dict[tuple[str, ...], int], list[tuple[str, str]]]:
    vocab_palabras = preparar_palabras(textos)
    merges: list[tuple[str, str]] = []

    for _ in range(num_merges):
        pares = contar_pares(vocab_palabras)
        if not pares:
            break
        par_mas_frecuente = max(pares, key=pares.get)
        vocab_palabras = fusionar_par(par_mas_frecuente, vocab_palabras)
        merges.append(par_mas_frecuente)

    return vocab_palabras, merges


def tokenizar_con_merges(texto: str, merges: list[tuple[str, str]]) -> list[str]:
    """Aplica las reglas de fusion aprendidas, en orden, a un texto nuevo."""
    tokens_por_palabra = []
    for palabra in texto.split():
        simbolos = list(palabra) + [FIN_PALABRA]
        for par in merges:
            simbolo_fusionado = "".join(par)
            i = 0
            nuevos_simbolos = []
            while i < len(simbolos):
                if i < len(simbolos) - 1 and (simbolos[i], simbolos[i + 1]) == par:
                    nuevos_simbolos.append(simbolo_fusionado)
                    i += 2
                else:
                    nuevos_simbolos.append(simbolos[i])
                    i += 1
            simbolos = nuevos_simbolos
        tokens_por_palabra.extend(simbolos)
    return tokens_por_palabra


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)

    preguntas = [registro["pregunta"] for registro in corpus]

    vocab_palabras_final, merges = entrenar_bpe(preguntas, NUM_MERGES)

    # Vocabulario final: todos los simbolos distintos que quedaron tras las fusiones.
    simbolos_finales = set()
    for simbolos in vocab_palabras_final:
        simbolos_finales.update(simbolos)

    longitudes_bpe = [len(tokenizar_con_merges(p, merges)) for p in preguntas[:200]]
    longitudes_char = [len(p) for p in preguntas[:200]]

    estadisticas = {
        "num_merges_aplicados": len(merges),
        "tamano_vocabulario_bpe": len(simbolos_finales),
        "longitud_promedio_bpe_tokens": round(sum(longitudes_bpe) / len(longitudes_bpe), 2),
        "longitud_promedio_char_tokens": round(sum(longitudes_char) / len(longitudes_char), 2),
        "muestra_calculada_sobre": "primeras 200 preguntas",
    }

    print("Estadisticas del tokenizador BPE:")
    for clave, valor in estadisticas.items():
        print(f"  {clave}: {valor}")

    ejemplos = []
    for pregunta in preguntas[:3]:
        tokens = tokenizar_con_merges(pregunta, merges)
        ejemplos.append({"texto": pregunta, "tokens": tokens, "num_tokens": len(tokens)})

    print("\nEjemplo de tokenizacion BPE:")
    print(f"  texto:  {ejemplos[0]['texto']}")
    print(f"  tokens: {ejemplos[0]['tokens']}")

    salida = {
        "estadisticas": estadisticas,
        "merges": [list(par) for par in merges],
        "vocabulario_final": sorted(simbolos_finales),
        "ejemplos": ejemplos,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print(f"\nSalida guardada en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
