"""
Fase 1 — Tokenizador char-level.

Primer nivel de tokenización (el más simple): cada caracter del texto es
un token. Sirve de base conceptual antes de implementar BPE (que fusiona
caracteres frecuentes en subpalabras).

Se tokeniza el campo `pregunta` del corpus (data/processed/corpus.json),
por ser el texto en lenguaje natural más representativo (a diferencia de
ids o nombres propios repetidos).
"""

import json
import sys
from pathlib import Path

CORPUS_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
OUTPUT_JSON = Path(__file__).parent.parent / "data" / "processed" / "tokenizer_char_stats.json"


def construir_vocabulario(textos: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    caracteres = sorted(set("".join(textos)))
    char_a_id = {char: idx for idx, char in enumerate(caracteres)}
    id_a_char = {idx: char for char, idx in char_a_id.items()}
    return char_a_id, id_a_char


def codificar(texto: str, char_a_id: dict[str, int]) -> list[int]:
    return [char_a_id[char] for char in texto]


def decodificar(ids: list[int], id_a_char: dict[int, str]) -> str:
    return "".join(id_a_char[idx] for idx in ids)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)

    preguntas = [registro["pregunta"] for registro in corpus]
    char_a_id, id_a_char = construir_vocabulario(preguntas)

    longitudes = [len(p) for p in preguntas]
    total_caracteres = sum(longitudes)

    # Verificacion de reversibilidad: codificar y decodificar debe devolver el texto original.
    ejemplos = []
    for pregunta in preguntas[:3]:
        ids = codificar(pregunta, char_a_id)
        recuperado = decodificar(ids, id_a_char)
        assert recuperado == pregunta, "La decodificacion no coincide con el texto original"
        ejemplos.append({"texto": pregunta, "ids": ids, "decodificado": recuperado})

    estadisticas = {
        "tamano_vocabulario": len(char_a_id),
        "total_preguntas": len(preguntas),
        "total_caracteres_en_corpus": total_caracteres,
        "longitud_promedio_pregunta": round(total_caracteres / len(preguntas), 2),
        "longitud_maxima_pregunta": max(longitudes),
        "longitud_minima_pregunta": min(longitudes),
    }

    print("Estadisticas del tokenizador char-level:")
    for clave, valor in estadisticas.items():
        print(f"  {clave}: {valor}")
    print("\nEjemplo de codificacion/decodificacion:")
    print(f"  texto:        {ejemplos[0]['texto']}")
    print(f"  ids:          {ejemplos[0]['ids']}")
    print(f"  decodificado: {ejemplos[0]['decodificado']}")

    salida = {
        "estadisticas": estadisticas,
        "vocabulario": char_a_id,
        "ejemplos": ejemplos,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print(f"\nSalida guardada en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
