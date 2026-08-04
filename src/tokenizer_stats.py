"""
Fase 1 — Comparacion final: tokenizador char-level vs. BPE.

Entregable de la fase (ver PROYECTO_BASE.md): corre ambos tokenizadores
sobre el corpus completo (data/processed/corpus.json) y compara tamano de
vocabulario y longitud de secuencia resultante, para cuantificar el
beneficio de BPE frente a char-level.

Reutiliza las funciones ya implementadas en tokenizer_char.py y
tokenizer_bpe.py en vez de duplicar la logica de tokenizacion.
"""

import json
import sys
from pathlib import Path

from tokenizer_bpe import NUM_MERGES, entrenar_bpe, tokenizar_con_merges
from tokenizer_char import codificar, construir_vocabulario

CORPUS_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
OUTPUT_JSON = Path(__file__).parent.parent / "data" / "processed" / "tokenizer_comparacion.json"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)

    preguntas = [registro["pregunta"] for registro in corpus]

    # --- Char-level, sobre el corpus completo ---
    char_a_id, _ = construir_vocabulario(preguntas)
    longitudes_char = [len(codificar(p, char_a_id)) for p in preguntas]

    # --- BPE, sobre el corpus completo ---
    vocab_palabras_final, merges = entrenar_bpe(preguntas, NUM_MERGES)
    simbolos_bpe = set()
    for simbolos in vocab_palabras_final:
        simbolos_bpe.update(simbolos)
    longitudes_bpe = [len(tokenizar_con_merges(p, merges)) for p in preguntas]

    total_char = sum(longitudes_char)
    total_bpe = sum(longitudes_bpe)

    comparacion = {
        "total_preguntas": len(preguntas),
        "char_level": {
            "tamano_vocabulario": len(char_a_id),
            "longitud_promedio": round(total_char / len(preguntas), 2),
            "longitud_maxima": max(longitudes_char),
            "total_tokens_corpus": total_char,
        },
        "bpe": {
            "num_merges": len(merges),
            "tamano_vocabulario": len(simbolos_bpe),
            "longitud_promedio": round(total_bpe / len(preguntas), 2),
            "longitud_maxima": max(longitudes_bpe),
            "total_tokens_corpus": total_bpe,
        },
        "factor_compresion_bpe_vs_char": round(total_char / total_bpe, 2),
    }

    print("Comparacion char-level vs. BPE (corpus completo, 6582 preguntas):\n")
    print(f"{'Metrica':<28}{'Char-level':>15}{'BPE':>15}")
    print("-" * 58)
    print(
        f"{'Tamano vocabulario':<28}"
        f"{comparacion['char_level']['tamano_vocabulario']:>15}"
        f"{comparacion['bpe']['tamano_vocabulario']:>15}"
    )
    print(
        f"{'Longitud promedio':<28}"
        f"{comparacion['char_level']['longitud_promedio']:>15}"
        f"{comparacion['bpe']['longitud_promedio']:>15}"
    )
    print(
        f"{'Longitud maxima':<28}"
        f"{comparacion['char_level']['longitud_maxima']:>15}"
        f"{comparacion['bpe']['longitud_maxima']:>15}"
    )
    print(
        f"{'Total tokens en corpus':<28}"
        f"{comparacion['char_level']['total_tokens_corpus']:>15}"
        f"{comparacion['bpe']['total_tokens_corpus']:>15}"
    )
    print(f"\nFactor de compresion BPE vs. char-level: {comparacion['factor_compresion_bpe_vs_char']}x")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparacion, f, ensure_ascii=False, indent=2)

    print(f"\nSalida guardada en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
