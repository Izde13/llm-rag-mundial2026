"""
Fase 0 — Preparación de datos.

Lee el Excel exportado del Google Sheets de la Polla Mundial 2026
(data/raw/Polla Mundial 2026.xlsx) y construye un corpus único en JSON
(data/processed/corpus.json), pensado como material de entrenamiento
para el LLM propio y como base documental del RAG.

Estrategia: la hoja PUNTUACIONES ya tiene, por fila, la respuesta de un
participante a una pregunta junto con el resultado oficial y los puntos
obtenidos. Le faltan textos legibles (el enunciado de la pregunta, el
nombre del participante, el nombre de la jornada), que viven en otras
hojas. Se cruzan (merge) por sus respectivos ids para completar esos
textos.
"""

import json
from pathlib import Path

import pandas as pd

RAW_XLSX = Path(__file__).parent.parent / "data" / "raw" / "Polla Mundial 2026.xlsx"
OUTPUT_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"


def cargar_hojas() -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(RAW_XLSX)
    hojas = ["PUNTUACIONES", "PREGUNTAS", "PARTICIPANTES", "JORNADAS"]
    return {nombre: pd.read_excel(xls, sheet_name=nombre) for nombre in hojas}


def construir_corpus(hojas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    puntuaciones = hojas["PUNTUACIONES"]
    preguntas = hojas["PREGUNTAS"][["pregunta_id", "enunciado"]]
    participantes = hojas["PARTICIPANTES"][["email", "nombre"]]
    jornadas = hojas["JORNADAS"][["jornada_id", "nombre"]]

    df = puntuaciones.merge(preguntas, on="pregunta_id", how="left")

    df = df.merge(
        participantes,
        left_on="participante_id",
        right_on="email",
        how="left",
        suffixes=("", "_participante"),
    )
    df = df.rename(columns={"nombre": "nombre_participante"})

    df = df.merge(
        jornadas,
        on="jornada_id",
        how="left",
        suffixes=("", "_jornada"),
    )
    df = df.rename(columns={"nombre": "nombre_jornada"})

    columnas_finales = {
        "jornada_id": "jornada_id",
        "nombre_jornada": "jornada",
        "nombre_participante": "participante",
        "pregunta_id": "pregunta_id",
        "enunciado": "pregunta",
        "respuesta_participante": "respuesta_participante",
        "resultado_oficial": "resultado_oficial",
        "puntos": "puntos",
    }
    return df[list(columnas_finales.keys())].rename(columns=columnas_finales)


def main() -> None:
    hojas = cargar_hojas()
    corpus = construir_corpus(hojas)

    registros = corpus.to_dict(orient="records")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2, default=str)

    print(f"Corpus generado: {OUTPUT_JSON} ({len(registros)} registros)")


if __name__ == "__main__":
    main()
