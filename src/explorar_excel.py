"""
Fase 0 — Exploración y diagnóstico del Excel fuente.

Antes de escribir el pipeline final (data_prep.py), se investigó el
archivo data/raw/Polla Mundial 2026.xlsx: qué hojas tiene, qué columnas
trae cada una, y si el texto (acentos, ñ, ¿) estaba corrupto.

Este módulo junta esas funciones de exploración para que el proceso sea
reproducible y no viva solo en el historial de una conversación. Se puede
correr como script (`python src/explorar_excel.py`) para repetir todo el
diagnóstico, o importar sus funciones sueltas desde una notebook.
"""

import sys
from pathlib import Path

import pandas as pd

RAW_XLSX = Path(__file__).parent.parent / "data" / "raw" / "Polla Mundial 2026.xlsx"


def listar_hojas(ruta: Path = RAW_XLSX) -> list[str]:
    """Devuelve los nombres de todas las hojas (pestañas) del libro."""
    xls = pd.ExcelFile(ruta)
    return xls.sheet_names


def describir_hojas(ruta: Path = RAW_XLSX) -> dict[str, dict]:
    """
    Para cada hoja, devuelve su forma (filas x columnas) y sus nombres
    de columna. Sirve para entender qué datos hay antes de diseñar
    cualquier cruce (merge) entre hojas.
    """
    xls = pd.ExcelFile(ruta)
    resumen = {}
    for nombre in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=nombre)
        resumen[nombre] = {
            "filas": df.shape[0],
            "columnas": df.shape[1],
            "nombres_columnas": list(df.columns),
        }
    return resumen


def muestra_de_hoja(nombre_hoja: str, n: int = 2, ruta: Path = RAW_XLSX) -> pd.DataFrame:
    """Devuelve las primeras `n` filas de una hoja específica, para inspección rápida."""
    return pd.read_excel(ruta, sheet_name=nombre_hoja).head(n)


def caracteres_no_ascii(ruta: Path = RAW_XLSX) -> set[str]:
    """
    Recorre todas las columnas de texto de todas las hojas y devuelve el
    conjunto de caracteres únicos fuera del rango ASCII básico (código > 127).

    Se usó esto para acotar el universo de caracteres especiales (acentos,
    ñ, símbolos) antes de decidir si el texto tenía un problema real de
    encoding o no.
    """
    xls = pd.ExcelFile(ruta)
    encontrados: set[str] = set()
    for nombre in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=nombre)
        for columna in df.select_dtypes(include=["object", "str"]).columns:
            for valor in df[columna].dropna():
                if isinstance(valor, str):
                    encontrados.update(ch for ch in valor if ord(ch) > 127)
    return encontrados


def verificar_encoding_correcto(ruta: Path = RAW_XLSX) -> bool:
    """
    Comprueba que el texto de la hoja PREGUNTAS esté en UTF-8 válido,
    escribiéndolo a un archivo temporal en UTF-8 y comparando que el
    resultado no contenga el carácter de reemplazo Unicode (U+FFFD, '�').

    Nota de diagnóstico (ver docs/fase-0-setup-y-datos.md, sección 4):
    el texto se veía corrupto en la terminal de Windows (que usa cp1252),
    pero el string en memoria de pandas siempre estuvo correcto. Esta
    función automatiza la prueba que lo confirmó, en vez de dejarlo como
    una inspección manual de una sola vez.
    """
    df = pd.read_excel(ruta, sheet_name="PREGUNTAS")
    textos = df["enunciado"].dropna().astype(str)
    tiene_caracter_de_reemplazo = textos.str.contains("�").any()
    return not tiene_caracter_de_reemplazo


def main() -> None:
    # La consola de Windows suele usar cp1252, que no sabe mostrar todos
    # los caracteres Unicode (ej. '⁷'). Se fuerza la salida estándar a
    # UTF-8 para que el script no falle al imprimir texto con acentos o
    # símbolos especiales — el mismo problema que se diagnosticó en la
    # sección 4 de docs/fase-0-setup-y-datos.md.
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== Hojas del libro ===")
    for hoja in listar_hojas():
        print("-", hoja)

    print("\n=== Filas/columnas por hoja ===")
    for nombre, info in describir_hojas().items():
        print(f"{nombre}: {info['filas']} filas x {info['columnas']} columnas")
        print(f"  columnas: {info['nombres_columnas']}")

    print("\n=== Caracteres no-ASCII encontrados en todo el libro ===")
    especiales = sorted(caracteres_no_ascii(), key=ord)
    print(especiales)

    print("\n=== Verificación de encoding (hoja PREGUNTAS) ===")
    if verificar_encoding_correcto():
        print("OK: el texto está en UTF-8 válido, sin caracteres de reemplazo.")
    else:
        print("ALERTA: se encontró el carácter de reemplazo Unicode (texto corrupto).")


if __name__ == "__main__":
    main()
