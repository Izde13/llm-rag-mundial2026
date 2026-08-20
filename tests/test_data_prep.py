"""
Pruebas de la Fase 0.

Verifican que el pipeline de datos (src/data_prep.py) produzca un corpus
sin huecos: cada registro debe tener jornada, participante y pregunta
resueltos, es decir, que ninguno de los merges (cruces) contra PREGUNTAS,
PARTICIPANTES o JORNADAS haya fallado en encontrar coincidencia.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_prep import cargar_hojas, construir_corpus  # noqa: E402


def test_corpus_no_tiene_campos_vacios():
    hojas = cargar_hojas()
    corpus = construir_corpus(hojas)

    columnas_obligatorias = ["jornada", "participante", "pregunta"]
    for columna in columnas_obligatorias:
        vacios = corpus[columna].isna().sum()
        assert vacios == 0, f"{vacios} registros sin '{columna}' (merge sin match)"


def test_corpus_tiene_las_columnas_esperadas():
    hojas = cargar_hojas()
    corpus = construir_corpus(hojas)

    columnas_esperadas = {
        "jornada_id",
        "jornada",
        "participante",
        "pregunta_id",
        "pregunta",
        "respuesta_participante",
        "resultado_oficial",
        "puntos",
    }
    assert columnas_esperadas.issubset(set(corpus.columns))


def test_corpus_tiene_registros():
    hojas = cargar_hojas()
    corpus = construir_corpus(hojas)

    assert len(corpus) > 0
