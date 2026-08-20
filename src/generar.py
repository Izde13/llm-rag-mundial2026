"""
Fase 4 — Generacion de texto (autoregresiva) con el modelo entrenado.

Repite, token a token: predecir el siguiente token -> agregarlo al texto
-> volver a predecir usando el texto ya extendido como nueva entrada.
Es la misma tarea que el modelo aprendio en el entrenamiento (train.py),
solo que aqui no hay "objetivo" real -- el propio modelo decide que sigue.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from bpe_vocab import cargar_merges, codificar_ids, construir_vocab_ids, decodificar_ids
from transformer_model import TransformerLM

MODELO_PATH = Path(__file__).parent.parent / "data" / "processed" / "modelo_fase4.pt"


def cargar_modelo(ruta: Path = MODELO_PATH) -> tuple[TransformerLM, dict[str, int]]:
    checkpoint = torch.load(ruta, weights_only=False)
    hp = checkpoint["hiperparametros"]

    modelo = TransformerLM(
        vocab_size=hp["vocab_size"],
        d_model=hp["d_model"],
        num_heads=hp["num_heads"],
        d_ff=hp["d_ff"],
        num_layers=hp["num_layers"],
        max_seq_len=hp["max_seq_len"],
    )
    modelo.load_state_dict(checkpoint["modelo_state_dict"])
    modelo.eval()

    return modelo, checkpoint["simbolo_a_id"]


def generar(
    modelo: TransformerLM,
    texto_inicial: str,
    merges: list[tuple[str, str]],
    simbolo_a_id: dict[str, int],
    id_a_simbolo: dict[int, str],
    max_seq_len: int,
    max_nuevos_tokens: int = 20,
    temperatura: float = 1.0,
    greedy: bool = False,
) -> str:
    """Genera texto token a token a partir de `texto_inicial`.

    greedy=True  -> siempre el token de mayor probabilidad (determinista).
    greedy=False -> sorteo ponderado por probabilidad, afilado/aplanado
                     por `temperatura` (mas alta = mas variedad/riesgo).
    """
    ids = codificar_ids(texto_inicial, merges, simbolo_a_id)

    with torch.no_grad():
        for _ in range(max_nuevos_tokens):
            entrada = torch.tensor([ids[-max_seq_len:]])  # recorta al contexto maximo del modelo
            logits = modelo(entrada)
            ultimo_logit = logits[0, -1]  # solo interesa la prediccion de la ultima posicion

            if greedy:
                siguiente_id = torch.argmax(ultimo_logit).item()
            else:
                probs = torch.softmax(ultimo_logit / temperatura, dim=-1)
                siguiente_id = torch.multinomial(probs, num_samples=1).item()

            ids.append(siguiente_id)

    return decodificar_ids(ids, id_a_simbolo)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    modelo, simbolo_a_id = cargar_modelo()
    id_a_simbolo = {v: k for k, v in simbolo_a_id.items()}
    merges = cargar_merges()
    max_seq_len = modelo.embedding.position_embedding.num_embeddings

    inicios = ["Marcador Partido", "¿Quién", "Roja"]

    for inicio in inicios:
        print(f"--- Inicio: {inicio!r} ---")

        texto_greedy = generar(
            modelo, inicio, merges, simbolo_a_id, id_a_simbolo, max_seq_len,
            max_nuevos_tokens=20, greedy=True,
        )
        print(f"  greedy:              {texto_greedy}")

        for temp in (0.5, 1.0):
            texto_sample = generar(
                modelo, inicio, merges, simbolo_a_id, id_a_simbolo, max_seq_len,
                max_nuevos_tokens=20, greedy=False, temperatura=temp,
            )
            print(f"  sampling (temp={temp}): {texto_sample}")
        print()


if __name__ == "__main__":
    main()
