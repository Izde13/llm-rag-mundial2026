"""
Fase 4 — Loop de entrenamiento del LLM propio.

Tarea de entrenamiento: dado un fragmento de la oracion, predecir la
siguiente palabra (autoregresivo, igual que GPT). El objetivo es la
misma oracion desplazada una posicion:

    oracion:   id0  id1  id2  id3
    entrada:   id0  id1  id2        (x)
    objetivo:       id1  id2  id3   (y)

Causal masking (ya presente en MultiHeadAttention con causal=True)
garantiza que la posicion i solo pudo usar id0..idi para predecir
id(i+1) -- nunca "hizo trampa" mirando el objetivo.
"""

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from bpe_vocab import cargar_merges, codificar_ids, construir_vocab_ids, TOKEN_PAD
from transformer_model import TransformerLM

CORPUS_JSON = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
MODELO_PATH = Path(__file__).parent.parent / "data" / "processed" / "modelo_fase4.pt"


def preparar_lote(
    textos: list[str],
    merges: list[tuple[str, str]],
    simbolo_a_id: dict[str, int],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Codifica una lista de textos y arma (entrada, objetivo) desplazados.

    Cada texto se codifica, se trunca/rellena a `max_len + 1` ids (un id
    extra para poder desplazar una posicion), y se parte en:
    - entrada  = ids[:-1]  (lo que el modelo ve)
    - objetivo = ids[1:]   (lo que deberia predecir en cada posicion)
    """
    id_pad = simbolo_a_id[TOKEN_PAD]
    entradas, objetivos = [], []

    for texto in textos:
        ids = codificar_ids(texto, merges, simbolo_a_id)
        ids = ids[: max_len + 1]
        ids = ids + [id_pad] * (max_len + 1 - len(ids))

        entradas.append(ids[:-1])
        objetivos.append(ids[1:])

    return torch.tensor(entradas), torch.tensor(objetivos)


def un_paso_de_entrenamiento(
    modelo: TransformerLM,
    optimizador: torch.optim.Optimizer,
    funcion_perdida: nn.Module,
    entrada: torch.Tensor,
    objetivo: torch.Tensor,
) -> float:
    """Un solo paso: predecir -> medir error -> ajustar pesos. Devuelve el loss."""
    logits = modelo(entrada)  # [batch, seq_len, vocab_size]

    vocab_size = logits.shape[-1]
    loss = funcion_perdida(logits.reshape(-1, vocab_size), objetivo.reshape(-1))

    optimizador.zero_grad()
    loss.backward()
    optimizador.step()

    return loss.item()


@torch.no_grad()
def evaluar(
    modelo: TransformerLM,
    funcion_perdida: nn.Module,
    textos: list[str],
    merges: list[tuple[str, str]],
    simbolo_a_id: dict[str, int],
    max_len: int,
    batch_size: int,
) -> float:
    """Mismo calculo de loss que un_paso_de_entrenamiento, pero sin
    backward() ni optimizador.step() -- solo mide, no ajusta pesos.
    Se usa sobre el set de validacion, que el modelo nunca ve para
    entrenar, para detectar si esta memorizando en vez de generalizar.
    """
    modelo.eval()
    vocab_size = len(simbolo_a_id)
    total_loss = 0.0
    num_lotes = 0

    for i in range(0, len(textos), batch_size):
        lote_textos = textos[i : i + batch_size]
        entrada, objetivo = preparar_lote(lote_textos, merges, simbolo_a_id, max_len)
        logits = modelo(entrada)
        loss = funcion_perdida(logits.reshape(-1, vocab_size), objetivo.reshape(-1))
        total_loss += loss.item()
        num_lotes += 1

    modelo.train()
    return total_loss / num_lotes


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(0)
    torch.manual_seed(0)

    with open(CORPUS_JSON, encoding="utf-8") as f:
        corpus = json.load(f)
    preguntas = [registro["pregunta"] for registro in corpus]

    merges = cargar_merges()
    simbolo_a_id, _ = construir_vocab_ids(merges, preguntas)
    vocab_size = len(simbolo_a_id)

    # split train/val 90/10, barajado para no separar por orden del corpus
    preguntas_barajadas = preguntas.copy()
    random.shuffle(preguntas_barajadas)
    corte = int(len(preguntas_barajadas) * 0.9)
    textos_train = preguntas_barajadas[:corte]
    textos_val = preguntas_barajadas[corte:]

    max_len = 64
    batch_size = 32
    num_epocas = 5
    lr = 3e-4
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 4

    modelo = TransformerLM(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=max_len)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    funcion_perdida = nn.CrossEntropyLoss(ignore_index=simbolo_a_id[TOKEN_PAD])

    total_params = sum(p.numel() for p in modelo.parameters())
    print("=== Entrenamiento TransformerLM (Fase 4) ===")
    print(f"Corpus: {len(preguntas)} preguntas -> {len(textos_train)} train / {len(textos_val)} val")
    print(f"vocab_size={vocab_size}, d_model={d_model}, num_layers={num_layers}, parametros={total_params:,}")
    print(f"max_len={max_len}, batch_size={batch_size}, epocas={num_epocas}, lr={lr}")
    print()

    for epoca in range(1, num_epocas + 1):
        random.shuffle(textos_train)
        loss_acumulado = 0.0
        num_lotes = 0

        for i in range(0, len(textos_train), batch_size):
            lote_textos = textos_train[i : i + batch_size]
            entrada, objetivo = preparar_lote(lote_textos, merges, simbolo_a_id, max_len)
            loss = un_paso_de_entrenamiento(modelo, optimizador, funcion_perdida, entrada, objetivo)
            loss_acumulado += loss
            num_lotes += 1

        loss_train = loss_acumulado / num_lotes
        loss_val = evaluar(modelo, funcion_perdida, textos_val, merges, simbolo_a_id, max_len, batch_size)

        print(f"epoca {epoca}/{num_epocas}  loss_train={loss_train:.4f}  loss_val={loss_val:.4f}")

    MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modelo_state_dict": modelo.state_dict(),
            "simbolo_a_id": simbolo_a_id,
            "hiperparametros": {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "num_layers": num_layers,
                "max_seq_len": max_len,
            },
        },
        MODELO_PATH,
    )
    print(f"\nModelo guardado en: {MODELO_PATH}")


if __name__ == "__main__":
    main()
