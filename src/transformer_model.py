"""
Fase 4 — Arquitectura Transformer desde cero, en PyTorch puro (sin la
libreria `transformers` de HuggingFace).

Construida pieza por pieza, en el mismo orden en que aparecen en un bloque
transformer real:

1. EmbeddingConPosicion  — ids -> vectores (token + posicion aprendidos)
2. SelfAttention         — cada token mezcla informacion de los demas
   (con causal masking opcional)
3. MultiHeadAttention    — varias "vistas" de atencion en paralelo
4. (siguiente) feed-forward, layer norm, TransformerBlock, cabeza de
   salida.
"""

import math

import torch
import torch.nn as nn


class EmbeddingConPosicion(nn.Module):
    """Convierte ids de token en vectores que ademas codifican su posicion.

    Dos tablas de pesos aprendibles (nn.Embedding es un lookup: la fila i
    es el vector del indice i), sumadas elemento a elemento:

    - token_embedding[id]   -> "que token es" (que significa, tras entrenar)
    - position_embedding[i] -> "en que posicion de la secuencia esta"

    Sumar ambas permite que el mismo token tenga un vector distinto segun
    donde aparezca en la oracion -- necesario porque self-attention (paso
    2) por si solo no distingue orden.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [batch, seq_len] -> salida: [batch, seq_len, d_model]
        _, seq_len = ids.shape
        posiciones = torch.arange(seq_len, device=ids.device)  # [seq_len]
        return self.token_embedding(ids) + self.position_embedding(posiciones)


class SelfAttention(nn.Module):
    """Cada token mezcla informacion de los demas tokens de la secuencia.

    Tres transformaciones lineales (matrices aprendibles, aleatorias hasta
    entrenar) sobre el mismo vector de entrada:

    - Q (query) : la "pregunta" de cada token -- que busca.
    - K (key)   : el "cartel" de cada token -- que ofrece.
    - V (value) : el "contenido" real que aportaria si le prestan atencion.

    Attention(Q,K,V) = softmax( Q @ K^T / sqrt(d_model) ) @ V

    Q@K^T compara cada pregunta contra cada cartel (que tanto calzan);
    dividir por sqrt(d_model) es solo estabilidad numerica (evita que el
    softmax se sature con d_model grande); softmax convierte esos puntajes,
    fila por fila, en un reparto de atencion que suma 1.0; y ese reparto
    pondera el promedio de los V -- Q y K solo deciden el reparto, nunca
    aparecen en el resultado final directamente.

    causal=True activa el causal masking: a cada posicion se le prohibe
    mirar posiciones futuras (mayores a la propia), tapando esos scores
    con -inf antes del softmax -- softmax les asigna 0% de atencion, sin
    tocar la formula. Necesario para entrenar un modelo generativo: sin
    esto, predecir la palabra en la posicion i podria "hacer trampa"
    mirando palabras que en la generacion real todavia no existirian.
    """

    def __init__(self, d_model: int, causal: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.shape[1]
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)  # [batch, seq_len, seq_len]

        if self.causal:
            mascara_futuro = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(mascara_futuro, float("-inf"))

        pesos_atencion = torch.softmax(scores, dim=-1)
        return pesos_atencion @ V  # [batch, seq_len, d_model]


class MultiHeadAttention(nn.Module):
    """Varias cabezas de SelfAttention en paralelo, cada una con su propia
    porcion de dimensiones (d_k = d_model // num_heads), concatenadas al
    final y mezcladas con una capa lineal de salida.

    Equivalente matematicamente a num_heads instancias independientes de
    SelfAttention, pero calculado con una sola matriz grande por Q/K/V y
    "cortada" en pedazos por cabeza -- mas eficiente en PyTorch que un
    bucle for de num_heads iteraciones, mismo resultado numerico.

    Cada cabeza, al tener sus propios pesos, tiene libertad de
    especializarse en un tipo de relacion distinto durante el
    entrenamiento (gramatical, tematica, posicional, etc.) -- una sola
    cabeza fuerza a todas esas senales a competir en un solo reparto.
    """

    def __init__(self, d_model: int, num_heads: int, causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)
        self.W_output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # [batch, heads, seq_len, seq_len]

        if self.causal:
            mascara_futuro = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(mascara_futuro, float("-inf"))

        pesos_atencion = torch.softmax(scores, dim=-1)
        salida = pesos_atencion @ V  # [batch, heads, seq_len, d_k]

        # concatenar cabezas: [batch, heads, seq_len, d_k] -> [batch, seq_len, d_model]
        salida = salida.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_output(salida)


class FeedForward(nn.Module):
    """Procesa cada posicion por separado (nunca mezcla entre posiciones,
    a diferencia de attention): estira el vector a un espacio mas grande
    (d_ff, tipicamente 4x d_model), aplica ReLU, y comprime de vuelta.

    Sin ReLU, dos capas lineales seguidas colapsan matematicamente en una
    sola transformacion lineal -- ReLU (max(0,x)) rompe esa linealidad y
    es lo que le da al modelo capacidad de aprender relaciones que no son
    un simple promedio ponderado (lo unico que attention, por si solo,
    puede producir).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Un bloque transformer completo: attention + residual + norm,
    seguido de feed-forward + residual + norm.

    Conexion residual (x + submodulo(x)): en vez de reemplazar el vector,
    se le suma la correccion -- si una capa no tiene nada util que
    aportar para cierta posicion, puede aprender a producir ~0 y dejar
    pasar la informacion original casi intacta. Necesario para poder
    apilar varias capas sin perder informacion capa tras capa.

    LayerNorm despues de cada suma: reescala los numeros (promedio ~0,
    dispersion estandar) sin cambiar su orden relativo -- mantenimiento
    de estabilidad numerica, no le agrega significado al vector.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, causal: bool = True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, causal=causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    """Columna vertebral del modelo: embedding + N bloques transformer
    apilados, mas la cabeza de salida (una capa lineal que convierte el
    vector final de cada posicion en un logit por simbolo del
    vocabulario -- "que tan probable es cada simbolo como siguiente
    token", antes de aplicarle softmax).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.embedding = EmbeddingConPosicion(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.cabeza_salida = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.cabeza_salida(x)  # [batch, seq_len, vocab_size] -- logits


def main() -> None:
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).parent))
    from bpe_vocab import cargar_merges, codificar_ids, construir_vocab_ids

    corpus_json = Path(__file__).parent.parent / "data" / "processed" / "corpus.json"
    import json

    with open(corpus_json, encoding="utf-8") as f:
        corpus = json.load(f)
    preguntas = [registro["pregunta"] for registro in corpus]

    merges = cargar_merges()
    simbolo_a_id, _ = construir_vocab_ids(merges, preguntas)

    lote_textos = preguntas[:3]
    ids_por_oracion = [codificar_ids(t, merges, simbolo_a_id) for t in lote_textos]
    largo_maximo = max(len(ids) for ids in ids_por_oracion)

    ids_rellenados = [
        ids + [0] * (largo_maximo - len(ids))  # 0 = <PAD>
        for ids in ids_por_oracion
    ]
    lote = torch.tensor(ids_rellenados)  # [batch, seq_len]

    d_model = 128
    embedding = EmbeddingConPosicion(
        vocab_size=len(simbolo_a_id), d_model=d_model, max_seq_len=largo_maximo
    )
    salida = embedding(lote)

    print("Textos del lote:")
    for t in lote_textos:
        print(f"  {t}")
    print(f"\nShape ids de entrada:  {tuple(lote.shape)}   (batch, seq_len)")
    print(f"Shape salida embedding: {tuple(salida.shape)}   (batch, seq_len, d_model)")
    print(f"\nParametros entrenables: {sum(p.numel() for p in embedding.parameters()):,}")

    print("\n--- SelfAttention sobre 'tarjeta amarilla' ---")
    torch.manual_seed(0)
    texto_chico = "tarjeta amarilla"
    ids_chico = codificar_ids(texto_chico, merges, simbolo_a_id)[:2]
    id_a_simbolo = {v: k for k, v in simbolo_a_id.items()}
    simbolos_chico = [id_a_simbolo[i] for i in ids_chico]

    d_model_chico = 4
    embedding_chico = EmbeddingConPosicion(len(simbolo_a_id), d_model_chico, max_seq_len=2)
    x = embedding_chico(torch.tensor([ids_chico]))

    atencion = SelfAttention(d_model_chico)
    salida_atencion = atencion(x)

    print(f"Texto: {texto_chico!r} -> simbolos: {simbolos_chico}")
    print(f"Shape entrada:  {tuple(x.shape)}")
    print(f"Shape salida:   {tuple(salida_atencion.shape)}")
    for i, s in enumerate(simbolos_chico):
        vector = [round(v, 2) for v in salida_atencion[0, i].tolist()]
        print(f"  nuevo vector de {s!r}: {vector}")


if __name__ == "__main__":
    main()
