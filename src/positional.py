# positional.py
#
# Codificaciones posicionales intercambiables para el modelo base (model_base_adaptado.py).
# Ampliación de la práctica: comparar cómo afecta cada una a la generalización a
# longitudes no vistas en entrenamiento (Kazemnejad et al., 2023, arXiv:2305.19466).
#
# Variantes (clave -> clase):
#   "learned"    LearnedAbsolutePE     APE aprendida: es la del modelo base (GPT-2)
#   "sinusoidal" SinusoidalAbsolutePE  APE sinusoidal (Vaswani et al.); es la APE que
#                                      usa el artículo, porque la aprendida no puede
#                                      producir posiciones que no ha visto
#   "nope"       NoPositionalEncoding  sin codificación posicional
#   "rope"       RotaryPE              RoPE: rota Q y K según la posición
#   "alibi"      ALiBiPE               ALiBi: resta a la puntuación un sesgo lineal
#                                      con la distancia (una pendiente por cabeza)
#   "t5"         T5RelativeBiasPE      sesgo relativo aprendido de T5 (por cubetas)
#
# Interfaz común (PositionalEncoding). Hay dos puntos de enganche en el modelo:
#   1) a la entrada:        x = pe.add_to_embeddings(tok_emb)      # APE
#   2) dentro de la atención, con q, k de forma (B, H, T, head_dim):
#        q, k = pe.apply_qk(q, k)                                  # RoPE
#        scores = q @ k^T / sqrt(head_dim)
#        scores = scores + pe.attn_bias(T, T)                      # ALiBi, T5
#      (el sesgo se suma DESPUÉS de dividir por sqrt(head_dim) y antes de la
#       máscara causal y del softmax). `reference_attention` implementa
#       exactamente este orden y es la referencia para la atención del modelo.
#
# Todas las variantes que no aplican un punto de enganche lo dejan como identidad
# (o devuelven None en el sesgo), así que el modelo puede llamar siempre a los
# tres métodos sin preguntar de qué variante se trata.
#
# Uso:
#   pe = build_positional_encoding("rope", emb_dim=128, n_heads=4, context_length=128)
#   input_pe, layer_pes = build_positional_encodings(cfg)   # a partir de cfg["pos_encoding"]
#   python positional.py selftest                           # comprobaciones

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PE_KINDS = ("learned", "sinusoidal", "nope", "rope", "alibi", "t5")


#####################################
# Interfaz común
#####################################
class PositionalEncoding(nn.Module):
    """Clase base: por defecto no hace nada (equivale a NoPE)."""

    name = "base"

    def add_to_embeddings(self, x, offset=0):
        """x: (B, T, emb_dim) -> (B, T, emb_dim). `offset` es la posición del primer token."""
        return x

    def apply_qk(self, q, k, offset=0):
        """q, k: (B, H, T, head_dim) de los tokens actuales (misma longitud T)."""
        return q, k

    def attn_bias(self, num_q, num_k, device=None, dtype=None):
        """Sesgo aditivo (H o 1, num_q, num_k) para las puntuaciones, o None.

        Las `num_q` consultas se consideran las últimas de las `num_k` posiciones
        (num_q == num_k en entrenamiento; num_q < num_k si hubiera caché).
        """
        return None


#####################################
# Sin codificación posicional (NoPE)
#####################################
class NoPositionalEncoding(PositionalEncoding):
    """NoPE: el modelo solo conoce el orden a través de la máscara causal."""

    name = "nope"


#####################################
# APE aprendida (la del modelo base)
#####################################
class LearnedAbsolutePE(PositionalEncoding):
    """Tabla de embeddings de posición aprendida, igual que `pos_emb` de GPTModel.

    No puede representar posiciones >= context_length: si se evalúa con secuencias
    más largas hay que crearla con context_length = longitud máxima de evaluación
    (las posiciones nunca entrenadas conservan su valor inicial aleatorio).
    """

    name = "learned"

    def __init__(self, context_length, emb_dim):
        super().__init__()
        self.pos_emb = nn.Embedding(context_length, emb_dim)

    def add_to_embeddings(self, x, offset=0):
        num_tokens = x.shape[1]
        if offset + num_tokens > self.pos_emb.num_embeddings:
            raise ValueError(
                f"La APE aprendida solo tiene {self.pos_emb.num_embeddings} posiciones y se piden "
                f"{offset + num_tokens}. Crea el modelo con context_length >= la longitud máxima de evaluación.")
        positions = torch.arange(offset, offset + num_tokens, device=x.device)
        return x + self.pos_emb(positions)


#####################################
# APE sinusoidal
#####################################
def sinusoidal_embeddings(num_positions, dim, offset=0, device=None, dtype=torch.float32, base=10000.0):
    """Tabla (num_positions, dim) con p_j = [sin(w_1 j), cos(w_1 j), sin(w_2 j), cos(w_2 j), ...],
    w_i = 1 / base^(2i/dim). Se calcula al vuelo, así que sirve para cualquier posición."""
    if dim % 2 != 0:
        raise ValueError("La dimensión debe ser par para la codificación sinusoidal")
    positions = torch.arange(offset, offset + num_positions, device=device, dtype=torch.float32).unsqueeze(1)
    inv_freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(base) / dim))
    angles = positions * inv_freq  # (num_positions, dim/2)
    table = torch.zeros(num_positions, dim, device=device, dtype=torch.float32)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table.to(dtype)


class SinusoidalAbsolutePE(PositionalEncoding):
    """APE no paramétrica: se suma a los embeddings de tokens, sin límite de longitud."""

    name = "sinusoidal"

    def __init__(self, emb_dim, base=10000.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.base = base

    def add_to_embeddings(self, x, offset=0):
        table = sinusoidal_embeddings(x.shape[1], self.emb_dim, offset, x.device, x.dtype, self.base)
        return x + table


#####################################
# RoPE (Rotary Position Embedding)
#####################################
class RotaryPE(PositionalEncoding):
    """Rota Q y K, por pares de dimensiones, un ángulo proporcional a la posición.

    Convención "mitades" (la de Llama y Hugging Face): la dimensión i se empareja con
    la i + head_dim/2. Es equivalente a la de pares consecutivos salvo una permutación
    de las dimensiones, que las matrices W_q y W_k aprenden de todas formas. Como
    consecuencia de la rotación, el producto q_m · k_n solo depende de m - n.
    No se aplica a V.
    """

    name = "rope"

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim debe ser par para RoPE")
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # no entra en el state_dict

    def _cos_sin(self, num_tokens, offset, device):
        positions = torch.arange(offset, offset + num_tokens, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                   # (T, head_dim)
        return emb.cos(), emb.sin()

    def _rotate(self, x, cos, sin):
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        # Se calcula en float32 y se devuelve al tipo original (evita errores en fp16/bf16)
        return (x.float() * cos + rotated.float() * sin).to(x.dtype)

    def apply_qk(self, q, k, offset=0):
        if q.shape[-2] != k.shape[-2]:
            raise ValueError("RoPE espera q y k de los tokens actuales, con la misma longitud")
        cos, sin = self._cos_sin(q.shape[-2], offset, q.device)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


#####################################
# ALiBi
#####################################
def alibi_slopes(n_heads):
    """Pendientes de ALiBi (Press et al., 2022): para n potencia de 2, 2^(-8i/n) con i = 1..n
    (con 8 cabezas: 1/2, 1/4, ..., 1/256). Para otros n se interpolan como en el código original."""
    def power_of_2(n):
        start = 2.0 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return power_of_2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return power_of_2(closest) + power_of_2(2 * closest)[0::2][: n_heads - closest]


class ALiBiPE(PositionalEncoding):
    """Sesgo -m_h * (i - j) en la puntuación de la consulta i con la clave j <= i.
    Sin parámetros: favorece los tokens recientes (sesgo de recencia)."""

    name = "alibi"

    def __init__(self, n_heads):
        super().__init__()
        self.register_buffer("slopes", torch.tensor(alibi_slopes(n_heads), dtype=torch.float32),
                             persistent=False)

    def attn_bias(self, num_q, num_k, device=None, dtype=None):
        device = device if device is not None else self.slopes.device
        q_pos = torch.arange(num_k - num_q, num_k, device=device)
        k_pos = torch.arange(num_k, device=device)
        distance = (q_pos[:, None] - k_pos[None, :]).clamp(min=0).to(torch.float32)  # (num_q, num_k)
        bias = -self.slopes.to(device)[:, None, None] * distance                     # (H, num_q, num_k)
        return bias.to(dtype if dtype is not None else torch.float32)


#####################################
# Sesgo relativo de T5
#####################################
def t5_relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
    """Cubeta de cada distancia relativa (memoria - consulta), como en T5.

    La mitad de las cubetas se reparte linealmente entre las distancias pequeñas y la
    otra mitad logarítmicamente hasta `max_distance`; las distancias mayores comparten
    la última cubeta. En el caso causal (bidirectional=False) solo cuentan las
    posiciones pasadas.
    """
    relative_buckets = torch.zeros_like(relative_position)
    if bidirectional:
        num_buckets //= 2
        relative_buckets = relative_buckets + (relative_position > 0).long() * num_buckets
        relative_position = relative_position.abs()
    else:
        relative_position = -torch.clamp(relative_position, max=0)

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    # clamp(min=1) evita log(0); esa rama no se usa para las distancias pequeñas
    large = max_exact + (
        torch.log(relative_position.clamp(min=1).float() / max_exact)
        / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).long()
    large = torch.clamp(large, max=num_buckets - 1)
    return relative_buckets + torch.where(is_small, relative_position, large)


class T5RelativeBiasPE(PositionalEncoding):
    """Sesgo aprendido por cubeta de distancia y por cabeza, sumado a las puntuaciones.

    En T5 una sola tabla se comparte entre todas las capas: para imitarlo, pasa la misma
    instancia a todas las capas de atención. Con una instancia por capa cada capa
    aprende la suya.
    """

    name = "t5"

    def __init__(self, n_heads, num_buckets=32, max_distance=128, init_std=0.02):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)
        nn.init.normal_(self.relative_attention_bias.weight, std=init_std)

    def attn_bias(self, num_q, num_k, device=None, dtype=None):
        device = device if device is not None else self.relative_attention_bias.weight.device
        q_pos = torch.arange(num_k - num_q, num_k, device=device)
        k_pos = torch.arange(num_k, device=device)
        relative_position = k_pos[None, :] - q_pos[:, None]  # memoria - consulta
        buckets = t5_relative_position_bucket(
            relative_position, bidirectional=False,
            num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(buckets)       # (num_q, num_k, H)
        values = values.permute(2, 0, 1)                      # (H, num_q, num_k)
        return values.to(dtype) if dtype is not None else values


#####################################
# Construcción
#####################################
def build_positional_encoding(kind, emb_dim, n_heads, context_length,
                              rope_base=10000.0, t5_num_buckets=32, t5_max_distance=128):
    """Crea la codificación posicional `kind` (una de PE_KINDS)."""
    kind = kind.lower()
    if kind == "learned":
        return LearnedAbsolutePE(context_length, emb_dim)
    if kind == "sinusoidal":
        return SinusoidalAbsolutePE(emb_dim)
    if kind == "nope":
        return NoPositionalEncoding()
    if kind == "rope":
        return RotaryPE(emb_dim // n_heads, base=rope_base)
    if kind == "alibi":
        return ALiBiPE(n_heads)
    if kind == "t5":
        return T5RelativeBiasPE(n_heads, t5_num_buckets, t5_max_distance)
    raise ValueError(f"Codificación posicional desconocida: {kind!r}. Opciones: {PE_KINDS}")


def build_positional_encodings(cfg):
    """Devuelve (input_pe, layer_pes) a partir de la configuración del modelo.

    Claves de `cfg` (todas opcionales salvo las del modelo base):
      pos_encoding   una de PE_KINDS (por defecto "learned", como el modelo base)
      rope_base, t5_num_buckets, t5_max_distance
      nope_every     solo con "rope": cada N capas (N, 2N, ...) usa NoPE en vez de RoPE
                     (híbrido por capas; None o 0 = desactivado)

    `input_pe` actúa sobre los embeddings de entrada y `layer_pes[i]` dentro de la
    atención de la capa i. Salvo en el híbrido, todas las capas comparten la misma
    instancia (que es lo que hace T5).
    """
    kind = cfg.get("pos_encoding", "learned")
    pe = build_positional_encoding(
        kind, cfg["emb_dim"], cfg["n_heads"], cfg["context_length"],
        rope_base=cfg.get("rope_base", 10000.0),
        t5_num_buckets=cfg.get("t5_num_buckets", 32),
        t5_max_distance=cfg.get("t5_max_distance", 128))

    nope_every = cfg.get("nope_every")
    if nope_every:
        if kind != "rope":
            raise ValueError("nope_every solo tiene sentido con pos_encoding='rope'")
        nope = NoPositionalEncoding()
        layer_pes = [nope if (i + 1) % nope_every == 0 else pe for i in range(cfg["n_layers"])]
    else:
        layer_pes = [pe] * cfg["n_layers"]
    return pe, layer_pes


#####################################
# Atención de referencia
#####################################
def reference_attention(q, k, v, pe, dropout_p=0.0, training=False):
    """Atención causal con la codificación `pe`. q, k, v: (B, H, T, head_dim).

    Fija el orden de las operaciones que debe respetar la atención del modelo:
    RoPE sobre q y k -> producto escalar / sqrt(head_dim) -> + sesgo -> máscara
    causal -> softmax -> dropout -> @ v.
    """
    q, k = pe.apply_qk(q, k)
    num_q, num_k = q.shape[-2], k.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    bias = pe.attn_bias(num_q, num_k, q.device, scores.dtype)
    if bias is not None:
        scores = scores + bias.unsqueeze(0)
    causal = torch.triu(torch.ones(num_q, num_k, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    weights = F.dropout(torch.softmax(scores, dim=-1), dropout_p, training)
    return weights @ v


#####################################
# Autotest: python positional.py selftest
#####################################
def _rope_complex_reference(x, base=10000.0, offset=0):
    """RoPE con aritmética compleja en doble precisión (misma convención de mitades)."""
    head_dim, num_tokens = x.shape[-1], x.shape[-2]
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).double() / head_dim))
    angles = torch.outer(torch.arange(offset, offset + num_tokens).double(), inv_freq)
    rotated = torch.complex(x[..., :half].double(), x[..., half:].double()) \
        * torch.polar(torch.ones_like(angles), angles)
    return torch.cat([rotated.real, rotated.imag], dim=-1).float()


def run_selftest():
    outcomes = []

    def check(name, cond, detail=""):
        outcomes.append(bool(cond))
        print(f"[{'OK' if cond else 'FALLO'}] {name} {detail}".rstrip())

    torch.manual_seed(0)
    B, H, T, hd = 2, 4, 24, 16
    dim = H * hd
    q, k, v = (torch.randn(B, H, T, hd) for _ in range(3))
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    # --- Fábrica ---------------------------------------------------------------
    sizes = {kind: sum(p.numel() for p in build_positional_encoding(kind, dim, H, 128).parameters())
             for kind in PE_KINDS}
    check("Parámetros: learned = ctx*dim, t5 = cubetas*cabezas, el resto 0",
          sizes == {"learned": 128 * dim, "sinusoidal": 0, "nope": 0, "rope": 0, "alibi": 0, "t5": 32 * H},
          f"({sizes})")
    try:
        build_positional_encoding("xyz", dim, H, 128)
        check("Una clave desconocida lanza ValueError", False)
    except ValueError:
        check("Una clave desconocida lanza ValueError", True)
    cfg = {"emb_dim": dim, "n_heads": H, "context_length": 64, "n_layers": 8}
    check("Por defecto se usa la APE aprendida (como el modelo base)",
          build_positional_encodings(cfg)[0].name == "learned")
    pe_in, layers = build_positional_encodings({**cfg, "pos_encoding": "rope", "nope_every": 4})
    check("Híbrido nope_every=4: capas 4 y 8 sin RoPE, el resto con RoPE",
          [p.name for p in layers] == ["rope"] * 3 + ["nope"] + ["rope"] * 3 + ["nope"])

    # --- NoPE ------------------------------------------------------------------
    nope = NoPositionalEncoding()
    check("NoPE: atención == scaled_dot_product_attention(is_causal=True) de PyTorch",
          torch.allclose(reference_attention(q, k, v, nope),
                         F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5))
    x = torch.randn(B, T, dim)
    check("NoPE: no modifica los embeddings de entrada", torch.equal(nope.add_to_embeddings(x), x))

    # --- APE aprendida: equivale a la del modelo base ---------------------------
    learned = LearnedAbsolutePE(32, dim)
    check("APE aprendida: x + pos_emb[0..T-1]",
          torch.allclose(learned.add_to_embeddings(x), x + learned.pos_emb.weight[:T]))
    check("APE aprendida: con offset usa las posiciones correctas",
          torch.allclose(learned.add_to_embeddings(x[:, :4], offset=5), x[:, :4] + learned.pos_emb.weight[5:9]))
    try:
        learned.add_to_embeddings(torch.randn(1, 33, dim))
        check("APE aprendida: error claro si se supera su tabla", False)
    except ValueError:
        check("APE aprendida: error claro si se supera su tabla", True)
    try:
        from model_base_adaptado import GPTModel, get_small_config
        base = GPTModel(get_small_config(50, context_length=32, emb_dim=dim, n_heads=H, n_layers=1))
        pe = LearnedAbsolutePE(32, dim)
        pe.pos_emb.weight.data.copy_(base.pos_emb.weight.data)
        ids = torch.randint(0, 50, (2, 10))
        expected = base.tok_emb(ids) + base.pos_emb(torch.arange(10))
        check("APE aprendida == tok_emb + pos_emb del GPTModel de model_base_adaptado.py",
              torch.allclose(pe.add_to_embeddings(base.tok_emb(ids)), expected))
    except ImportError:
        print("[--] model_base_adaptado.py no está en la ruta: se omite la comparación con GPTModel")

    # --- APE sinusoidal --------------------------------------------------------
    table = sinusoidal_embeddings(50, 8)
    j, i = 7, 2  # posición 7, par de dimensiones i=2 (columnas 4 y 5)
    w = 1.0 / 10000 ** (2 * i / 8)
    check("Sinusoidal: coincide con la fórmula sin(w_i j), cos(w_i j)",
          abs(table[j, 2 * i].item() - math.sin(w * j)) < 1e-5 and abs(table[j, 2 * i + 1].item() - math.cos(w * j)) < 1e-5)
    check("Sinusoidal: la posición 0 es [0, 1, 0, 1, ...]", torch.allclose(table[0], torch.tensor([0., 1.] * 4)))
    sin_pe = SinusoidalAbsolutePE(dim)
    long_x = torch.randn(1, 500, dim)
    check("Sinusoidal: sirve para secuencias de cualquier longitud (500 > cualquier tabla)",
          sin_pe.add_to_embeddings(long_x).shape == long_x.shape)
    check("Sinusoidal: offset equivale a recortar",
          torch.allclose(sin_pe.add_to_embeddings(x, 0)[:, 5:], sin_pe.add_to_embeddings(x[:, 5:], 5), atol=1e-6))

    # --- RoPE ------------------------------------------------------------------
    rope = RotaryPE(hd)
    rq, rk = rope.apply_qk(q, k)
    check("RoPE == implementación independiente con números complejos",
          torch.allclose(rq, _rope_complex_reference(q), atol=1e-5) and torch.allclose(rk, _rope_complex_reference(k), atol=1e-5))
    check("RoPE conserva la norma de los vectores", torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5))
    check("RoPE: la posición 0 no cambia el vector", torch.allclose(rq[..., 0, :], q[..., 0, :], atol=1e-6))
    qv = torch.randn(1, 1, 1, hd).expand(1, 1, T, hd).contiguous()
    kv = torch.randn(1, 1, 1, hd).expand(1, 1, T, hd).contiguous()
    rqv, rkv = rope.apply_qk(qv, kv)
    scores = (rqv @ rkv.transpose(-2, -1))[0, 0]
    shift = 5
    check("RoPE: q_m·k_n solo depende de m-n (desplazar ambas posiciones no cambia la puntuación)",
          torch.allclose(scores[:-shift, :-shift], scores[shift:, shift:], atol=1e-4))
    oq, ok_ = rope.apply_qk(q[:, :, 6:], k[:, :, 6:], offset=6)
    check("RoPE: offset equivale a recortar la secuencia completa",
          torch.allclose(oq, rq[:, :, 6:], atol=1e-5) and torch.allclose(ok_, rk[:, :, 6:], atol=1e-5))
    check("RoPE conserva el tipo (bfloat16)", rope.apply_qk(q.bfloat16(), k.bfloat16())[0].dtype == torch.bfloat16)
    check("RoPE: atención == SDPA de PyTorch con q, k rotados",
          torch.allclose(reference_attention(q, k, v, rope),
                         F.scaled_dot_product_attention(rq, rk, v, is_causal=True), atol=1e-5))

    # --- ALiBi -----------------------------------------------------------------
    check("ALiBi: 8 cabezas -> pendientes 1/2, 1/4, ..., 1/256",
          torch.allclose(torch.tensor(alibi_slopes(8)), torch.tensor([2.0 ** -(i + 1) for i in range(8)])))
    check("ALiBi: 6 cabezas (no potencia de 2) -> interpolación del código original",
          torch.allclose(torch.tensor(alibi_slopes(6)),
                         torch.tensor([0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125])))
    alibi = ALiBiPE(H)
    bias = alibi.attn_bias(T, T)
    check("ALiBi: sesgo (H, T, T) con valor -m_h*(i-j) y 0 en la diagonal",
          bias.shape == (H, T, T) and abs(bias[1, 10, 3].item() + alibi.slopes[1].item() * 7) < 1e-6
          and torch.all(torch.diagonal(bias, dim1=1, dim2=2) == 0))
    check("ALiBi: el sesgo decrece al alejarse de la consulta", torch.all(bias[:, 20, :21].diff(dim=-1) > 0))
    mask = bias.masked_fill(causal_mask, float("-inf"))
    check("ALiBi: atención == SDPA de PyTorch con attn_mask = sesgo + máscara causal",
          torch.allclose(reference_attention(q, k, v, alibi), F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5))
    check("ALiBi: num_q < num_k equivale a las últimas filas",
          torch.allclose(alibi.attn_bias(3, T), bias[:, -3:, :]))

    # --- Sesgo relativo de T5 --------------------------------------------------
    d = torch.arange(0, 600).long()
    buckets = t5_relative_position_bucket(-d)  # distancias pasadas: memoria - consulta = -d
    check("T5: distancias 0..15 -> cubetas exactas 0..15 y la 16 empieza la zona logarítmica",
          torch.equal(buckets[:16], torch.arange(16)) and buckets[16].item() == 16)
    check("T5: cubetas no decrecientes; distancia 127 -> 31; >= 128 comparten la última",
          torch.all(buckets.diff() >= 0) and buckets[127].item() == 31 and torch.all(buckets[128:] == 31))
    try:
        from transformers.models.t5.modeling_t5 import T5Attention
        ref = T5Attention._relative_position_bucket(-d, bidirectional=False, num_buckets=32, max_distance=128)
        check("T5: cubetas == implementación de Hugging Face", torch.equal(buckets, ref))
        ref_bi = T5Attention._relative_position_bucket(torch.arange(-40, 40), bidirectional=True, num_buckets=32, max_distance=128)
        check("T5 (bidireccional): cubetas == Hugging Face",
              torch.equal(t5_relative_position_bucket(torch.arange(-40, 40), bidirectional=True), ref_bi))
    except Exception:
        print("[--] transformers no instalado: se omite la comparación con T5 de Hugging Face")
    t5 = T5RelativeBiasPE(H)
    t5_bias = t5.attn_bias(T, T)
    check("T5: sesgo (H, T, T) que depende solo de la distancia (mismo valor en cada diagonal)",
          t5_bias.shape == (H, T, T) and torch.allclose(t5_bias[:, 10, 4], t5_bias[:, 20, 14]))
    t5_bias.sum().backward()
    check("T5: el sesgo es entrenable (llega gradiente a la tabla)", t5.relative_attention_bias.weight.grad is not None)

    # --- Causalidad común a todas ---------------------------------------------
    for kind in PE_KINDS:
        pe = build_positional_encoding(kind, dim, H, 64)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, -1] += 1.0
        v2[:, :, -1] += 1.0
        same = torch.allclose(reference_attention(q, k, v, pe)[:, :, :-1],
                              reference_attention(q, k2, v2, pe)[:, :, :-1], atol=1e-6)
        check(f"Causalidad con '{kind}': cambiar el último token no altera los anteriores", same)

    ok = all(outcomes)
    print(f"\n{sum(outcomes)}/{len(outcomes)} comprobaciones correctas" + ("" if ok else " (hay fallos)"))
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(0 if run_selftest() else 1)
    print(__doc__ if __doc__ else "Uso: python positional.py selftest")
    print(f"Variantes disponibles: {', '.join(PE_KINDS)}")
    print("Uso: python positional.py selftest")
