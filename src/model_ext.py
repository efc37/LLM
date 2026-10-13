# model_ext.py
#
# GPT con codificación posicional intercambiable: engancha positional.py al modelo
# base (model_base_adaptado.py). Con pos_encoding="learned" reproduce exactamente el modelo
# base, que es la referencia contra la que se comparan las demás variantes.
#
# Qué cambia respecto a model_base_adaptado.GPTModel:
#   - La atención (MultiHeadAttentionExt) llama a los hooks de la codificación
#     posicional: apply_qk (RoPE) y attn_bias (ALiBi, T5), en el orden fijado por
#     positional.reference_attention.
#   - GPTModelExt suma add_to_embeddings (APE) a la entrada.
#   - La máscara causal se crea al vuelo, sin buffer de tamaño fijo: con las
#     variantes no aprendidas el modelo acepta secuencias más largas que las de
#     entrenamiento (la APE aprendida lanza un error claro).
#   - Los nombres de los parámetros coinciden con los del base (tok_emb, trf_blocks,
#     att.W_query, ..., final_norm, out_head), salvo la tabla posicional, que vive en
#     pos_enc. `load_from_base` copia los pesos de un GPTModel del base.
#
# Configuración (las claves del base más estas, todas opcionales):
#   pos_encoding   "learned" (por defecto), "sinusoidal", "nope", "rope", "alibi", "t5"
#   max_positions  tamaño de la tabla de la APE aprendida (por defecto context_length);
#                  útil para entrenar con secuencias cortas y evaluar con otras más largas
#   rope_base, t5_num_buckets, t5_max_distance, nope_every   (ver positional.py)
# Nota: context_length sigue siendo la longitud de entrenamiento que usa
# model_base_adaptado.pretrain; el modelo en sí no impone ese límite.
#
# Uso con el pipeline del base (mismos datos, semilla y métricas para todas las variantes):
#   from model_base_adaptado import pretrain, get_small_config, CharTokenizer
#   from model_ext import GPTModelExt, make_config
#   cfg = make_config(get_small_config(tok.n_vocab), "rope")
#   res = pretrain(cfg, text, tok, model_factory=GPTModelExt, seed=123)
#
#   python model_ext.py selftest      # comprobaciones

import math

import torch
import torch.nn as nn

from model_base_adaptado import FeedForward, LayerNorm
from positional import PE_KINDS, build_positional_encodings


def make_config(base_cfg, pos_encoding="learned", **extra):
    """Copia `base_cfg` añadiendo la codificación posicional y opciones extra."""
    cfg = dict(base_cfg)
    cfg["pos_encoding"] = pos_encoding
    cfg.update(extra)
    return cfg


#####################################
# Atención con codificación posicional
#####################################
class MultiHeadAttentionExt(nn.Module):
    """Igual que MultiHeadAttention del base, pero con los hooks de la codificación
    posicional y la máscara causal calculada para la longitud real de la secuencia."""

    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pe):
        b, num_tokens, _ = x.shape

        # (b, T, d_out) -> (b, num_heads, T, head_dim)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Hook 1: RoPE rota q y k (el resto de variantes las deja igual)
        queries, keys = pe.apply_qk(queries, keys)

        attn_scores = queries @ keys.transpose(2, 3) / math.sqrt(self.head_dim)

        # Hook 2: ALiBi / T5 suman un sesgo a las puntuaciones ya escaladas
        bias = pe.attn_bias(num_tokens, num_tokens, x.device, attn_scores.dtype)
        if bias is not None:
            attn_scores = attn_scores + bias.unsqueeze(0)

        # Máscara causal para la longitud actual (sin límite de contexto)
        mask = torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=x.device), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))

        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlockExt(nn.Module):
    """Bloque pre-norm del base; la codificación posicional de la capa llega por forward."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionExt(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pe):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, pe)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModelExt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.context_length = cfg["context_length"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # La tabla de la APE aprendida puede ser mayor que la longitud de entrenamiento
        pe_cfg = {**cfg, "context_length": cfg.get("max_positions", cfg["context_length"])}
        pe, layer_pes = build_positional_encodings(pe_cfg)
        self.pos_enc = pe       # único módulo registrado: lleva parámetros y buffers
        # Referencias por capa (la misma instancia compartida, o NoPE en el híbrido). Es una
        # lista normal a propósito: así no se registran varias veces los mismos parámetros.
        self._layer_pes = layer_pes

        self.trf_blocks = nn.ModuleList([TransformerBlockExt(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    @property
    def pos_emb(self):
        """Tabla de la APE aprendida (como `pos_emb` del base). Solo existe con "learned";
        permite reutilizar load_weights_into_gpt y las funciones del cap. 5 y 6."""
        if self.pos_enc.name != "learned":
            raise AttributeError(f"'{self.pos_enc.name}' no tiene tabla de posiciones aprendida")
        return self.pos_enc.pos_emb

    @property
    def max_seq_len(self):
        """Longitud máxima admitida, o None si no hay límite."""
        return self.pos_emb.num_embeddings if self.pos_enc.name == "learned" else None

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)                  # (batch, num_tokens, emb_dim)
        x = self.pos_enc.add_to_embeddings(x)     # Hook de entrada (solo la APE hace algo)
        x = self.drop_emb(x)
        for block, pe in zip(self.trf_blocks, self._layer_pes):
            x = block(x, pe)
        x = self.final_norm(x)
        return self.out_head(x)                   # (batch, num_tokens, vocab_size)

    def load_from_base(self, base_model):
        """Copia los pesos de un model_base_adaptado.GPTModel (todo salvo lo posicional).

        Con "learned" también copia la tabla de posiciones. Con otras variantes no se
        copia ninguna codificación posicional: quedan con su inicialización.
        """
        state = {}
        for key, value in base_model.state_dict().items():
            if key.endswith(".att.mask"):
                continue
            if key == "pos_emb.weight":
                if self.pos_enc.name != "learned":
                    continue
                key = "pos_enc.pos_emb.weight"
            state[key] = value
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("pos_enc.")]
        if missing or unexpected:
            raise RuntimeError(f"Pesos incompatibles con el base. Faltan: {missing}. Sobran: {unexpected}")


#####################################
# Autotest: python model_ext.py selftest
#####################################
def run_selftest():
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    import numpy as np

    from model_base_adaptado import (CharTokenizer, GPTModel, count_parameters, generate_text_simple,
                            get_small_config, load_checkpoint, load_weights_into_gpt, pretrain,
                            save_checkpoint)
    from positional import build_positional_encoding, reference_attention, sinusoidal_embeddings

    outcomes = []

    def check(name, cond, detail=""):
        outcomes.append(bool(cond))
        print(f"[{'OK' if cond else 'FALLO'}] {name} {detail}".rstrip())

    torch.manual_seed(0)
    V, C, D, H, L = 30, 16, 32, 4, 2
    cfg = get_small_config(V, C, D, H, L, drop_rate=0.0)
    x = torch.randint(0, V, (3, C))

    # --- Equivalencia con el modelo base ---------------------------------------
    base = GPTModel(cfg).eval()
    n_base = count_parameters(base)

    ext = GPTModelExt(make_config(cfg, "learned")).eval()
    ext.load_from_base(base)
    check("learned: mismos parámetros que el base", count_parameters(ext) == n_base, f"({n_base:,})")
    check("learned: logits idénticos a GPTModel de model_base_adaptado (pesos copiados)",
          torch.allclose(base(x), ext(x), atol=1e-6))

    base_nope = GPTModel(cfg).eval()
    base_nope.pos_emb.weight.data.zero_()
    ext_nope = GPTModelExt(make_config(cfg, "nope")).eval()
    ext_nope.load_from_base(base_nope)
    check("nope: logits idénticos al base con la tabla de posiciones a cero",
          torch.allclose(base_nope(x), ext_nope(x), atol=1e-6))

    base_sin = GPTModel(cfg).eval()
    base_sin.pos_emb.weight.data.copy_(sinusoidal_embeddings(C, D))
    ext_sin = GPTModelExt(make_config(cfg, "sinusoidal")).eval()
    ext_sin.load_from_base(base_sin)
    check("sinusoidal: logits idénticos al base con la tabla sinusoidal como pos_emb",
          torch.allclose(base_sin(x), ext_sin(x), atol=1e-6))

    # Pesos de GPT-2 (formato OpenAI, aquí sintéticos) cargados en base y en ext
    rng = np.random.default_rng(0)
    r = lambda *s: rng.standard_normal(s).astype(np.float32)
    gcfg = get_small_config(50, C, D, H, L, 0.0, qkv_bias=True)
    block = lambda: {"attn": {"c_attn": {"w": r(D, 3 * D), "b": r(3 * D)}, "c_proj": {"w": r(D, D), "b": r(D)}},
                     "mlp": {"c_fc": {"w": r(D, 4 * D), "b": r(4 * D)}, "c_proj": {"w": r(4 * D, D), "b": r(D)}},
                     "ln_1": {"g": r(D), "b": r(D)}, "ln_2": {"g": r(D), "b": r(D)}}
    params = {"wpe": r(C, D), "wte": r(50, D), "g": r(D), "b": r(D), "blocks": [block() for _ in range(L)]}
    gb, ge = GPTModel(gcfg), GPTModelExt(make_config(gcfg, "learned"))
    load_weights_into_gpt(gb, params)
    load_weights_into_gpt(ge, params)
    xi = torch.randint(0, 50, (2, C))
    check("learned: load_weights_into_gpt (pesos de GPT-2) funciona igual que en el base",
          torch.allclose(gb.eval()(xi), ge.eval()(xi), atol=1e-5))

    # --- La atención sigue la referencia de positional.py ----------------------
    hd = D // H
    for kind in PE_KINDS:
        pe = build_positional_encoding(kind, D, H, 64)
        att = MultiHeadAttentionExt(D, D, 0.0, H).eval()
        h = torch.randn(2, 20, D)
        q, k, v = (proj(h).view(2, 20, H, hd).transpose(1, 2) for proj in (att.W_query, att.W_key, att.W_value))
        ref = att.out_proj(reference_attention(q, k, v, pe).transpose(1, 2).reshape(2, 20, D))
        check(f"'{kind}': MultiHeadAttentionExt == reference_attention de positional.py",
              torch.allclose(att(h, pe), ref, atol=1e-5))

    # --- Parámetros, causalidad y longitudes -----------------------------------
    expected = {"learned": n_base, "sinusoidal": n_base - C * D, "nope": n_base - C * D,
                "rope": n_base - C * D, "alibi": n_base - C * D, "t5": n_base - C * D + 32 * H}
    got = {kind: count_parameters(GPTModelExt(make_config(cfg, kind))) for kind in PE_KINDS}
    check("Parámetros por variante: base - tabla posicional (+ 32*H en t5, tabla compartida)",
          got == expected, f"({got})")

    x40 = torch.randint(0, V, (2, 40))
    for kind in PE_KINDS:
        model = GPTModelExt(make_config(cfg, kind)).eval()
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % V
        check(f"'{kind}': causalidad del modelo completo",
              torch.allclose(model(x)[:, :-1], model(x2)[:, :-1], atol=1e-6))
        if kind == "learned":
            try:
                model(x40)
                check("learned: error claro con secuencias más largas que su tabla", False)
            except ValueError:
                check("learned: error claro con secuencias más largas que su tabla", True)
        else:
            check(f"'{kind}': acepta secuencias de 40 tokens con context_length={C}",
                  tuple(model(x40).shape) == (2, 40, V))
    wide = GPTModelExt(make_config(cfg, "learned", max_positions=48)).eval()
    check("learned con max_positions=48 acepta 40 tokens", tuple(wide(x40).shape) == (2, 40, V))

    hyb = GPTModelExt(make_config(cfg, "rope", n_layers=4, nope_every=2)).eval()
    full = GPTModelExt(make_config(cfg, "rope", n_layers=4)).eval()
    full.load_state_dict(hyb.state_dict())
    check("Híbrido RoPE/NoPE (nope_every=2): salida distinta de RoPE en todas las capas",
          [p.name for p in hyb._layer_pes] == ["rope", "nope", "rope", "nope"]
          and not torch.allclose(hyb(x), full(x), atol=1e-4))

    # --- Entrenamiento y generación con el pipeline del base -------------------
    text = "abcde fghij " * 200
    tok = CharTokenizer.from_text(text)
    tcfg = get_small_config(tok.n_vocab, 16, 32, 4, 2, drop_rate=0.0)
    for kind in PE_KINDS:
        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet), tempfile.TemporaryDirectory() as tmp:
            res = pretrain(make_config(tcfg, kind), text, tok, model_factory=GPTModelExt, num_epochs=15,
                           batch_size=8, learning_rate=3e-3, eval_freq=20, eval_iter=2,
                           start_context="abcde", device="cpu", seed=7, out_dir=tmp)
            reloaded, rcfg, _ = load_checkpoint(Path(tmp) / "model.pt", model_factory=GPTModelExt)
        xt = torch.randint(0, tok.n_vocab, (2, 12))
        same = torch.allclose(res["model"].eval()(xt), reloaded.eval()(xt), atol=1e-6)
        check(f"'{kind}': pretrain() con GPTModelExt aprende (val loss < 0.5) y el checkpoint se recarga",
              res["final_val_loss"] < 0.5 and same and rcfg["pos_encoding"] == kind,
              f"(val {res['final_val_loss']:.3f})")
    gen = generate_text_simple(GPTModelExt(make_config(cfg, "rope")).eval(), x[:1, :4], 5, 64)
    check("generate_text_simple funciona con GPTModelExt", gen.shape[1] == 9)

    ok = all(outcomes)
    print(f"\n{sum(outcomes)}/{len(outcomes)} comprobaciones correctas" + ("" if ok else " (hay fallos)"))
    return ok


if __name__ == "__main__":
    run_selftest()
    