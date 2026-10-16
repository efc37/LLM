# validate_rope_hf.py
#
# Validación numérica de RoPE (positional.RotaryPE) frente a la implementación de
# referencia de Hugging Face transformers (la de Llama): misma entrada, misma base,
# mismas posiciones, y las salidas tensoriales deben coincidir salvo error de coma
# flotante. Es la vía "comparar numéricamente con una librería consolidada" del
# enunciado de la práctica.
#
# Qué se compara:
#   - las frecuencias inv_freq (rope.inv_freq de RotaryPE frente al de LlamaRotaryEmbedding),
#   - q y k rotados, con apply_rotary_pos_emb de Hugging Face y cos/sin de
#     LlamaRotaryEmbedding, para varias bases, dimensiones de cabeza y desplazamientos.
#
# Requiere:  pip install transformers
# Uso:       python validate_rope_hf.py
#
# La API de transformers cambia entre versiones; el constructor de LlamaRotaryEmbedding
# se prueba en el orden: LlamaConfig con rope_theta, LlamaConfig con rope_parameters y
# la firma antigua (dim, max_position_embeddings, base). Si ninguna funciona, el script
# lo dice y termina con código 2 en vez de dar un falso fallo.

import sys

import torch

from positional import RotaryPE


def build_hf_rope(head_dim, base, n_heads=4, max_positions=4096):
    """Devuelve el módulo LlamaRotaryEmbedding de Hugging Face para (head_dim, base)."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    hidden = head_dim * n_heads
    common = dict(hidden_size=hidden, num_attention_heads=n_heads, num_hidden_layers=1,
                  intermediate_size=2 * hidden, max_position_embeddings=max_positions)
    attempts = [
        lambda: LlamaRotaryEmbedding(config=LlamaConfig(**common, rope_theta=base)),
        lambda: LlamaRotaryEmbedding(config=LlamaConfig(
            **common, rope_parameters={"rope_type": "default", "rope_theta": base})),
        lambda: LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=max_positions, base=base),
    ]
    errors = []
    for attempt in attempts:
        try:
            rope = attempt()
            # Comprueba que la base se ha aplicado (la segunda frecuencia es base^(-2/head_dim))
            expected = base ** (-2.0 / head_dim)
            if abs(float(rope.inv_freq[1]) - expected) / expected < 1e-4:
                return rope
            errors.append("la base no se aplicó")
        except Exception as e:  # versión de transformers con otra API
            errors.append(f"{type(e).__name__}: {str(e)[:80]}")
    raise RuntimeError("No se pudo construir LlamaRotaryEmbedding con esta versión de transformers: "
                       + " | ".join(errors))


def run_validation():
    try:
        import transformers
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:
        print("transformers no está instalado: pip install transformers")
        return None

    print(f"transformers {transformers.__version__} | torch {torch.__version__}")
    outcomes = []

    def check(name, cond, detail=""):
        outcomes.append(bool(cond))
        print(f"[{'OK' if cond else 'FALLO'}] {name} {detail}".rstrip())

    torch.manual_seed(0)
    B, H, T = 2, 4, 24
    for base in (10000.0, 500000.0):  # Llama 2 y Llama 3
        for head_dim in (16, 64):
            try:
                hf_rope = build_hf_rope(head_dim, base)
            except RuntimeError as e:
                print(f"[--] {e}")
                return None
            ours = RotaryPE(head_dim, base=base)
            tag = f"base={base:g}, head_dim={head_dim}"

            check(f"inv_freq coincide ({tag})",
                  torch.allclose(ours.inv_freq, hf_rope.inv_freq.float(), rtol=1e-6, atol=1e-9))

            q, k = torch.randn(B, H, T, head_dim), torch.randn(B, H, T, head_dim)
            for offset in (0, 7, 100):
                positions = (offset + torch.arange(T)).unsqueeze(0).expand(B, -1)
                cos, sin = hf_rope(q, positions)
                hf_q, hf_k = apply_rotary_pos_emb(q, k, cos, sin)  # unsqueeze_dim=1 por defecto
                our_q, our_k = ours.apply_qk(q, k, offset=offset)
                err = max((our_q - hf_q).abs().max().item(), (our_k - hf_k).abs().max().item())
                check(f"q y k rotados coinciden con Hugging Face ({tag}, offset={offset})",
                      torch.allclose(our_q, hf_q, atol=1e-5) and torch.allclose(our_k, hf_k, atol=1e-5),
                      f"(error máx. {err:.2e})")

    # Control: la comparación no es trivial (la rotación sí cambia los vectores)
    ctrl = RotaryPE(16)
    qc = torch.randn(B, H, T, 16)
    check("Control: RoPE modifica q para posiciones > 0 (la comparación no es trivial)",
          not torch.allclose(ctrl.apply_qk(qc, qc, offset=7)[0], qc, atol=1e-3))

    # Consecuencia observable: las puntuaciones de atención q·k^T son las mismas
    head_dim = 16
    hf_rope, ours = build_hf_rope(head_dim, 10000.0), RotaryPE(head_dim)
    q, k = torch.randn(B, H, T, head_dim), torch.randn(B, H, T, head_dim)
    cos, sin = hf_rope(q, torch.arange(T).unsqueeze(0).expand(B, -1))
    hf_q, hf_k = apply_rotary_pos_emb(q, k, cos, sin)
    our_q, our_k = ours.apply_qk(q, k)
    check("Puntuaciones q·k^T idénticas a las de Hugging Face",
          torch.allclose(our_q @ our_k.transpose(-2, -1), hf_q @ hf_k.transpose(-2, -1), atol=1e-4))

    ok = all(outcomes)
    print(f"\n{sum(outcomes)}/{len(outcomes)} comprobaciones correctas" + ("" if ok else " (hay fallos)"))
    return ok


if __name__ == "__main__":
    result = run_validation()
    sys.exit(2 if result is None else (0 if result else 1))
