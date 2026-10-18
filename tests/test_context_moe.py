# Tests del análisis del experimento combinado (scripts/context_ppl_moe.py): reparto del
# router por posición, parámetros activos y diferencias emparejadas. No entrena nada.
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from context_ppl_moe import count_active_params, routing_by_position, summarize  # noqa: E402

from model_base_adaptado import get_small_config  # noqa: E402
from model_ext import make_config  # noqa: E402
from moe_ext import build_ext_model  # noqa: E402


def tiny_model(moe=None):
    cfg = make_config(get_small_config(20, context_length=16, emb_dim=32, n_heads=4, n_layers=2,
                                       drop_rate=0.0), "alibi")
    if moe:
        cfg["moe"] = moe
    torch.manual_seed(0)
    return build_ext_model(cfg)


def test_count_active_params():
    dense_total, dense_active = count_active_params(tiny_model())
    total, active = count_active_params(tiny_model(dict(n_experts=4, top_k=1)))
    per_expert = 2 * 32 * 128 + 128 + 32          # Linear(32,128) + Linear(128,32) con sesgos
    assert dense_total == dense_active
    assert total - active == 2 * 3 * per_expert   # 2 capas x (E - k) expertos inactivos
    assert active == dense_active + 2 * 32 * 4    # el denso + los routers (D x E por capa)


def test_routing_by_position_shapes_and_reference():
    model = tiny_model(dict(n_experts=4, top_k=1)).eval()
    tokens = torch.randint(0, 20, (10 * 64,))
    out = routing_by_position(model, tokens, context=64, bucket=16, n_windows=10)
    assert out["n_windows"] == 10 and len(out["layers"]) == 2
    for layer in out["layers"]:
        assert len(layer["fractions"]) == 4                   # 64 / 16 tramos
        assert all(abs(sum(f) - 1) < 1e-6 for f in layer["fractions"])
        assert layer["tv_vs_first"][0] == 0.0                 # el primer tramo frente a sí mismo
        assert all(0.0 <= tv <= 1.0 for tv in layer["tv_vs_first"])
    # Las fracciones del primer tramo coinciden con un recuento hecho a mano
    with torch.no_grad():
        model(tokens.view(10, 64))
    first = model.trf_blocks[0].ff.last_indices.view(10, 64)[:, :16].reshape(-1)
    expected = torch.bincount(first, minlength=4) / first.numel()
    assert torch.allclose(torch.tensor(out["layers"][0]["fractions"][0]), expected)


def test_routing_by_position_uses_available_windows():
    model = tiny_model(dict(n_experts=4, top_k=2)).eval()
    out = routing_by_position(model, torch.randint(0, 20, (100,)), context=32, bucket=16, n_windows=40)
    assert out["n_windows"] == 3                              # solo caben 3 ventanas de 32


def _record(pe, ffn, seed, val, ppl):
    return {"pos_encoding": pe, "ffn": ffn, "seed": seed, "n_params": 10, "n_params_active": 5,
            "val_loss_train_ctx": val, "train_tokens_per_s": 1.0, "train_time_s": 1.0,
            "ppl": {"last": {"128": ppl, "1024": ppl * 2}, "full": {"128": ppl, "1024": ppl * 2}}}


def test_summarize_paired_deltas():
    results = [_record("alibi", "dense", s, 1.8 + 0.01 * s, 6.0) for s in (0, 1, 2)]
    results += [_record("alibi", "moe_e4k1", s, 1.7 + 0.01 * s, 5.5) for s in (0, 1, 2)]
    summary = summarize(results)
    delta = summary["alibi|moe_e4k1"]["paired_delta_vs_dense"]
    assert delta["seeds"] == [0, 1, 2]
    assert delta["val_loss_train_ctx"][0] == pytest.approx(-0.1)
    assert delta["val_loss_train_ctx"][1] == pytest.approx(0.0, abs=1e-12)   # diferencia constante
    assert delta["ppl_last"][1024][0] == pytest.approx(-1.0)
    assert "paired_delta_vs_dense" not in summary["alibi|dense"]
