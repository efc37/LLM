# Tests de la unión de las dos ampliaciones: MoE (moe.py) sobre el GPT con
# codificación posicional del equipo (model_ext.py). pytest tests/test_moe_ext.py -v
import copy

import pytest
import torch
import torch.nn.functional as F

from model_base_adaptado import CharTokenizer, get_small_config, load_checkpoint, pretrain, save_checkpoint
from model_ext import GPTModelExt, make_config
from moe import attach_moe, moe_aux_loss, moe_expert_stats, moe_layers, upcycle_from_dense
from moe_ext import build_ext_model
from pe_labels import parse_pe_label

PE_LABELS = ["learned", "sinusoidal", "nope", "alibi", "t5", "rope_nope2"]


def ext_cfg(pe_label, moe=None, n_layers=4):
    kind, extra = parse_pe_label(pe_label)
    cfg = make_config(get_small_config(30, context_length=16, emb_dim=32, n_heads=4,
                                       n_layers=n_layers, drop_rate=0.0), kind, **extra)
    if moe:
        cfg["moe"] = moe
    return cfg


def test_build_ext_model_without_moe_is_team_model():
    cfg = ext_cfg("alibi")
    torch.manual_seed(0)
    ours = build_ext_model(cfg).eval()
    torch.manual_seed(0)
    theirs = GPTModelExt(cfg).eval()
    assert type(ours) is GPTModelExt and not moe_layers(ours)
    idx = torch.randint(0, 30, (2, 16))
    assert torch.equal(ours(idx), theirs(idx))


def test_attach_moe_respects_every_n_layers():
    model = build_ext_model(ext_cfg("nope", dict(n_experts=4, top_k=2, every_n_layers=2)))
    kinds = [type(blk.ff).__name__ for blk in model.trf_blocks]
    assert kinds == ["FeedForward", "MoEFeedForward", "FeedForward", "MoEFeedForward"]


@pytest.mark.parametrize("pe_label", PE_LABELS)
@pytest.mark.parametrize("moe", [
    dict(n_experts=1, top_k=1),                       # 1 experto ≡ FFN
    dict(n_experts=4, top_k=2),                       # expertos idénticos, pesos suman 1
    dict(n_experts=4, top_k=2, every_n_layers=2, dispatch="loop"),
])
def test_upcycled_ext_moe_equals_team_dense_model(pe_label, moe):
    """Con cada codificación posicional del equipo, el modelo con MoE inicializado desde
    el denso (cada experto = su FFN) da exactamente los mismos logits."""
    torch.manual_seed(123)
    dense = GPTModelExt(ext_cfg(pe_label)).eval()
    model = upcycle_from_dense(build_ext_model(ext_cfg(pe_label, moe)), dense).eval()
    idx = torch.randint(0, 30, (3, 16))
    with torch.no_grad():
        assert torch.allclose(model(idx), dense(idx), atol=1e-5)
        # También con secuencias más largas que las de entrenamiento (salvo la APE aprendida)
        if pe_label != "learned":
            long_idx = torch.randint(0, 30, (2, 48))
            assert torch.allclose(model(long_idx), dense(long_idx), atol=1e-5)


@pytest.mark.parametrize("moe", [
    dict(n_experts=4, top_k=1, aux_loss_coef=0.3),
    dict(n_experts=8, top_k=2, aux_loss_coef=0.3, every_n_layers=2),
])
def test_injected_aux_loss_equals_explicit_loss(moe):
    """Optimizar solo la CE con la inyección da los MISMOS gradientes que optimizar
    CE + coef · media(L_aux) sin inyección (lo que hace calc_loss_batch_moe)."""
    torch.manual_seed(0)
    injected = build_ext_model(ext_cfg("alibi", moe)).train()
    explicit = copy.deepcopy(injected)
    for layer in moe_layers(explicit):
        layer.aux_inject_coef = 0.0
    idx = torch.randint(0, 30, (4, 16))
    tgt = torch.randint(0, 30, (4, 16))

    ce_inj = F.cross_entropy(injected(idx).flatten(0, 1), tgt.flatten())
    ce_inj.backward()
    ce_exp = F.cross_entropy(explicit(idx).flatten(0, 1), tgt.flatten())
    (ce_exp + moe["aux_loss_coef"] * moe_aux_loss(explicit)).backward()

    assert torch.allclose(ce_inj, ce_exp)   # el forward no cambia
    for (name, p_inj), (_, p_exp) in zip(injected.named_parameters(), explicit.named_parameters()):
        assert torch.allclose(p_inj.grad, p_exp.grad, atol=1e-6), name


def test_injection_gradient_reaches_router_and_everything_upstream():
    """L_aux depende de las probabilidades del router, que dependen de sus pesos Y de su
    entrada (todo lo que hay antes en la red). Por eso la inyección cambia el gradiente
    de los routers y de lo que está aguas arriba (embeddings, atención, capas previas),
    pero NO el de lo que solo está aguas abajo de todos los routers: los expertos de la
    última capa MoE, la normalización final y la cabeza de salida."""
    torch.manual_seed(0)
    injected = build_ext_model(ext_cfg("nope", dict(n_experts=4, top_k=1, aux_loss_coef=0.5))).train()
    plain = copy.deepcopy(injected)
    for layer in moe_layers(plain):
        layer.aux_inject_coef = 0.0
    idx = torch.randint(0, 30, (4, 16))
    for model in (injected, plain):
        F.cross_entropy(model(idx).flatten(0, 1), idx.flatten()).backward()

    grads = {name: (a.grad, b.grad) for (name, a), (_, b)
             in zip(injected.named_parameters(), plain.named_parameters())}
    last = len(injected.trf_blocks) - 1
    downstream_only = [n for n in grads if n.startswith(("out_head.", "final_norm."))
                       or n.startswith(f"trf_blocks.{last}.ff.experts.")]
    upstream = ["tok_emb.weight", "trf_blocks.0.att.W_query.weight", f"trf_blocks.{last}.norm2.scale"]
    routers = [n for n in grads if ".router." in n]

    assert len(routers) == 4 and len(downstream_only) > 0
    for name in routers + upstream:
        assert not torch.allclose(*grads[name]), name
    for name in downstream_only:
        assert torch.allclose(*grads[name], atol=1e-7), name


def test_injection_inactive_in_eval_and_without_grad():
    torch.manual_seed(0)
    model = build_ext_model(ext_cfg("t5", dict(n_experts=4, top_k=2, aux_loss_coef=0.5)))
    idx = torch.randint(0, 30, (2, 16))
    model.eval()
    out_eval = model(idx)
    assert out_eval.grad_fn is None or "AuxLoss" not in type(out_eval.grad_fn).__name__
    with torch.no_grad():
        model.train()
        out = model(idx)          # sin gradiente no se inyecta nada (ni falla)
    assert out.shape == (2, 16, 30)


def test_checkpoint_roundtrip_with_team_utilities(tmp_path):
    """save_checkpoint / load_checkpoint del equipo funcionan con build_ext_model."""
    torch.manual_seed(0)
    cfg = ext_cfg("alibi", dict(n_experts=4, top_k=1, aux_loss_coef=0.01))
    model = build_ext_model(cfg).eval()
    tok = CharTokenizer.from_text("abcdefghijklmnopqrstuvwxyz ,.\n")
    save_checkpoint(tmp_path / "m.pt", model, cfg, tok)
    loaded, cfg2, tok2 = load_checkpoint(tmp_path / "m.pt", model_factory=build_ext_model)
    idx = torch.randint(0, 30, (2, 16))
    with torch.no_grad():
        assert torch.equal(loaded.eval()(idx), model(idx))
    assert cfg2["moe"] == cfg["moe"] and len(moe_layers(loaded)) == 4


def test_team_pretrain_trains_moe_model():
    """El pretrain del equipo, sin modificar, entrena el modelo con MoE: la pérdida baja,
    hay estadísticas de expertos y la pérdida de balanceo llega a los routers."""
    text = "abcde fghij klmno " * 200
    tok = CharTokenizer.from_text(text)
    cfg = make_config(get_small_config(tok.n_vocab, context_length=16, emb_dim=32, n_heads=4,
                                       n_layers=2, drop_rate=0.0), "alibi",
                      moe=dict(n_experts=4, top_k=1, aux_loss_coef=0.01))
    res = pretrain(cfg, text, tok, model_factory=build_ext_model, num_epochs=2, batch_size=8,
                   learning_rate=3e-3, eval_freq=1000, eval_iter=2, start_context="", seed=0, device="cpu")
    model = res["model"]
    assert res["final_val_loss"] < 1.0                    # aprende el texto periódico
    assert len(moe_layers(model)) == 2
    model(torch.tensor([tok.encode("abcde fghij klmno")]))
    stats = moe_expert_stats(model)
    assert stats.shape == (2, 4) and stats.sum().item() == 2 * 17
