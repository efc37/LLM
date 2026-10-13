# Tests de la ampliación MoE. Ejecutar con: pytest tests/test_moe.py -v
import pytest
import torch
import torch.nn as nn

from model_base import FeedForward, GPTModel, TransformerBlock
from moe import (GPTMoEModel, MoEFeedForward, MoETransformerBlock, TopKRouter,
                 build_model, load_balancing_loss, upcycle_from_dense)


def small_cfg(**moe):
    return {"emb_dim": 32, "moe": moe}


@pytest.mark.parametrize("n_experts,top_k", [(1, 1), (4, 1), (4, 2), (8, 2)])
def test_router_shapes(n_experts, top_k):
    torch.manual_seed(0)
    N, D = 24, 16
    router = TopKRouter(D, n_experts, top_k)
    weights, indices, probs = router(torch.randn(N, D))

    assert weights.shape == (N, top_k)
    assert indices.shape == (N, top_k)
    assert probs.shape == (N, n_experts)

    # Las probabilidades suman 1 por token; los pesos, solo si se renormalizan (k>1).
    # Con k=1 (Switch) el peso es la probabilidad del experto elegido.
    assert torch.allclose(probs.sum(-1), torch.ones(N))
    if router.normalize_topk:
        assert torch.allclose(weights.sum(-1), torch.ones(N))
    else:
        assert torch.allclose(weights, probs.gather(1, indices))
    # Índices válidos y sin repetir experto dentro de un mismo token
    assert indices.min() >= 0 and indices.max() < n_experts
    assert all(len(set(row.tolist())) == top_k for row in indices)
    # Los elegidos son realmente los de mayor probabilidad
    kth_best = probs.sort(dim=-1, descending=True).values[:, top_k - 1]
    assert torch.all(probs.gather(1, indices) >= kth_best.unsqueeze(1))


def test_router_gradient_flows():
    """El router se entrena a través de los pesos (topk no es derivable en los índices)."""
    torch.manual_seed(0)
    router = TopKRouter(16, 4, 2)
    weights, _, _ = router(torch.randn(8, 16))
    (weights * torch.randn_like(weights)).sum().backward()
    assert router.gate.weight.grad is not None
    assert router.gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("normalize", [False, True])
def test_router_k1_gradient_depends_on_normalization(normalize):
    """Con k=1, renormalizar deja el peso en 1 constante: el router no recibe gradiente
    de la pérdida principal. Sin renormalizar (Switch) sí lo recibe."""
    torch.manual_seed(0)
    router = TopKRouter(16, 4, 1, normalize_topk=normalize)
    weights, _, _ = router(torch.randn(8, 16))
    (weights * torch.randn_like(weights)).sum().backward()
    grad_norm = router.gate.weight.grad.abs().sum().item()
    if normalize:
        # p/p = 1 salvo redondeo en float32 (~1e-6): gradiente nulo en la práctica
        assert grad_norm < 1e-4
    else:
        assert grad_norm > 1e-2


@pytest.mark.parametrize("n_experts,top_k", [(1, 1), (4, 1), (4, 2), (8, 2)])
def test_moe_output_shape(n_experts, top_k):
    torch.manual_seed(0)
    moe = MoEFeedForward(small_cfg(n_experts=n_experts, top_k=top_k))
    x = torch.randn(3, 10, 32)
    assert moe(x).shape == x.shape
    assert moe.last_probs.shape == (30, n_experts)
    assert moe.last_indices.shape == (30, top_k)


@pytest.mark.parametrize("normalize", [None, False, True])
def test_moe_single_expert_equals_dense_ffn(normalize):
    """E=1, k=1 con los pesos del FeedForward del libro ⇒ misma salida que el FFN denso."""
    torch.manual_seed(123)
    cfg = small_cfg(n_experts=1, top_k=1, normalize_topk=normalize)
    ffn = FeedForward(cfg)
    moe = MoEFeedForward(cfg)
    moe.experts[0].load_state_dict(ffn.state_dict())

    x = torch.randn(4, 16, 32)
    assert torch.allclose(moe(x), ffn(x), atol=1e-6)


def test_moe_all_experts_is_softmax_mixture():
    """Con k=E (y renormalización) la capa es la mezcla blanda sum_e p_e(x) * E_e(x),
    calculada aquí token a token sin pasar por el código de despacho."""
    torch.manual_seed(0)
    moe = MoEFeedForward(small_cfg(n_experts=4, top_k=4, normalize_topk=True))
    x = torch.randn(2, 5, 32)
    out = moe(x).reshape(-1, 32)

    x_flat = x.reshape(-1, 32)
    probs = torch.softmax(moe.router.gate(x_flat), dim=-1)
    for n in range(x_flat.shape[0]):
        expected = sum(probs[n, e] * moe.experts[e](x_flat[n]) for e in range(4))
        assert torch.allclose(out[n], expected, atol=1e-5)


def _forward_backward(moe, x, dispatch):
    """Ejecuta forward+backward con un despacho dado y devuelve salida y gradientes."""
    moe.zero_grad()
    x = x.clone().requires_grad_(True)
    moe.dispatch = dispatch
    out = moe(x)
    # Pérdida arbitraria pero no simétrica para que los gradientes sean informativos
    (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().backward()
    grads = {name: p.grad.clone() for name, p in moe.named_parameters() if p.grad is not None}
    return out.detach(), x.grad.clone(), grads


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_experts,top_k,normalize", [(4, 1, None), (8, 2, None), (8, 2, False), (4, 4, True)])
@pytest.mark.parametrize("dispatch", ["loop", "sorted"])
def test_dispatch_matches_dense(dispatch, n_experts, top_k, normalize, seed):
    """Salida, gradiente respecto a la entrada y gradiente de cada parámetro iguales
    a los de la referencia densa."""
    torch.manual_seed(seed)
    moe = MoEFeedForward(small_cfg(n_experts=n_experts, top_k=top_k, normalize_topk=normalize))
    x = torch.randn(4, 16, 32)

    out_ref, gx_ref, g_ref = _forward_backward(moe, x, "dense")
    out, gx, g = _forward_backward(moe, x, dispatch)

    assert torch.allclose(out, out_ref, atol=1e-5)
    assert torch.allclose(gx, gx_ref, atol=1e-5)
    # En la densa todos los expertos tienen gradiente (aunque sea 0 si nadie los elige)
    for name, grad_ref in g_ref.items():
        grad = g.get(name, torch.zeros_like(grad_ref))
        assert torch.allclose(grad, grad_ref, atol=1e-5), name


@pytest.mark.parametrize("dispatch", ["loop", "sorted"])
def test_dispatch_with_unused_experts(dispatch):
    """Pocos tokens y muchos expertos: algunos expertos no reciben ningún token."""
    torch.manual_seed(0)
    moe = MoEFeedForward(small_cfg(n_experts=16, top_k=1))
    x = torch.randn(1, 3, 32)
    moe.dispatch = "dense"
    ref = moe(x)
    moe.dispatch = dispatch
    assert torch.allclose(moe(x), ref, atol=1e-6)
    assert len(moe.last_indices.unique()) < 16


@pytest.mark.parametrize("n_experts,top_k", [(4, 1), (8, 2)])
def test_aux_loss_perfect_balance_is_one(n_experts, top_k):
    """Cada experto recibe las mismas asignaciones y la misma probabilidad media ⇒ 1."""
    N = 4 * n_experts
    # Token n elige los expertos n, n+1, ..., n+k-1 (mod E): f_i = 1/E
    indices = (torch.arange(N).unsqueeze(1) + torch.arange(top_k)) % n_experts  # (N, k)
    probs = torch.zeros(N, n_experts).scatter(1, indices, 1.0 / top_k)          # P_i = 1/E
    assert torch.isclose(load_balancing_loss(probs, indices, n_experts), torch.tensor(1.0))


@pytest.mark.parametrize("n_experts", [2, 4, 8])
def test_aux_loss_collapse_is_n_experts(n_experts):
    """Todos los tokens al experto 0 con probabilidad 1 ⇒ E (el máximo)."""
    N = 32
    indices = torch.zeros(N, 1, dtype=torch.long)
    probs = torch.zeros(N, n_experts)
    probs[:, 0] = 1.0
    assert torch.isclose(load_balancing_loss(probs, indices, n_experts),
                         torch.tensor(float(n_experts)))


def test_aux_loss_uniform_router_is_one():
    """Router con probabilidades uniformes (pesos a 0): P_i = 1/E ⇒ L = sum f_i = 1,
    aunque topk desempate eligiendo siempre los mismos expertos."""
    moe = MoEFeedForward(small_cfg(n_experts=8, top_k=2))
    nn.init.zeros_(moe.router.gate.weight)
    moe(torch.randn(2, 10, 32))
    assert torch.isclose(moe.aux_loss(), torch.tensor(1.0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aux_loss_gradient_rebalances_router(seed):
    """Router sesgado hacia el experto 0; optimizar SOLO la pérdida auxiliar debe
    repartir la carga (la fracción del experto más usado baja hacia 1/E = 0.25).

    Nota: con sesgos mucho mayores el softmax se satura (P_0 = 1), su gradiente
    desaparece y la pérdida auxiliar ya no puede recuperar al router; con lr
    grandes la carga salta en bloque a otro experto. Por eso sesgo y lr moderados.
    """
    torch.manual_seed(seed)
    E = 4
    moe = MoEFeedForward(small_cfg(n_experts=E, top_k=1))
    with torch.no_grad():
        moe.router.gate.weight[0] += 0.1
    # Entrada con media 1: logit_0 recibe un extra de ~0.1 * D = 3.2 ⇒ casi todo al experto 0
    x = torch.randn(8, 32, 32) + 1.0

    def max_fraction():
        with torch.no_grad():
            moe(x)
        return torch.bincount(moe.last_indices.reshape(-1), minlength=E).max().item() / moe.last_indices.numel()

    before = max_fraction()
    opt = torch.optim.SGD(moe.router.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        moe(x)
        moe.aux_loss().backward()
        assert moe.router.gate.weight.grad.abs().sum() > 0
        opt.step()
    after = max_fraction()

    assert before > 0.9
    assert after < 0.35


def tiny_gpt_cfg(**moe):
    cfg = {"vocab_size": 100, "context_length": 16, "emb_dim": 32, "n_heads": 4,
           "n_layers": 4, "drop_rate": 0.0, "qkv_bias": False}
    if moe:
        cfg["moe"] = moe
    return cfg


def test_build_model_without_moe_is_base():
    model = build_model(tiny_gpt_cfg())
    assert type(model) is GPTModel


@pytest.mark.parametrize("moe", [
    dict(n_experts=1, top_k=1),                                # 1 experto ≡ FFN
    dict(n_experts=4, top_k=2),                                # expertos idénticos, pesos suman 1
    dict(n_experts=4, top_k=1, normalize_topk=True),
    dict(n_experts=4, top_k=2, every_n_layers=2, dispatch="loop"),
])
def test_gpt_moe_upcycled_equals_base(moe):
    """Prueba de extremo a extremo: GPTMoEModel inicializado desde un GPTModel
    (cada experto = el FFN denso) da los mismos logits que el modelo base."""
    torch.manual_seed(123)
    base = GPTModel(tiny_gpt_cfg()).eval()
    model = upcycle_from_dense(build_model(tiny_gpt_cfg(**moe)), base).eval()
    assert type(model) is GPTMoEModel

    idx = torch.randint(0, 100, (3, 16))
    with torch.no_grad():
        assert torch.allclose(model(idx), base(idx), atol=1e-5)


def test_gpt_moe_k1_switch_is_not_dense():
    """Control negativo: con k=1 sin renormalizar la salida es p·FFN(x) ≠ FFN(x)."""
    torch.manual_seed(123)
    base = GPTModel(tiny_gpt_cfg()).eval()
    model = upcycle_from_dense(build_model(tiny_gpt_cfg(n_experts=4, top_k=1)), base).eval()
    idx = torch.randint(0, 100, (3, 16))
    with torch.no_grad():
        assert not torch.allclose(model(idx), base(idx), atol=1e-3)


def test_every_n_layers_alternates_blocks():
    model = build_model(tiny_gpt_cfg(n_experts=4, top_k=2, every_n_layers=2))
    kinds = [type(blk) for blk in model.trf_blocks]
    assert kinds == [TransformerBlock, MoETransformerBlock, TransformerBlock, MoETransformerBlock]
    assert len(model.moe_layers()) == 2


def test_gpt_moe_aux_loss_and_stats():
    torch.manual_seed(0)
    model = build_model(tiny_gpt_cfg(n_experts=4, top_k=2))
    idx = torch.randint(0, 100, (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, 100)

    stats = model.expert_stats()
    assert stats.shape == (4, 4)                      # (capas MoE, expertos)
    assert torch.all(stats.sum(dim=1) == 2 * 16 * 2)  # N·k asignaciones por capa

    aux = model.aux_loss()
    assert aux.dim() == 0 and 1.0 <= aux.item() <= 4.0
    # El gradiente de (CE + coef·L_aux) llega al router de todas las capas
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), idx.flatten()) + 0.01 * aux
    loss.backward()
    for layer in model.moe_layers():
        assert layer.router.gate.weight.grad.abs().sum() > 0
