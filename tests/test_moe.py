# Tests de la ampliación MoE. Ejecutar con: pytest tests/test_moe.py -v
import pytest
import torch

from model_base import FeedForward
from moe import MoEFeedForward, TopKRouter


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
