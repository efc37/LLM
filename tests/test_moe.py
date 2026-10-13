# Tests de la ampliación MoE. Ejecutar con: pytest tests/test_moe.py -v
import pytest
import torch

from moe import TopKRouter


@pytest.mark.parametrize("n_experts,top_k", [(1, 1), (4, 1), (4, 2), (8, 2)])
def test_router_shapes(n_experts, top_k):
    torch.manual_seed(0)
    N, D = 24, 16
    router = TopKRouter(D, n_experts, top_k)
    weights, indices, probs = router(torch.randn(N, D))

    assert weights.shape == (N, top_k)
    assert indices.shape == (N, top_k)
    assert probs.shape == (N, n_experts)

    # Pesos renormalizados y probabilidades suman 1 por token
    assert torch.allclose(weights.sum(-1), torch.ones(N))
    assert torch.allclose(probs.sum(-1), torch.ones(N))
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
