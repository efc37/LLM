# Validación numérica del MoE contra las implementaciones de referencia de Hugging Face
# transformers (vía 1 del enunciado): misma entrada, mismos pesos, salidas iguales salvo
# error de coma flotante.
#   - Mixtral (MixtralSparseMoeBlock): top-k renormalizado, expertos SwiGLU.
#   - Switch Transformer (SwitchTransformersSparseMLP): top-1 sin renormalizar, que es la
#     configuración E=4 k=1 de nuestros experimentos (con expertos GELU-tanh sin sesgo).
# Se salta si transformers no está instalado.
#   pytest tests/test_moe_hf.py -v
import pytest
import torch

from moe import MoEFeedForward, load_balancing_loss

transformers = pytest.importorskip("transformers")
from transformers import MixtralConfig, SwitchTransformersConfig  # noqa: E402
from transformers.models.mixtral.modeling_mixtral import (  # noqa: E402
    MixtralSparseMoeBlock, load_balancing_loss_func as mixtral_aux_loss)
from transformers.models.switch_transformers.modeling_switch_transformers import (  # noqa: E402
    SwitchTransformersSparseMLP, load_balancing_loss_func as switch_aux_loss, router_z_loss_func)

D, H = 32, 64


def mixtral_pair(n_experts, top_k, dispatch, seed=0):
    """Bloque MoE de Mixtral con pesos aleatorios y nuestro MoEFeedForward con los mismos."""
    torch.manual_seed(seed)
    hf = MixtralSparseMoeBlock(MixtralConfig(
        hidden_size=D, intermediate_size=H, num_local_experts=n_experts,
        num_experts_per_tok=top_k, hidden_act="silu", router_jitter_noise=0.0)).eval()
    with torch.no_grad():
        for p in hf.parameters():           # transformers los crea con torch.empty
            p.normal_(0.0, 0.2)
    ours = MoEFeedForward({"emb_dim": D, "moe": {
        "n_experts": n_experts, "top_k": top_k, "expert_hidden": H, "expert_type": "swiglu",
        "normalize_topk": True, "dispatch": dispatch}}).eval()
    with torch.no_grad():
        ours.router.gate.weight.copy_(hf.gate.weight)                    # (E, D)
        for e, expert in enumerate(ours.experts):
            gate, up = hf.experts.gate_up_proj[e].chunk(2, dim=0)        # (H, D) cada uno
            expert.w_gate.weight.copy_(gate)
            expert.w_up.weight.copy_(up)
            expert.w_down.weight.copy_(hf.experts.down_proj[e])           # (D, H)
    return hf, ours


@pytest.mark.parametrize("dispatch", ["dense", "loop", "sorted"])
@pytest.mark.parametrize("n_experts,top_k", [(4, 2), (8, 2), (8, 3)])
def test_matches_huggingface_mixtral(n_experts, top_k, dispatch):
    hf, ours = mixtral_pair(n_experts, top_k, dispatch)
    x = torch.randn(3, 10, D)
    with torch.no_grad():
        assert torch.allclose(ours(x), hf(x), atol=1e-5)
        logits_hf, _, idx_hf = hf.gate(x.view(-1, D))
    assert torch.allclose(ours.router.last_logits, logits_hf, atol=1e-6)
    assert torch.equal(ours.last_indices, idx_hf)


@pytest.mark.parametrize("n_experts,top_k", [(4, 1), (4, 2), (8, 2)])
def test_aux_loss_matches_huggingface_mixtral(n_experts, top_k):
    """HF normaliza f_i por N (suma k) y nosotros por N·k (suma 1): L_HF = k · L_nuestra.
    Con k=1 coinciden; con nuestra normalización, 1 = equilibrio para cualquier k."""
    _, ours = mixtral_pair(n_experts, top_k, "sorted")
    with torch.no_grad():
        ours(torch.randn(4, 16, D))
        hf_loss = mixtral_aux_loss((ours.router.last_logits,), n_experts, top_k)
    assert torch.allclose(hf_loss, top_k * ours.aux_loss(), atol=1e-6)


def switch_pair(n_experts, seq_len, seed=0):
    """Capa MoE de Switch Transformer y nuestro MoEFeedForward (k=1, sin renormalizar,
    expertos GELU-tanh con sesgo 0) con los mismos pesos. La capacidad por experto se
    fija en seq_len para que no se descarte ningún token."""
    torch.manual_seed(seed)
    hf = SwitchTransformersSparseMLP(SwitchTransformersConfig(
        d_model=D, d_ff=H, num_experts=n_experts, expert_capacity=seq_len, router_bias=False,
        router_jitter_noise=0.0, dense_act_fn="gelu_new", dropout_rate=0.0,
        router_dtype="float32")).eval()
    ours = MoEFeedForward({"emb_dim": D, "moe": {
        "n_experts": n_experts, "top_k": 1, "expert_hidden": H, "normalize_topk": False}}).eval()
    with torch.no_grad():
        ours.router.gate.weight.copy_(hf.router.classifier.weight)
        for e, expert in enumerate(ours.experts):
            hf_expert = hf.experts[f"expert_{e}"]
            expert.layers[0].weight.copy_(hf_expert.wi.weight)
            expert.layers[0].bias.zero_()
            expert.layers[2].weight.copy_(hf_expert.wo.weight)
            expert.layers[2].bias.zero_()
    return hf, ours


@pytest.mark.parametrize("dispatch", ["dense", "loop", "sorted"])
@pytest.mark.parametrize("n_experts", [4, 8])
def test_matches_huggingface_switch_transformer(n_experts, dispatch):
    B, T = 3, 12
    hf, ours = switch_pair(n_experts, T)
    ours.dispatch = dispatch
    x = torch.randn(B, T, D)
    with torch.no_grad():
        assert torch.allclose(ours(x), hf(x), atol=1e-5)


def test_switch_losses_match_huggingface():
    """Balanceo (HF lo calcula por secuencia y promedia) y router z-loss."""
    B, T, E = 4, 12, 4
    hf, ours = switch_pair(E, T)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        ours(x)
        logits = ours.router.last_logits.view(B, T, E)
        probs = ours.last_probs.view(B, T, E)
        indices = ours.last_indices.view(B, T)
        per_sequence = torch.stack([load_balancing_loss(probs[b], indices[b].unsqueeze(-1), E)
                                    for b in range(B)]).mean()
        assert torch.allclose(per_sequence, switch_aux_loss(probs, indices), atol=1e-6)
        assert torch.allclose(ours.z_loss(), router_z_loss_func(logits), atol=1e-6)


def test_switch_negative_control_renormalized_k1_differs():
    """Control negativo: con k=1 renormalizado el peso es 1 (no p_max) y ya NO coincide."""
    hf, ours = switch_pair(4, 12)
    ours.router.normalize_topk = True
    x = torch.randn(3, 12, D)
    with torch.no_grad():
        assert not torch.allclose(ours(x), hf(x), atol=1e-3)
