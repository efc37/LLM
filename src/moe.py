# moe.py
#
# Ampliación: Mixture of Experts (MoE) en la capa feed-forward del GPT del libro.
# Se construye encima de model_base.py sin modificarlo: si la configuración no
# tiene la clave "moe", se usa el modelo base tal cual.
#
# Notación de shapes:
#   B = batch, T = tokens, D = emb_dim, E = número de expertos,
#   k = top_k (expertos activos por token), N = B * T (tokens aplanados)

import torch
import torch.nn as nn


#####################################
# Router top-k
#####################################
class TopKRouter(nn.Module):
    """Decide qué k expertos procesan cada token y con qué peso.

    Devuelve:
      weights: (N, k) pesos de los k expertos elegidos, renormalizados para sumar 1
      indices: (N, k) índices de esos expertos, en [0, E)
      probs:   (N, E) softmax sobre todos los expertos (lo usa la pérdida auxiliar)
    """

    def __init__(self, emb_dim, n_experts, top_k):
        super().__init__()
        assert 1 <= top_k <= n_experts, "top_k debe estar entre 1 y n_experts"
        self.n_experts = n_experts
        self.top_k = top_k
        # Una puntuación por experto; sin sesgo, como en Switch/Mixtral
        self.gate = nn.Linear(emb_dim, n_experts, bias=False)

    def forward(self, x):
        # x: (N, D)
        logits = self.gate(x)                     # (N, E)
        probs = torch.softmax(logits, dim=-1)     # (N, E)

        # Los k expertos con mayor probabilidad para cada token
        top_probs, indices = torch.topk(probs, self.top_k, dim=-1)  # (N, k), (N, k)

        # Renormaliza para que los k pesos sumen 1 (con k=1 el peso es exactamente 1)
        weights = top_probs / top_probs.sum(dim=-1, keepdim=True)   # (N, k)
        return weights, indices, probs
