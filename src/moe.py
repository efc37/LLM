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

from model_base import GELU


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

    def __init__(self, emb_dim, n_experts, top_k, normalize_topk=None):
        super().__init__()
        assert 1 <= top_k <= n_experts, "top_k debe estar entre 1 y n_experts"
        self.n_experts = n_experts
        self.top_k = top_k
        # Renormalizar los k pesos (Mixtral) o usar la probabilidad tal cual (Switch).
        # Con k=1 renormalizar daría peso 1 constante y el router no recibiría
        # gradiente de la pérdida principal, así que por defecto solo se hace si k>1.
        self.normalize_topk = (top_k > 1) if normalize_topk is None else normalize_topk
        # Una puntuación por experto; sin sesgo, como en Switch/Mixtral
        self.gate = nn.Linear(emb_dim, n_experts, bias=False)

    def forward(self, x):
        # x: (N, D)
        logits = self.gate(x)                     # (N, E)
        probs = torch.softmax(logits, dim=-1)     # (N, E)

        # Los k expertos con mayor probabilidad para cada token
        top_probs, indices = torch.topk(probs, self.top_k, dim=-1)  # (N, k), (N, k)

        if self.normalize_topk:
            # Los k pesos pasan a sumar 1 (con k=1 el peso sería exactamente 1)
            weights = top_probs / top_probs.sum(dim=-1, keepdim=True)  # (N, k)
        else:
            weights = top_probs                                        # (N, k)
        return weights, indices, probs


#####################################
# Expertos y capa MoE
#####################################
class Expert(nn.Module):
    """Igual que FeedForward del libro, pero con dimensión oculta configurable.

    Mantiene el mismo nombre de submódulos (self.layers) para que los pesos de
    un FeedForward se puedan copiar con load_state_dict.
    """

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class MoEFeedForward(nn.Module):
    """Sustituye al FeedForward denso: cada token pasa solo por k de los E expertos.

    salida(x) = sum_{j=1..k} weights_j(x) * Experto_{indices_j(x)}(x)

    cfg["moe"] admite: n_experts, top_k, expert_hidden (por defecto 4*emb_dim),
    normalize_topk (por defecto k>1) y dispatch ("dense").
    """

    def __init__(self, cfg):
        super().__init__()
        moe_cfg = cfg["moe"]
        emb_dim = cfg["emb_dim"]
        self.n_experts = moe_cfg["n_experts"]
        self.top_k = moe_cfg["top_k"]
        self.dispatch = moe_cfg.get("dispatch", "dense")
        hidden = moe_cfg.get("expert_hidden", 4 * emb_dim)

        self.router = TopKRouter(emb_dim, self.n_experts, self.top_k,
                                 moe_cfg.get("normalize_topk"))
        self.experts = nn.ModuleList(
            [Expert(emb_dim, hidden) for _ in range(self.n_experts)])

        # Se rellenan en cada forward; los usan la pérdida auxiliar y las estadísticas
        self.last_probs = None    # (N, E)
        self.last_indices = None  # (N, k)

    def forward(self, x):
        b, num_tokens, emb_dim = x.shape
        x_flat = x.reshape(-1, emb_dim)                 # (N, D)

        weights, indices, probs = self.router(x_flat)   # (N, k), (N, k), (N, E)
        self.last_probs, self.last_indices = probs, indices

        if self.dispatch == "dense":
            out = self._dispatch_dense(x_flat, weights, indices)
        else:
            raise ValueError(f"dispatch desconocido: {self.dispatch}")

        return out.view(b, num_tokens, emb_dim)         # (B, T, D)

    def _dispatch_dense(self, x_flat, weights, indices):
        """Referencia: todos los expertos procesan todos los tokens (coste E veces el
        de un FFN) y se combinan con una matriz de pesos que es 0 fuera del top-k.
        Es lenta pero obviamente correcta; sirve para validar el despacho eficiente."""
        # Salida de cada experto para cada token: (N, E, D)
        all_out = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        # Matriz de puertas dispersa: weights colocados en sus columnas, 0 en el resto
        gates = torch.zeros(x_flat.shape[0], self.n_experts,
                            dtype=x_flat.dtype, device=x_flat.device)
        gates = gates.scatter(1, indices, weights)       # (N, E)
        # Suma ponderada sobre expertos: (N, E, 1) * (N, E, D) -> (N, D)
        return (gates.unsqueeze(-1) * all_out).sum(dim=1)
