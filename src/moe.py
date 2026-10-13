# moe.py
#
# Ampliación: Mixture of Experts (MoE) en la capa feed-forward del GPT del libro.
# Se construye encima de model_base.py sin modificarlo: si la configuración no
# tiene la clave "moe", se usa el modelo base tal cual.
#
# Notación de shapes:
#   B = batch, T = tokens, D = emb_dim, E = número de expertos,
#   k = top_k (expertos activos por token), N = B * T (tokens aplanados)

import re

import torch
import torch.nn as nn

from model_base import GELU, GPTModel, TransformerBlock


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
# Pérdida auxiliar de balanceo de carga (Switch Transformer, Fedus et al. 2022)
#####################################
def load_balancing_loss(probs, indices, n_experts):
    """L_aux = E * sum_i f_i * P_i

      f_i: fracción de las N·k asignaciones que van al experto i (no derivable,
           sale de topk)
      P_i: probabilidad media que el router da al experto i (derivable)

    Vale 1 si el reparto es uniforme (f_i = P_i = 1/E) y E si todo va a un
    experto con probabilidad 1. Minimizarla empuja al router a bajar la
    probabilidad de los expertos sobrecargados.
    """
    # probs: (N, E), indices: (N, k)
    counts = torch.bincount(indices.reshape(-1), minlength=n_experts)  # (E,)
    f = counts.to(probs.dtype) / indices.numel()                       # (E,)
    P = probs.mean(dim=0)                                              # (E,)
    return n_experts * torch.sum(f * P)


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
    normalize_topk (por defecto k>1) y dispatch:
      "dense"  referencia: todos los expertos con todos los tokens (coste E·FFN)
      "loop"   por experto, selecciona con una máscara los tokens que lo eligieron
      "sorted" ordena las asignaciones por experto y procesa lotes contiguos (por defecto)
    Los tres dan el mismo resultado; solo cambia el coste.
    """

    def __init__(self, cfg):
        super().__init__()
        moe_cfg = cfg["moe"]
        emb_dim = cfg["emb_dim"]
        self.n_experts = moe_cfg["n_experts"]
        self.top_k = moe_cfg["top_k"]
        self.dispatch = moe_cfg.get("dispatch", "sorted")
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

        if self.dispatch == "sorted":
            out = self._dispatch_sorted(x_flat, weights, indices)
        elif self.dispatch == "loop":
            out = self._dispatch_loop(x_flat, weights, indices)
        elif self.dispatch == "dense":
            out = self._dispatch_dense(x_flat, weights, indices)
        else:
            raise ValueError(f"dispatch desconocido: {self.dispatch}")

        return out.view(b, num_tokens, emb_dim)         # (B, T, D)

    def aux_loss(self):
        """Pérdida de balanceo del último forward."""
        return load_balancing_loss(self.last_probs, self.last_indices, self.n_experts)

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

    def _dispatch_loop(self, x_flat, weights, indices):
        """Para cada experto, busca los tokens que lo han elegido (en cualquiera de
        sus k posiciones), los procesa juntos y suma el resultado ponderado."""
        out = torch.zeros_like(x_flat)                            # (N, D)
        for e, expert in enumerate(self.experts):
            # token_idx: qué tokens eligieron e; slot: en qué posición del top-k
            token_idx, slot = torch.where(indices == e)           # (n_e,), (n_e,)
            if token_idx.numel() == 0:
                continue
            y = expert(x_flat[token_idx])                         # (n_e, D)
            y = y * weights[token_idx, slot].unsqueeze(-1)        # (n_e, D)
            out.index_add_(0, token_idx, y)                       # suma en las filas de esos tokens
        return out

    def _dispatch_sorted(self, x_flat, weights, indices):
        """Despacho eficiente: una sola ordenación agrupa las N·k asignaciones
        (token, experto) por experto, cada experto procesa un bloque contiguo y
        index_add_ devuelve cada resultado ponderado a la fila de su token."""
        n_tokens, k = indices.shape

        # Aplana las asignaciones: la asignación i es (token_ids[i] -> flat_experts[i])
        flat_experts = indices.reshape(-1)                        # (N·k,)
        flat_weights = weights.reshape(-1)                        # (N·k,)
        token_ids = torch.arange(n_tokens, device=x_flat.device).repeat_interleave(k)  # (N·k,)

        # Ordena por experto: las asignaciones de un mismo experto quedan contiguas
        order = torch.argsort(flat_experts, stable=True)          # (N·k,)
        sorted_tokens = token_ids[order]                          # (N·k,)
        sorted_weights = flat_weights[order]                      # (N·k,)
        # Cuántas asignaciones recibe cada experto (tamaño de cada bloque)
        counts = torch.bincount(flat_experts, minlength=self.n_experts)  # (E,)

        # Copia cada token tantas veces como expertos lo procesan, ya agrupado
        x_sorted = x_flat[sorted_tokens]                          # (N·k, D)
        chunks = x_sorted.split(counts.tolist())                  # E bloques (n_e, D)
        y_sorted = torch.cat([expert(chunk) for expert, chunk in zip(self.experts, chunks)
                              if chunk.shape[0] > 0])             # (N·k, D)
        y_sorted = y_sorted * sorted_weights.unsqueeze(-1)        # (N·k, D)

        # Devuelve cada fila a su token; los k resultados de un token se suman
        out = torch.zeros_like(x_flat)                            # (N, D)
        out.index_add_(0, sorted_tokens, y_sorted)
        return out


#####################################
# Integración en el GPT del libro
#####################################
class MoETransformerBlock(TransformerBlock):
    """Bloque del libro con el FeedForward denso sustituido por MoEFeedForward.
    El forward (pre-norm, residuales, dropout) se hereda sin cambios."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ff = MoEFeedForward(cfg)


class GPTMoEModel(GPTModel):
    """GPTModel con bloques MoE. Con cfg["moe"]["every_n_layers"] = n, solo uno de
    cada n bloques es MoE (el n-ésimo, como en Switch/GShard); el resto, densos."""

    def __init__(self, cfg):
        super().__init__(cfg)
        every = cfg["moe"].get("every_n_layers", 1)
        self.trf_blocks = nn.Sequential(*[
            MoETransformerBlock(cfg) if (i + 1) % every == 0 else TransformerBlock(cfg)
            for i in range(cfg["n_layers"])])

    def moe_layers(self):
        return [blk.ff for blk in self.trf_blocks if isinstance(blk.ff, MoEFeedForward)]

    def aux_loss(self):
        """Media de L_aux sobre las capas MoE (del último forward). Se promedia para
        que el coeficiente no dependa del número de capas y 1 siga significando
        'equilibrado'."""
        return torch.stack([layer.aux_loss() for layer in self.moe_layers()]).mean()

    def expert_stats(self):
        """Asignaciones por experto en cada capa MoE del último forward: (L_moe, E)."""
        return torch.stack([
            torch.bincount(layer.last_indices.reshape(-1), minlength=layer.n_experts)
            for layer in self.moe_layers()])


def build_model(cfg):
    """Modelo base si la configuración no tiene "moe"; GPTMoEModel si la tiene."""
    return GPTMoEModel(cfg) if "moe" in cfg else GPTModel(cfg)


def upcycle_from_dense(moe_model, dense_model):
    """Sparse upcycling: copia todos los pesos de un GPTModel denso y replica su
    FeedForward en cada experto. El router conserva su inicialización.

    Si todos los expertos son iguales y los k pesos suman 1, la salida es la del
    modelo denso: sum_j w_j * FFN(x) = FFN(x). Requiere expert_hidden = 4*emb_dim.
    """
    dense_sd = dense_model.state_dict()
    new_sd = {}
    for key, value in moe_model.state_dict().items():
        if ".ff.router." in key:
            new_sd[key] = value
        else:
            # trf_blocks.i.ff.experts.j.layers.X -> trf_blocks.i.ff.layers.X
            new_sd[key] = dense_sd[re.sub(r"\.ff\.experts\.\d+\.", ".ff.", key)].clone()
    moe_model.load_state_dict(new_sd)
    return moe_model
