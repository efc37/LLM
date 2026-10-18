# router_stats.py
#
# Estado de los routers de los MoE entrenados (checkpoints/), medido sobre texto de
# validación: ¿están saturados? Por capa MoE:
#   z        router z-loss = media de logsumexp(logits)^2 (escala de los logits)
#   conf     confianza media = media de la probabilidad del experto elegido (1/E..1)
#   logit    media del logit máximo
# Sirve para la ablación de la router z-loss (moe_e4k1_z) frente a moe_e4k1.
#
#   python scripts/router_stats.py
# Guarda results/router_stats.json y .md

import json
import os
import statistics as st
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from moe import build_model, moe_layers, router_z_loss  # noqa: E402
from train_shakespeare import CharTokenizer, load_text  # noqa: E402

NAMES = ["moe_e4k1", "moe_e4k1_z", "moe_e4k1_aux0", "moe_e4k1_aux0.1", "moe_e4k2"]
SEEDS = [1, 2, 3]


@torch.no_grad()
def stats_for(name, seed, chunks):
    with open(os.path.join(ROOT, "results", f"{name}_seed{seed}.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]["model"]
    model = build_model(cfg)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"{name}_seed{seed}.pt")))
    model.eval()
    model(chunks)
    out = []
    for layer in moe_layers(model):
        logits = layer.router.last_logits                          # (N, E)
        probs = torch.softmax(logits, dim=-1)
        out.append({"z": router_z_loss(logits).item(),
                    "conf": probs.max(dim=-1).values.mean().item(),
                    "logit": logits.max(dim=-1).values.mean().item()})
    return out


def main():
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    val = data[int(0.9 * len(data)):]
    chunks = val[:200 * 128].view(200, 128)                        # mismos trozos que specialization.py

    results = {}
    for name in NAMES:
        if not all(os.path.exists(os.path.join(ROOT, "checkpoints", f"{name}_seed{s}.pt")) for s in SEEDS):
            print(f"[saltado] {name}: faltan checkpoints")
            continue
        results[name] = {s: stats_for(name, s, chunks) for s in SEEDS}

    lines = ["## Estado de los routers (texto de validación, 200 trozos de 128)", "",
             "Media ± desviación típica (n-1) entre semillas de la media sobre capas; entre corchetes, la peor capa.", "",
             "| Modelo | z-loss | confianza media | logit máx. medio |", "|---|---|---|---|"]
    summary = {}
    for name, per_seed in results.items():
        row = {}
        for key in ("z", "conf", "logit"):
            means = [st.mean(layer[key] for layer in layers) for layers in per_seed.values()]
            worst = st.mean(max(layer[key] for layer in layers) for layers in per_seed.values())
            row[key] = {"mean": st.mean(means), "std": st.stdev(means), "worst_layer": worst}
        summary[name] = row
        lines.append(f"| {name} | " + " | ".join(
            f"{row[k]['mean']:.3f} ± {row[k]['std']:.3f} [{row[k]['worst_layer']:.3f}]"
            for k in ("z", "conf", "logit")) + " |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "results", "router_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"per_seed": results, "summary": summary}, f, indent=1)
    with open(os.path.join(ROOT, "results", "router_stats.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
