# router_position_probe.py
#
# ¿Responde el router del MoE a la posición? Para cada modelo MoE del experimento
# combinado (runs/context_moe/ckpt/) y cada capa MoE: se pasan ventanas de 1024 tokens
# de validación y se mide, para el experto que domina en las posiciones lejanas
# (512-1023):
#   corr       correlación de Pearson entre la posición (0..1023) y su logit medio
#   dlogit     logit medio en 896-1023 menos logit medio en 0-127 (posiciones vistas)
#   p_first / p_last   probabilidad media del experto en 0-127 y en 896-1023
#   top1_first / top1_last   fracción de tokens que lo eligen (top-1) en esos tramos
# Se informa la capa con mayor cambio de reparto (top1_last - top1_first).
#
#   python scripts/router_position_probe.py
# Guarda runs/context_moe/router_probe.json y .md

import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_base_adaptado import get_tokenizer, load_checkpoint, load_text  # noqa: E402
from moe import moe_layers  # noqa: E402
from moe_ext import build_ext_model  # noqa: E402

RUN_DIR = ROOT / "runs" / "context_moe"
PES = ["learned", "nope", "alibi", "t5"]
SEEDS = [0, 1, 2]
W, T, BUCKET = 20, 1024, 128


@torch.no_grad()
def probe(path, tokens):
    model, _, _ = load_checkpoint(path, model_factory=build_ext_model)
    model.eval()
    model(tokens[: W * T].view(W, T))
    pos = torch.arange(T, dtype=torch.float)
    layers = []
    for layer in moe_layers(model):
        logits = layer.router.last_logits.view(W, T, -1)
        probs = torch.softmax(logits, dim=-1)
        top1 = layer.last_indices.view(W, T, -1)[..., 0]
        dom = probs[:, T // 2:].mean(dim=(0, 1)).argmax().item()
        mean_logit = logits[..., dom].mean(0)                          # (T,)
        first, last = slice(0, BUCKET), slice(T - BUCKET, T)
        layers.append({
            "expert": dom,
            "corr": torch.corrcoef(torch.stack([pos, mean_logit]))[0, 1].item(),
            "dlogit": (mean_logit[last].mean() - mean_logit[first].mean()).item(),
            "p_first": probs[:, first, dom].mean().item(), "p_last": probs[:, last, dom].mean().item(),
            "top1_first": (top1[:, first] == dom).float().mean().item(),
            "top1_last": (top1[:, last] == dom).float().mean().item(),
        })
    return layers


def main():
    text = load_text("tinyshakespeare", str(ROOT / "data"))
    tok = get_tokenizer("char", text)
    tokens = torch.tensor(tok.encode(text[int(0.9 * len(text)):]), dtype=torch.long)
    results = {}
    for pe in PES:
        for seed in SEEDS:
            path = RUN_DIR / "ckpt" / f"{pe}_moe_e4k1_s{seed}.pt"
            if path.exists():
                results.setdefault(pe, {})[seed] = probe(path, tokens)

    lines = ["## ¿Responde el router a la posición? (MoE E=4 k=1, ventanas de 1024, 20 ventanas de validación)", "",
             "Capa con mayor cambio de reparto; experto dominante en 512-1023. Media ± desv. típica entre semillas.", "",
             "| Codificación | capa (por semilla) | corr(posición, logit) | Δ logit (896-1023 − 0-127) | prob. 0-127 → 896-1023 | top-1 0-127 → 896-1023 |",
             "|---|---|---|---|---|---|"]
    summary = {}
    for pe, per_seed in results.items():
        worst = [max(layers, key=lambda l: l["top1_last"] - l["top1_first"]) for layers in per_seed.values()]
        idx = [max(range(len(layers)), key=lambda i: layers[i]["top1_last"] - layers[i]["top1_first"]) + 1
               for layers in per_seed.values()]

        def ms(key):
            vals = [w[key] for w in worst]
            return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

        summary[pe] = {k: ms(k) for k in ("corr", "dlogit", "p_first", "p_last", "top1_first", "top1_last")}
        summary[pe]["layers"] = idx
        s = summary[pe]
        lines.append(f"| {pe} | {idx} | {s['corr'][0]:.2f} ± {s['corr'][1]:.2f} | {s['dlogit'][0]:+.2f} ± {s['dlogit'][1]:.2f} "
                     f"| {s['p_first'][0]:.2f} → {s['p_last'][0]:.2f} | {s['top1_first'][0]:.2f} → {s['top1_last'][0]:.2f} |")
    md = "\n".join(lines) + "\n"
    (RUN_DIR / "router_probe.json").write_text(json.dumps({"per_seed": results, "summary": summary}, indent=1),
                                               encoding="utf-8")
    (RUN_DIR / "router_probe.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
