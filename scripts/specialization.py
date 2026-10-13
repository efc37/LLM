# specialization.py
#
# ¿Se especializan los expertos? Carga los pesos entrenados de un MoE (checkpoints/),
# pasa el texto de validación y cuenta a qué experto va cada carácter, por clases
# (espacio, salto de línea, vocal, consonante, mayúscula, puntuación).
# Resume la especialización de cada capa con la información mutua normalizada
# I(clase; experto) / H(clase), que es 0 si el router ignora la clase y 1 si la
# clase determina el experto.
#
#   python scripts/specialization.py [--name moe_e4k1]
# Guarda results/specialization_<nombre>.json y .md

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from moe import build_model  # noqa: E402
from train_shakespeare import CharTokenizer, load_text  # noqa: E402

CLASSES = ["espacio", "salto de línea", "vocal", "consonante", "mayúscula", "puntuación"]


def char_class(c):
    if c == " ":
        return "espacio"
    if c == "\n":
        return "salto de línea"
    if c.isupper():
        return "mayúscula"
    if c in "aeiou":
        return "vocal"
    if c.isalpha():
        return "consonante"
    return "puntuación"


def mutual_information(counts):
    """counts: matriz (clases, expertos). Devuelve I(clase; experto) / H(clase)."""
    total = counts.sum()
    p = counts / total
    p_class = p.sum(dim=1, keepdim=True)
    p_exp = p.sum(dim=0, keepdim=True)
    nz = p > 0
    mi = (p[nz] * torch.log(p[nz] / (p_class @ p_exp)[nz])).sum().item()
    h_class = -sum(v * math.log(v) for v in p_class.flatten().tolist() if v > 0)
    return mi / h_class


@torch.no_grad()
def analyze(name, seed, text, tok, n_chunks=200):
    with open(os.path.join(ROOT, "results", f"{name}_seed{seed}.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]["model"]
    model = build_model(cfg)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"{name}_seed{seed}.pt")))
    model.eval()

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    val = data[int(0.9 * len(data)):]
    T = cfg["context_length"]
    chunks = val[:n_chunks * T].view(n_chunks, T)            # (n_chunks, T) trozos sin solapar
    model(chunks)

    class_ids = torch.tensor([CLASSES.index(char_class(tok.chars[i])) for i in chunks.flatten()])
    layers = []
    for layer in model.moe_layers():
        expert = layer.last_indices[:, 0]                    # experto principal de cada token (N,)
        counts = torch.zeros(len(CLASSES), layer.n_experts)
        counts.index_put_((class_ids, expert), torch.ones_like(expert, dtype=torch.float), accumulate=True)
        layers.append({"counts": counts.tolist(), "nmi": mutual_information(counts)})
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="moe_e4k1")
    args = parser.parse_args()

    text = load_text()
    tok = CharTokenizer(text)
    results = {seed: analyze(args.name, seed, text, tok) for seed in (1, 2, 3)}

    lines = [f"## Especialización de expertos: {args.name}", "",
             "Información mutua normalizada I(clase; experto)/H(clase) por capa (0 = sin relación, 1 = la clase determina el experto)", "",
             "| Semilla | " + " | ".join(f"Capa {i + 1}" for i in range(len(results[1]))) + " |",
             "|---|" + "---|" * len(results[1])]
    for seed, layers in results.items():
        lines.append(f"| {seed} | " + " | ".join(f"{l['nmi']:.3f}" for l in layers) + " |")

    # Detalle de la semilla 1: qué fracción de cada clase va a cada experto, primera y última capa
    for li in (0, len(results[1]) - 1):
        counts = torch.tensor(results[1][li]["counts"])
        frac = counts / counts.sum(dim=1, keepdim=True)
        lines += ["", f"Semilla 1, capa {li + 1}: fracción de cada clase enviada a cada experto", "",
                  "| Clase (n) | " + " | ".join(f"E{e}" for e in range(counts.shape[1])) + " |",
                  "|---|" + "---|" * counts.shape[1]]
        for ci, cname in enumerate(CLASSES):
            lines.append(f"| {cname} ({int(counts[ci].sum())}) | "
                         + " | ".join(f"{v:.2f}" for v in frac[ci].tolist()) + " |")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "results", f"specialization_{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    with open(os.path.join(ROOT, "results", f"specialization_{args.name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
