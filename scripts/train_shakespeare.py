# train_shakespeare.py
#
# Entrena desde cero un GPT pequeño (base o MoE, según la configuración) sobre
# Tiny Shakespeare a nivel de carácter y guarda métricas en results/<nombre>_seed<s>.json.
#
#   python scripts/train_shakespeare.py --config configs/moe_e4k2.json --seed 1
#
# Configuración (JSON):
#   {"name": ..., "model": {cfg del GPT, con "moe" opcional},
#    "train": {"steps", "batch_size", "lr", "weight_decay", "eval_every", "eval_batches"}}
# vocab_size se rellena con el del corpus. El coeficiente de la pérdida auxiliar va
# en model.moe.aux_loss_coef.

import argparse
import json
import os
import sys
import time
import urllib.request

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from model_base import generate  # noqa: E402
from moe import GPTMoEModel, build_model, calc_loss_batch_moe  # noqa: E402

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(ROOT, "data", "tinyshakespeare.txt")


#####################################
# Datos: Tiny Shakespeare a nivel de carácter
#####################################
def load_text(path=DATA_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Descargando Tiny Shakespeare en {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Un token por carácter distinto del corpus (65 en Tiny Shakespeare)."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def get_batch(data, batch_size, context_length, generator):
    """Trozos contiguos en posiciones aleatorias; el objetivo es la entrada desplazada 1."""
    starts = torch.randint(len(data) - context_length - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s:s + context_length] for s in starts])          # (B, T)
    y = torch.stack([data[s + 1:s + context_length + 1] for s in starts])  # (B, T)
    return x, y


#####################################
# Utilidades de medida
#####################################
def count_params(model):
    """Parámetros totales y activos por token (en un MoE solo cuentan k expertos)."""
    total = sum(p.numel() for p in model.parameters())
    active = total
    if isinstance(model, GPTMoEModel):
        for layer in model.moe_layers():
            per_expert = sum(p.numel() for p in layer.experts[0].parameters())
            active -= (layer.n_experts - layer.top_k) * per_expert
    return total, active


@torch.no_grad()
def evaluate(model, data, cfg, tcfg, aux_coef, seed):
    """Pérdida media sobre eval_batches lotes fijos (misma semilla para todos los
    modelos, para comparar sobre exactamente el mismo texto)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ce_sum, aux_sum = 0.0, 0.0
    for _ in range(tcfg["eval_batches"]):
        x, y = get_batch(data, tcfg["batch_size"], cfg["context_length"], gen)
        _, ce, aux = calc_loss_batch_moe(x, y, model, "cpu", aux_coef)
        ce_sum += ce.item()
        aux_sum += aux.item()
    model.train()
    n = tcfg["eval_batches"]
    return ce_sum / n, aux_sum / n


#####################################
# Entrenamiento
#####################################
def train(config, seed, text=None, steps=None, verbose=True, prompt="ROMEO:"):
    """Entrena según la configuración y devuelve un dict con todos los resultados."""
    text = load_text() if text is None else text
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    cfg = dict(config["model"], vocab_size=tok.vocab_size)
    tcfg = dict(config["train"])
    if steps is not None:
        tcfg["steps"] = steps
    aux_coef = cfg.get("moe", {}).get("aux_loss_coef", 0.0)

    torch.manual_seed(seed)
    model = build_model(cfg)
    total_params, active_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    is_moe = isinstance(model, GPTMoEModel)

    history = []
    usage = None            # asignaciones acumuladas por (capa MoE, experto) desde la última evaluación
    train_ce_acc, train_aux_acc, n_acc = 0.0, 0.0, 0
    train_time, tokens_seen = 0.0, 0

    model.train()
    for step in range(1, tcfg["steps"] + 1):
        x, y = get_batch(train_data, tcfg["batch_size"], cfg["context_length"], gen)

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss, ce, aux = calc_loss_batch_moe(x, y, model, "cpu", aux_coef)
        loss.backward()
        optimizer.step()
        train_time += time.perf_counter() - t0
        tokens_seen += x.numel()

        train_ce_acc += ce.item()
        train_aux_acc += aux.item()
        n_acc += 1
        if is_moe:
            stats = model.expert_stats()
            usage = stats if usage is None else usage + stats

        if step % tcfg["eval_every"] == 0 or step == tcfg["steps"]:
            val_ce, val_aux = evaluate(model, val_data, cfg, tcfg, aux_coef, seed=1234)
            entry = {"step": step, "tokens_seen": tokens_seen,
                     "train_ce": train_ce_acc / n_acc, "train_aux": train_aux_acc / n_acc,
                     "val_ce": val_ce, "val_aux": val_aux}
            if is_moe:
                # Fracción de asignaciones por experto en cada capa: (L_moe, E)
                entry["expert_fraction"] = (usage / usage.sum(dim=1, keepdim=True)).tolist()
            history.append(entry)
            if verbose:
                print(f"step {step:5d} | train CE {entry['train_ce']:.3f} | val CE {val_ce:.3f}"
                      f" | aux {entry['train_aux']:.3f} | {tokens_seen / train_time:,.0f} tok/s")
            train_ce_acc, train_aux_acc, n_acc, usage = 0.0, 0.0, 0, None

    # Muestra de texto generado (semilla fija)
    torch.manual_seed(seed)
    model.eval()
    start = torch.tensor([tok.encode(prompt)])
    sample = tok.decode(generate(model, start, max_new_tokens=200,
                                 context_size=cfg["context_length"],
                                 temperature=0.8, top_k=min(20, tok.vocab_size))[0].tolist())

    return {
        "name": config["name"], "seed": seed, "config": {"model": cfg, "train": tcfg},
        "params_total": total_params, "params_active": active_params,
        "train_time_s": train_time, "train_tokens_per_s": tokens_seen / train_time,
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "final_val_ce": history[-1]["val_ce"], "history": history, "sample": sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="sobrescribe train.steps")
    parser.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    result = train(config, args.seed, steps=args.steps)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{config['name']}_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"Parámetros: {result['params_total']:,} totales, {result['params_active']:,} activos")
    print(f"Val CE final: {result['final_val_ce']:.4f} | guardado en {out_path}")
    print("Muestra:\n" + result["sample"])


if __name__ == "__main__":
    main()
