# context_ppl_moe.py
#
# Experimento que une las dos ampliaciones: codificación posicional (equipo) x
# feed-forward denso o MoE (nuestro), con el MISMO protocolo que src/context_ppl.py del
# equipo: mismo pretrain, datos, división train/val, semillas, hiperparámetros y
# evaluación de perplejidad frente a la longitud de contexto (se entrena con 128 y se
# evalúa hasta 1024). Lo único que cambia es el model_factory: moe_ext.build_ext_model,
# que con ffn=dense construye exactamente el GPTModelExt del equipo.
#
# Además, para los modelos MoE mide el reparto del router por posición: ¿cambia la
# asignación de expertos en posiciones que el modelo no vio al entrenar (> 128)?
#
#   python scripts/context_ppl_moe.py run --pe learned nope alibi t5 --ffn dense moe_e4k1 --seeds 0 1 2
#   python scripts/context_ppl_moe.py summarize
# `run` es reanudable (salta lo que ya está en results.jsonl) y guarda checkpoints en
# runs/context_moe/ckpt/ (no se versionan).

import argparse
import contextlib
import io
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from context_ppl import _val_tokens, evaluate_ppl  # noqa: E402
from model_base_adaptado import get_small_config, get_tokenizer, load_text, pretrain, save_checkpoint  # noqa: E402
from model_ext import make_config  # noqa: E402
from moe import moe_layers  # noqa: E402
from moe_ext import build_ext_model  # noqa: E402
from pe_labels import check_layers, parse_pe_label  # noqa: E402

# Variantes del feed-forward. dense = el FFN del libro (modelo del equipo sin tocar)
FFN = {
    "dense": None,
    "moe_e4k1": {"n_experts": 4, "top_k": 1, "aux_loss_coef": 0.01},
    "moe_e4k2": {"n_experts": 4, "top_k": 2, "aux_loss_coef": 0.01},
}
OUT_DIR = ROOT / "runs" / "context_moe"


def count_active_params(model):
    """Parámetros totales y activos por token (un MoE solo usa k de sus E expertos)."""
    total = sum(p.numel() for p in model.parameters())
    active = total
    for layer in moe_layers(model):
        per_expert = sum(p.numel() for p in layer.experts[0].parameters())
        active -= (layer.n_experts - layer.top_k) * per_expert
    return total, active


@torch.no_grad()
def routing_by_position(model, tokens, context=1024, bucket=128, n_windows=40):
    """Fracción de asignaciones a cada experto por tramo de posición, en ventanas de
    `context` tokens sin solapar. Devuelve, por capa MoE, una lista (tramos) de
    fracciones (E,) y la distancia de variación total de cada tramo respecto al
    primero (posiciones vistas en entrenamiento): TV = 0.5 * sum |f_tramo - f_0|."""
    model.eval()
    n_windows = min(n_windows, len(tokens) // context)
    windows = tokens[: n_windows * context].view(n_windows, context)
    model(windows)
    layers = []
    for layer in moe_layers(model):
        idx = layer.last_indices.view(n_windows, context, -1)            # (W, T, k)
        fractions = []
        for start in range(0, context, bucket):
            counts = torch.bincount(idx[:, start:start + bucket].reshape(-1), minlength=layer.n_experts)
            fractions.append((counts / counts.sum()).tolist())
        ref = torch.tensor(fractions[0])
        tv = [0.5 * (torch.tensor(f) - ref).abs().sum().item() for f in fractions]
        layers.append({"fractions": fractions, "tv_vs_first": tv})
    return {"context": context, "bucket": bucket, "n_windows": n_windows, "layers": layers}


def run_one(pe_label, ffn, seed, text, tokenizer, tc, ec, device=None, ckpt_dir=None):
    """Como context_ppl.run_one del equipo, con el FFN `ffn` (clave de FFN)."""
    pos_kind, pos_extra = parse_pe_label(pe_label)
    check_layers(pe_label, tc["n_layers"])
    contexts = ec["contexts"]
    base = get_small_config(tokenizer.n_vocab, tc["train_ctx"], tc["emb_dim"], tc["n_heads"],
                            tc["n_layers"], tc["drop_rate"])
    extra = dict(pos_extra)
    if FFN[ffn] is not None:
        extra["moe"] = dict(FFN[ffn])
    cfg = make_config(base, pos_kind, max_positions=max(max(contexts), tc["train_ctx"]), **extra)
    with contextlib.redirect_stdout(io.StringIO()):
        res = pretrain(cfg, text, tokenizer, model_factory=build_ext_model, num_epochs=tc["epochs"],
                       batch_size=tc["batch_size"], learning_rate=tc["lr"], weight_decay=tc["weight_decay"],
                       eval_freq=50, eval_iter=5, start_context="", seed=seed, device=device)
    model = res["model"]
    tokens = _val_tokens(text, tokenizer)
    ppl = evaluate_ppl(model, tokens, contexts, ec["last_k"], ec["max_eval_tokens"], device)
    total, active = count_active_params(model)
    record = {
        "pos_encoding": pe_label, "ffn": ffn, "seed": seed, "train_config": tc, "eval_config": ec,
        "moe": FFN[ffn], "n_params": total, "n_params_active": active,
        "val_loss_train_ctx": res["final_val_loss"], "train_loss_train_ctx": res["final_train_loss"],
        "ppl": {m: {str(c): v for c, v in d.items()} for m, d in ppl.items()},
        "train_time_s": res["train_time_s"], "train_tokens_per_s": res["train_tokens_per_s"],
        "torch_threads": torch.get_num_threads(),
    }
    if moe_layers(model):
        record["routing"] = routing_by_position(model, tokens, context=max(contexts))
    if ckpt_dir:
        save_checkpoint(Path(ckpt_dir) / f"{pe_label}_{ffn}_s{seed}.pt", model, cfg, tokenizer)
    return record


def load_results(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_experiment(pes, ffns, seeds, out_dir, text, tokenizer, tc, ec, device=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    existing = load_results(path)
    for r in existing:
        if r["train_config"] != tc or r["eval_config"] != ec:
            raise ValueError(f"{path} tiene resultados con otra configuración; usa otra carpeta")
    done = {(r["pos_encoding"], r["ffn"], r["seed"]) for r in existing}
    # Se recorre por semilla para tener cuanto antes una semilla completa de todo
    for seed in seeds:
        for pe in pes:
            for ffn in ffns:
                if (pe, ffn, seed) in done:
                    continue
                rec = run_one(pe, ffn, seed, text, tokenizer, tc, ec, device, out_dir / "ckpt")
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                last = rec["ppl"]["last"]
                print(f"{pe:10s} {ffn:9s} seed {seed}: val {rec['val_loss_train_ctx']:.4f} | ppl(last) "
                      + "  ".join(f"{c}:{v:.2f}" if v is not None else f"{c}:-" for c, v in last.items())
                      + f" | {rec['train_time_s']:.0f} s", flush=True)
    return load_results(path)


#####################################
# Resumen
#####################################
def mean_std(values):
    """Media y desviación típica muestral (n-1)."""
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def summarize(results):
    groups = {}
    for r in results:
        groups.setdefault((r["pos_encoding"], r["ffn"]), []).append(r)
    summary = {}
    for (pe, ffn), rs in groups.items():
        contexts = sorted({int(c) for c in rs[0]["ppl"]["last"]})
        entry = {"pos_encoding": pe, "ffn": ffn, "n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs),
                 "n_params": rs[0]["n_params"], "n_params_active": rs[0]["n_params_active"],
                 "val_loss_train_ctx": mean_std([r["val_loss_train_ctx"] for r in rs]),
                 "train_tokens_per_s": mean_std([r["train_tokens_per_s"] for r in rs]),
                 "train_time_s": mean_std([r["train_time_s"] for r in rs]), "ppl": {}}
        for metric in ("last", "full"):
            entry["ppl"][metric] = {}
            for c in contexts:
                vals = [r["ppl"][metric][str(c)] for r in rs if r["ppl"][metric][str(c)] is not None]
                entry["ppl"][metric][c] = mean_std(vals) if vals else None
        if "routing" in rs[0]:
            # TV del último tramo (posiciones 896-1023) frente al primero, peor capa, por semilla
            entry["routing_tv_last_bucket"] = mean_std(
                [max(layer["tv_vs_first"][-1] for layer in r["routing"]["layers"]) for r in rs])
            entry["routing_tv_by_bucket"] = [
                st.mean(max(layer["tv_vs_first"][b] for layer in r["routing"]["layers"]) for r in rs)
                for b in range(len(rs[0]["routing"]["layers"][0]["tv_vs_first"]))]
            entry["max_expert_load_first_bucket"] = mean_std(
                [max(max(layer["fractions"][0]) for layer in r["routing"]["layers"]) for r in rs])
        summary[f"{pe}|{ffn}"] = entry
    return summary


def markdown(summary, metric="last"):
    entries = list(summary.values())
    contexts = sorted({c for e in entries for c in e["ppl"][metric]})
    order_pe = ["learned", "sinusoidal", "nope", "alibi", "t5", "rope", "rope_nope2", "rope_nope4"]
    entries.sort(key=lambda e: (order_pe.index(e["pos_encoding"]) if e["pos_encoding"] in order_pe else 99,
                                list(FFN).index(e["ffn"])))
    lines = [f"## Perplejidad '{metric}' frente al contexto (entrenado con 128), media ± desv. típica (n-1)", "",
             "| Codificación | FFN | Parám. tot./act. | Val CE (ctx 128) | "
             + " | ".join(f"ctx {c}" for c in contexts) + " | tok/s |",
             "|---|---|---|---|" + "---|" * len(contexts) + "---|"]
    for e in entries:
        cells = []
        for c in contexts:
            v = e["ppl"][metric][c]
            cells.append("-" if v is None else f"{v[0]:.2f} ± {v[1]:.2f}")
        lines.append(f"| {e['pos_encoding']} | {e['ffn']} | {e['n_params']:,}/{e['n_params_active']:,} "
                     f"| {e['val_loss_train_ctx'][0]:.4f} ± {e['val_loss_train_ctx'][1]:.4f} | "
                     + " | ".join(cells) + f" | {e['train_tokens_per_s'][0]:,.0f} |")
    routed = [e for e in entries if "routing_tv_last_bucket" in e]
    if routed and metric == "last":
        lines += ["", "## Reparto del router por posición (MoE): distancia de variación total frente a las "
                      "posiciones 0-127, peor capa, media de semillas", "",
                  "| Codificación | FFN | carga máx. (0-127) | " + " | ".join(
                      f"{128 * b}-{128 * b + 127}" for b in range(len(routed[0]['routing_tv_by_bucket']))) + " |",
                  "|---|---|---|" + "---|" * len(routed[0]["routing_tv_by_bucket"])]
        for e in routed:
            lines.append(f"| {e['pos_encoding']} | {e['ffn']} | {e['max_expert_load_first_bucket'][0]:.3f} | "
                         + " | ".join(f"{v:.3f}" for v in e["routing_tv_by_bucket"]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Codificación posicional x FFN denso/MoE")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pe", nargs="+", default=["learned", "nope", "alibi", "t5"])
    r.add_argument("--ffn", nargs="+", default=["dense", "moe_e4k1"], choices=list(FFN))
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--out-dir", default=str(OUT_DIR))
    r.add_argument("--device", default="cpu")
    s = sub.add_parser("summarize")
    s.add_argument("--dir", default=str(OUT_DIR))
    args = p.parse_args(argv)

    if args.command == "run":
        # Mismos valores por defecto que `python src/context_ppl.py run` del equipo
        text = load_text("tinyshakespeare", str(ROOT / "data"))
        tokenizer = get_tokenizer("char", text)
        tc = dict(train_ctx=128, epochs=5, batch_size=32, lr=1e-3, weight_decay=0.1, emb_dim=128, n_heads=4,
                  n_layers=4, drop_rate=0.1, data_id=f"tinyshakespeare|max_chars=None|tok=char|n={len(text)}")
        ec = dict(contexts=[32, 64, 128, 256, 512, 1024], last_k=32, max_eval_tokens=40000)
        print(f"Texto: {len(text):,} caracteres | vocab {tokenizer.n_vocab} | hilos {torch.get_num_threads()}",
              flush=True)
        run_experiment(args.pe, args.ffn, args.seeds, args.out_dir, text, tokenizer, tc, ec, args.device)
        args.dir = args.out_dir

    results = load_results(Path(args.dir) / "results.jsonl")
    if not results:
        raise SystemExit(f"No hay resultados en {args.dir}/results.jsonl")
    summary = summarize(results)
    (Path(args.dir) / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    md = markdown(summary, "last") + "\n" + markdown(summary, "full")
    (Path(args.dir) / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
