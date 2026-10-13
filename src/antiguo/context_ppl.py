# context_ppl.py
#
# Experimento complementario: perplejidad frente a longitud de contexto. Cada
# codificación posicional se entrena como modelo de lenguaje con contexto corto
# (train_ctx, por defecto 128) sobre Tiny Shakespeare y se evalúa en el conjunto de
# validación con contextos más largos (por defecto hasta 512 = 4 x train_ctx), que no
# ha visto en entrenamiento. Es la versión a pequeña escala del experimento con modelos
# de lenguaje del artículo (Kazemnejad et al., 2023, arXiv:2305.19466).
#
# Dos métricas (se calculan las dos):
#   "last"  perplejidad solo de los últimos k tokens de cada ventana (k = last_k, por
#           defecto 32; las ventanas avanzan k tokens, así que se puntúan todos los
#           tokens una vez). Cada token puntuado tiene unos `contexto` tokens previos,
#           de modo que mide limpiamente el efecto de la longitud del contexto. ES LA
#           MÉTRICA PRINCIPAL.
#   "full"  perplejidad de todos los tokens de ventanas sin solapar de longitud
#           `contexto`. Mezcla tokens con poco y mucho contexto (los primeros de cada
#           ventana casi no tienen), por lo que atenúa las diferencias.
#
# Mismo flujo para todas las variantes: model_base.pretrain con GPTModelExt, misma
# semilla, mismos datos e hiperparámetros. La tabla de la APE aprendida se dimensiona
# al contexto máximo de evaluación (sus posiciones no entrenadas conservan su
# inicialización: es el baseline).
#
# Uso:
#   python context_ppl.py selftest
#   python context_ppl.py run --pe learned sinusoidal nope rope alibi t5 --seeds 0 --epochs 5 \
#                             --out-dir runs/context_ppl
#   python context_ppl.py summarize --dir runs/context_ppl
#   python context_ppl.py reeval --out-dir runs/context_ppl --contexts 64 128 256 512 1024
# `run` es reanudable y guarda un checkpoint por ejecución (para `reeval`, que recalcula la
# perplejidad con otros contextos sin volver a entrenar). Si cambia algún hiperparámetro
# de entrenamiento, usa otra carpeta: `run` lo detecta y se niega a mezclar resultados.

import argparse
import contextlib
import io
import json
import math
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model_base_adaptado import (get_device, get_small_config, get_tokenizer, load_checkpoint, load_text,
                        pretrain, save_checkpoint)
from model_ext import GPTModelExt, make_config
from positional import PE_KINDS


#####################################
# Perplejidad por ventanas
#####################################
@torch.no_grad()
def window_ppl(model, tokens, context, score_last, stride, device, tokens_per_batch=16384):
    """Perplejidad sobre ventanas de `context` tokens que avanzan `stride` tokens, puntuando
    solo los últimos `score_last` tokens de cada ventana. tokens: tensor 1-D de ids."""
    model.eval()
    score_last = min(score_last, context)
    if len(tokens) < context + 1:
        raise ValueError(f"Hay {len(tokens)} tokens de evaluación; no caben ventanas de contexto {context}")
    windows = tokens.unfold(0, context + 1, stride)  # (num_ventanas, context + 1)
    batch = max(1, tokens_per_batch // context)
    nll, count = 0.0, 0
    for i in range(0, len(windows), batch):
        w = windows[i:i + batch].to(device)
        logits = model(w[:, :-1])[:, -score_last:]
        targets = w[:, 1:][:, -score_last:]
        nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               targets.reshape(-1), reduction="sum").item()
        count += targets.numel()
    return math.exp(min(nll / count, 700.0))


def evaluate_ppl(model, tokens, contexts, last_k=32, max_tokens=None, device=None):
    """Devuelve {"full": {ctx: ppl}, "last": {ctx: ppl}}. Un contexto que supera la tabla de la
    APE aprendida (model.max_seq_len) da None."""
    device = get_device(device) if device is not None else next(model.parameters()).device
    if max_tokens:
        tokens = tokens[:max_tokens]
    limit = getattr(model, "max_seq_len", None)
    out = {"full": {}, "last": {}}
    for ctx in contexts:
        if limit is not None and ctx > limit:
            out["full"][ctx] = out["last"][ctx] = None
            continue
        out["full"][ctx] = window_ppl(model, tokens, ctx, ctx, ctx, device)
        k = min(last_k, ctx)
        out["last"][ctx] = window_ppl(model, tokens, ctx, k, k, device)
    return out


#####################################
# Una ejecución y el experimento
#####################################
def _val_tokens(text, tokenizer, train_ratio=0.9):
    """Mismos datos de validación que model_base.pretrain (división por caracteres)."""
    return torch.tensor(tokenizer.encode(text[int(train_ratio * len(text)):]), dtype=torch.long)


def run_one(kind, seed, text, tokenizer, train_config, eval_config, device=None, ckpt_dir=None):
    """Entrena un modelo con la codificación `kind` y evalúa la perplejidad por contexto."""
    tc = train_config
    contexts = eval_config["contexts"]
    base = get_small_config(tokenizer.n_vocab, tc["train_ctx"], tc["emb_dim"], tc["n_heads"],
                            tc["n_layers"], tc["drop_rate"])
    cfg = make_config(base, kind, max_positions=max(max(contexts), tc["train_ctx"]))
    with contextlib.redirect_stdout(io.StringIO()) if tc.get("quiet") else contextlib.nullcontext():
        res = pretrain(cfg, text, tokenizer, model_factory=GPTModelExt, num_epochs=tc["epochs"],
                       batch_size=tc["batch_size"], learning_rate=tc["lr"], weight_decay=tc["weight_decay"],
                       eval_freq=50, eval_iter=5, start_context="", seed=seed, device=device)
    model = res["model"]
    ppl = evaluate_ppl(model, _val_tokens(text, tokenizer), contexts, eval_config["last_k"],
                       eval_config["max_eval_tokens"], device)
    if ckpt_dir:
        save_checkpoint(Path(ckpt_dir) / f"{kind}_s{seed}.pt", model, cfg, tokenizer)
    return {
        "pos_encoding": kind, "seed": seed, "train_config": tc, "eval_config": eval_config,
        "n_params": res["n_params"], "val_loss_train_ctx": res["final_val_loss"],
        "ppl": {m: {str(c): v for c, v in d.items()} for m, d in ppl.items()},
        "train_time_s": res["train_time_s"], "train_tokens_per_s": res["train_tokens_per_s"],
    }


def load_results(path):
    """Lee results.jsonl; las claves de contexto vuelven a ser enteros."""
    path = Path(path)
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["ppl"] = {m: {int(c): v for c, v in d.items()} for m, d in r["ppl"].items()}
            results.append(r)
    return results


def _save_results(path, results):
    lines = []
    for r in results:
        r = dict(r, ppl={m: {str(c): v for c, v in d.items()} for m, d in r["ppl"].items()})
        lines.append(json.dumps(r))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_experiment(kinds, seeds, out_dir, text, tokenizer, train_config, eval_config, device=None):
    """Ejecuta las combinaciones (codificación, semilla) que falten. Reanudable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    existing = load_results(path)
    for r in existing:
        if r["train_config"] != train_config:
            raise ValueError(
                f"{path} contiene resultados con otros hiperparámetros de entrenamiento "
                f"({r['train_config']}). Usa otra carpeta con --out-dir.")
    done = {(r["pos_encoding"], r["seed"]) for r in existing}
    for kind in kinds:
        for seed in seeds:
            if (kind, seed) in done:
                continue
            rec = run_one(kind, seed, text, tokenizer, train_config, eval_config, device, out_dir / "ckpt")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            last = rec["ppl"]["last"]
            print(f"{kind:11s} seed {seed}: ppl(last) " + "  ".join(
                f"{c}:{v:.2f}" if v is not None else f"{c}:-" for c, v in last.items())
                + f" | {rec['train_time_s']:.0f} s", flush=True)
    return load_results(path)


def reeval(out_dir, text, tokenizer, contexts, last_k=32, max_eval_tokens=40000, device=None):
    """Recalcula la perplejidad de los checkpoints guardados con otros contextos, sin
    reentrenar, y actualiza results.jsonl."""
    out_dir = Path(out_dir)
    results = load_results(out_dir / "results.jsonl")
    tokens = _val_tokens(text, tokenizer)
    for r in results:
        model, _, _ = load_checkpoint(out_dir / "ckpt" / f"{r['pos_encoding']}_s{r['seed']}.pt",
                                      model_factory=GPTModelExt, device=get_device(device))
        r["ppl"] = evaluate_ppl(model, tokens, contexts, last_k, max_eval_tokens, device)
        r["eval_config"] = {"contexts": list(contexts), "last_k": last_k, "max_eval_tokens": max_eval_tokens}
    _save_results(out_dir / "results.jsonl", results)
    return load_results(out_dir / "results.jsonl")


#####################################
# Resúmenes
#####################################
def _mean_std(values):
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def summarize(results, metric="last"):
    """{codificación: {contexto: (media, desviación, nº de semillas)}} (ignora los None)."""
    groups = {}
    for r in results:
        for ctx, v in r["ppl"][metric].items():
            if v is not None:
                groups.setdefault(r["pos_encoding"], {}).setdefault(ctx, []).append(v)
    return {k: {c: (*_mean_std(v), len(v)) for c, v in sorted(d.items())} for k, d in groups.items()}


def format_table(results, metric="last"):
    stats = summarize(results, metric)
    contexts = sorted({c for d in stats.values() for c in d})
    train_ctx = results[0]["train_config"]["train_ctx"]
    lines = ["codificación".ljust(13) + "".join((f"ctx {c}" + ("*" if c == train_ctx else "")).rjust(16) for c in contexts),
             f"(perplejidad '{metric}', media ± desv.; * = contexto de entrenamiento)"]
    for kind in [k for k in PE_KINDS if k in stats]:
        row = kind.ljust(13)
        for c in contexts:
            row += (f"{stats[kind][c][0]:.2f}±{stats[kind][c][1]:.2f}" if c in stats[kind] else "-").rjust(16)
        lines.append(row)
    return "\n".join(lines)


def plot_ppl_curves(results, metric="last", save_path=None):
    """Perplejidad (escala logarítmica) frente al contexto, una curva por codificación."""
    import matplotlib.pyplot as plt
    stats = summarize(results, metric)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for kind in [k for k in PE_KINDS if k in stats]:
        ctxs = sorted(stats[kind])
        mean = [stats[kind][c][0] for c in ctxs]
        std = [stats[kind][c][1] for c in ctxs]
        ax.plot(ctxs, mean, marker="o", markersize=3, label=kind)
        ax.fill_between(ctxs, [max(m - s, 1e-9) for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], alpha=0.15)
    ax.axvline(results[0]["train_config"]["train_ctx"], color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Longitud de contexto")
    ax.set_ylabel(f"Perplejidad ({metric})")
    ax.set_title("Línea discontinua: contexto de entrenamiento", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


#####################################
# Autotest: python context_ppl.py selftest
#####################################
def run_selftest():
    import numpy as np  # noqa: F401  (comprobación de entorno)
    import torch.nn as nn

    outcomes = []

    def check(name, cond, detail=""):
        outcomes.append(bool(cond))
        print(f"[{'OK' if cond else 'FALLO'}] {name} {detail}".rstrip())

    V = 11
    cyc = torch.arange(500) % V

    class Uniform(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.zeros(x.shape[0], x.shape[1], V) + self.p

    class Perfect(nn.Module):  # predice (token + 1) % V con total seguridad
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return F.one_hot((x + 1) % V, V).float() * 30.0 + self.p

    dev = torch.device("cpu")
    check("Modelo uniforme: la perplejidad es exactamente el tamaño del vocabulario",
          abs(window_ppl(Uniform(), cyc, 16, 16, 16, dev) - V) < 1e-4
          and abs(window_ppl(Uniform(), cyc, 64, 8, 8, dev) - V) < 1e-4)
    check("Modelo perfecto: perplejidad 1",
          window_ppl(Perfect(), cyc, 32, 32, 32, dev) < 1.0001 and window_ppl(Perfect(), cyc, 32, 4, 4, dev) < 1.0001)
    try:
        window_ppl(Uniform(), cyc[:10], 32, 32, 32, dev)
        check("Error claro si no caben ventanas", False)
    except ValueError:
        check("Error claro si no caben ventanas", True)

    # Comparación con un bucle manual y un modelo real
    torch.manual_seed(0)
    cfg = make_config(get_small_config(V, 16, 32, 4, 2, 0.0), "rope")
    model = GPTModelExt(cfg).eval()
    tokens = torch.randint(0, V, (200,))
    for ctx, k in ((16, 16), (24, 8)):
        nll, cnt = 0.0, 0
        for s in range(0, len(tokens) - ctx, k):  # ventanas que avanzan k tokens
            w = tokens[s:s + ctx + 1]
            logits = model(w[:-1].unsqueeze(0))[0, -k:]
            nll += F.cross_entropy(logits, w[1:][-k:], reduction="sum").item()
            cnt += k
        check(f"window_ppl == bucle manual (contexto {ctx}, últimos {k})",
              abs(window_ppl(model, tokens, ctx, k, k, dev) - math.exp(nll / cnt)) < 1e-4)
    check("evaluate_ppl: la APE aprendida devuelve None más allá de su tabla",
          evaluate_ppl(GPTModelExt(make_config(get_small_config(V, 16, 32, 4, 2, 0.0), "learned")).eval(),
                       tokens, (8, 32))["last"][32] is None)

    # Experimento pequeño de extremo a extremo (texto periódico)
    text = "abcde fghij " * 300
    tok = get_tokenizer("char", text)
    tc = dict(train_ctx=16, epochs=10, batch_size=8, lr=3e-3, weight_decay=0.1, emb_dim=32, n_heads=4,
              n_layers=2, drop_rate=0.0, data_id="periodico", quiet=True)
    ec = dict(contexts=[8, 16, 32], last_k=8, max_eval_tokens=None)
    with tempfile.TemporaryDirectory() as tmp:
        kinds = ["learned", "nope", "rope", "alibi"]
        res = run_experiment(kinds, [0], tmp, text, tok, tc, ec, device="cpu")
        finite = all(v is not None and 1.0 <= v < 1e6 for r in res for d in r["ppl"].values() for v in d.values())
        check("run_experiment: 4 ejecuciones con perplejidades finitas en todos los contextos", len(res) == 4 and finite)
        check("Los modelos aprenden el texto periódico (ppl 'last' en el contexto de entrenamiento < 1.5)",
              all(r["ppl"]["last"][16] < 1.5 for r in res),
              "(" + ", ".join(f"{r['pos_encoding']}: {r['ppl']['last'][16]:.2f}" for r in res) + ")")
        check("Se guardan los checkpoints", all((Path(tmp) / "ckpt" / f"{k}_s0.pt").exists() for k in kinds))

        n_lines = len((Path(tmp) / "results.jsonl").read_text().splitlines())
        run_experiment(kinds, [0], tmp, text, tok, tc, ec, device="cpu")
        check("Reanudable: la segunda llamada no repite ejecuciones",
              len((Path(tmp) / "results.jsonl").read_text().splitlines()) == n_lines)
        try:
            run_experiment(kinds, [1], tmp, text, tok, {**tc, "epochs": 11}, ec, device="cpu")
            check("Se niega a mezclar hiperparámetros de entrenamiento distintos", False)
        except ValueError:
            check("Se niega a mezclar hiperparámetros de entrenamiento distintos", True)

        again = reeval(tmp, text, tok, [8, 16, 32], last_k=8, max_eval_tokens=None, device="cpu")
        same = all(abs(a["ppl"]["last"][c] - b["ppl"]["last"][c]) < 1e-5 for a, b in zip(res, again) for c in (8, 16, 32))
        check("reeval con los mismos contextos reproduce la perplejidad guardada", same)
        wide = reeval(tmp, text, tok, [8, 24, 48], last_k=8, max_eval_tokens=None, device="cpu")
        by_kind = {r["pos_encoding"]: r["ppl"]["last"] for r in wide}
        check("reeval con otros contextos: la APE aprendida usa su tabla (hasta 32) y el resto llega a 48",
              by_kind["learned"][8] is not None and by_kind["learned"][48] is None
              and by_kind["rope"][48] is not None)

        check("format_table incluye todas las codificaciones", all(k in format_table(wide) for k in kinds))
        fig = plot_ppl_curves(wide, save_path=Path(tmp) / "ppl.png")
        import matplotlib.pyplot as plt
        plt.close(fig)
        check("plot_ppl_curves guarda la figura", (Path(tmp) / "ppl.png").exists())

    ok = all(outcomes)
    print(f"\n{sum(outcomes)}/{len(outcomes)} comprobaciones correctas" + ("" if ok else " (hay fallos)"))
    return ok


#####################################
# Línea de comandos
#####################################
def _load_data(args):
    text = load_text(args.data, args.data_dir)
    if args.max_chars:
        text = text[:args.max_chars]
    return text, get_tokenizer(args.tokenizer, text)


def main(argv=None):
    p = argparse.ArgumentParser(description="Perplejidad frente a longitud de contexto")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")

    def add_data_args(sp):
        sp.add_argument("--data", default="tinyshakespeare", help="'tinyshakespeare', 'verdict' o ruta a un .txt")
        sp.add_argument("--data-dir", default="data")
        sp.add_argument("--max-chars", type=int, default=None)
        sp.add_argument("--tokenizer", choices=["char", "gpt2"], default="char")
        sp.add_argument("--device", default=None)

    r = sub.add_parser("run", help="Entrena y evalúa (reanudable)")
    add_data_args(r)
    r.add_argument("--pe", nargs="+", default=["learned", "sinusoidal", "nope", "rope", "alibi", "t5"], choices=PE_KINDS)
    r.add_argument("--seeds", nargs="+", type=int, default=[0])
    r.add_argument("--train-ctx", type=int, default=128)
    r.add_argument("--contexts", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    r.add_argument("--last-k", type=int, default=32)
    r.add_argument("--max-eval-tokens", type=int, default=40000)
    r.add_argument("--epochs", type=int, default=5)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--weight-decay", type=float, default=0.1)
    r.add_argument("--emb-dim", type=int, default=128)
    r.add_argument("--n-heads", type=int, default=4)
    r.add_argument("--n-layers", type=int, default=4)
    r.add_argument("--drop-rate", type=float, default=0.1)
    r.add_argument("--out-dir", default="runs/context_ppl")

    e = sub.add_parser("reeval", help="Recalcula la perplejidad con otros contextos sin reentrenar")
    add_data_args(e)
    e.add_argument("--out-dir", default="runs/context_ppl")
    e.add_argument("--contexts", nargs="+", type=int, required=True)
    e.add_argument("--last-k", type=int, default=32)
    e.add_argument("--max-eval-tokens", type=int, default=40000)

    s = sub.add_parser("summarize", help="Tablas y gráficas")
    s.add_argument("--dir", default="runs/context_ppl")

    args = p.parse_args(argv)
    if args.command == "selftest":
        raise SystemExit(0 if run_selftest() else 1)

    if args.command == "run":
        text, tokenizer = _load_data(args)
        train_config = dict(train_ctx=args.train_ctx, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            weight_decay=args.weight_decay, emb_dim=args.emb_dim, n_heads=args.n_heads,
                            n_layers=args.n_layers, drop_rate=args.drop_rate,
                            data_id=f"{args.data}|max_chars={args.max_chars}|tok={args.tokenizer}|n={len(text)}")
        eval_config = dict(contexts=args.contexts, last_k=args.last_k, max_eval_tokens=args.max_eval_tokens)
        print(f"Texto: {len(text):,} caracteres | vocab {tokenizer.n_vocab} | dispositivo {get_device(args.device)}")
        run_experiment(args.pe, args.seeds, args.out_dir, text, tokenizer, train_config, eval_config, args.device)
        args.dir = args.out_dir
    elif args.command == "reeval":
        text, tokenizer = _load_data(args)
        reeval(args.out_dir, text, tokenizer, args.contexts, args.last_k, args.max_eval_tokens, args.device)
        args.dir = args.out_dir

    results = load_results(Path(args.dir) / "results.jsonl")
    if not results:
        raise SystemExit(f"No hay resultados en {args.dir}/results.jsonl")
    import matplotlib.pyplot as plt
    for metric in ("last", "full"):
        print("\n" + format_table(results, metric))
        plt.close(plot_ppl_curves(results, metric, save_path=Path(args.dir) / f"ppl_{metric}.png"))
    print(f"\nGráficas en {args.dir}/ppl_last.png y ppl_full.png")


if __name__ == "__main__":
    main()
