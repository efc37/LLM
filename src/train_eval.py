# train_eval.py
#
# Banco de pruebas de generalización de longitud: entrena GPTModelExt (model_ext.py)
# con una codificación posicional (positional.py) sobre una tarea sintética (tasks.py)
# con longitudes 1..L y mide el exact-match por longitud hasta eval_max_len > L.
#
# Qué fija este módulo para que la comparación sea justa:
#   - mismo flujo de datos para todas las variantes: los lotes salen de un generador
#     aleatorio propio inicializado con la semilla (no del estado global de PyTorch);
#   - mismos hiperparámetros, calendario de la tasa de aprendizaje y número de pasos;
#   - mismas muestras de evaluación para todos (tasks.make_eval_set es determinista);
#   - la tabla de la APE aprendida se dimensiona a la longitud máxima de evaluación
#     (las posiciones nunca entrenadas conservan su inicialización: es el baseline).
#
# Desviación respecto al artículo (a mencionar en la memoria): los lotes se muestrean
# en línea (datos ilimitados) en vez de usar un conjunto finito, y el modelo es mucho
# más pequeño (por defecto 4 capas, emb_dim 128).
#
# Codificaciones: las de positional.PE_KINDS y los híbridos "rope_nopeN" (pe_labels.py).
#
# Uso:
#   python train_eval.py selftest
#   python train_eval.py run --tasks copy reverse --pe learned sinusoidal nope rope alibi \
#                            --seeds 0 1 2 --steps 3000 --out-dir runs/length
#   python train_eval.py summarize --dir runs/length
# `run` es reanudable: guarda cada ejecución en results.jsonl y omite las ya hechas.

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from model_base_adaptado import calc_loss_batch, count_parameters, get_device, get_small_config
from model_ext import GPTModelExt, make_config
from pe_labels import check_layers, ordered_labels, parse_pe_label, pe_label_type
from positional import PE_KINDS
from tasks import TASKS, VOCAB, check_generation, make_eval_set, max_total_length, sample_batch


#####################################
# Modelo y decodificación
#####################################
def make_model_config(task_name, pos_encoding, train_max_len, eval_max_len, emb_dim=128,
                      n_heads=4, n_layers=4, drop_rate=0.0):
    """Configuración del modelo para una tarea y una codificación posicional.

    context_length es la longitud de entrenamiento más larga; max_positions (que solo usa
    la APE aprendida) cubre la secuencia más larga de la evaluación.

    `pos_encoding` es una etiqueta de pe_labels: una clave de PE_KINDS o un híbrido
    "rope_nopeN" (RoPE salvo en las capas N, 2N, ... que van sin codificación posicional).
    """
    kind, extra = parse_pe_label(pos_encoding)
    check_layers(pos_encoding, n_layers)
    cfg = get_small_config(
        VOCAB.size, context_length=max_total_length(task_name, train_max_len),
        emb_dim=emb_dim, n_heads=n_heads, n_layers=n_layers, drop_rate=drop_rate)
    return make_config(cfg, kind, max_positions=max_total_length(task_name, eval_max_len), **extra)


@torch.no_grad()
def greedy_generate(model, prompts, max_new_tokens):
    """Decodificación voraz de un lote de prompts de igual longitud. Devuelve los
    max_new_tokens tokens generados, de forma (N, max_new_tokens)."""
    model.eval()
    idx = prompts
    for _ in range(max_new_tokens):
        next_token = torch.argmax(model(idx)[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return idx[:, prompts.shape[1]:]


@torch.no_grad()
def evaluate_lengths(model, task_name, lengths, n_per_length, device, eval_seed=0, batch_size=256):
    """Exact-match por longitud. Devuelve {n: precisión}."""
    task = TASKS[task_name]
    acc = {}
    for n in lengths:
        prompts, answers = make_eval_set(task_name, n, n_per_length, eval_seed)
        correct = []
        for i in range(0, len(prompts), batch_size):
            new = greedy_generate(model, prompts[i:i + batch_size].to(device), task.max_answer_len(n) + 1)
            correct += check_generation(new.cpu(), answers[i:i + batch_size])
        acc[n] = sum(correct) / len(correct)
    return acc


#####################################
# Una ejecución (tarea, codificación, semilla)
#####################################
def run_one(task, pos_encoding, seed, train_max_len=20, eval_max_len=40, steps=3000, batch_size=64,
            lr=1e-3, weight_decay=0.01, warmup_frac=0.05, emb_dim=128, n_heads=4, n_layers=4,
            drop_rate=0.0, n_eval=50, device=None, log_every=0):
    """Entrena y evalúa un modelo. Devuelve un diccionario serializable en JSON.

    id_acc: precisión media en las longitudes vistas (<= train_max_len).
    ood_acc: precisión media en las longitudes no vistas (> train_max_len).
    """
    device = get_device(device)
    cfg = make_model_config(task, pos_encoding, train_max_len, eval_max_len, emb_dim, n_heads, n_layers, drop_rate)
    torch.manual_seed(seed)
    model = GPTModelExt(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup = max(1, int(warmup_frac * steps))

    def lr_factor(step):  # calentamiento lineal y coseno hasta el 10 % de lr
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    data_rng = random.Random(seed)  # flujo de datos independiente del estado de PyTorch
    log_every = log_every or max(1, steps // 20)
    loss_curve, window, tokens_seen = [], [], 0
    t0 = time.time()
    model.train()
    for step in range(steps):
        inputs, targets = sample_batch(task, batch_size, 1, train_max_len, data_rng)
        loss = calc_loss_batch(inputs, targets, model, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tokens_seen += inputs.numel()
        window.append(loss.item())
        if (step + 1) % log_every == 0:
            loss_curve.append((step + 1, sum(window) / len(window)))
            window = []
    train_time = time.time() - t0

    acc = evaluate_lengths(model, task, range(1, eval_max_len + 1), n_eval, device)
    seen = [a for n, a in acc.items() if n <= train_max_len]
    unseen = [a for n, a in acc.items() if n > train_max_len]
    return {
        "task": task, "pos_encoding": pos_encoding, "seed": seed,
        "train_max_len": train_max_len, "eval_max_len": eval_max_len, "steps": steps,
        "batch_size": batch_size, "lr": lr, "n_params": count_parameters(model),
        "acc_by_length": acc,
        "id_acc": sum(seen) / len(seen), "ood_acc": sum(unseen) / len(unseen) if unseen else float("nan"),
        "loss_curve": loss_curve, "train_time_s": train_time,
        "train_tokens_per_s": tokens_seen / train_time,
    }


#####################################
# Experimentos reanudables y resúmenes
#####################################
def load_results(path):
    """Lee results.jsonl. Las claves de acc_by_length vuelven a ser enteros."""
    path = Path(path)
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["acc_by_length"] = {int(n): a for n, a in r["acc_by_length"].items()}
            results.append(r)
    return results


def run_experiment(tasks, kinds, seeds, out_dir, **run_kwargs):
    """Ejecuta todas las combinaciones (tarea, codificación, semilla) que falten y las
    guarda en out_dir/results.jsonl. Devuelve todos los resultados."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"
    done = {(r["task"], r["pos_encoding"], r["seed"]) for r in load_results(path)}
    for task in tasks:
        for kind in kinds:
            for seed in seeds:
                if (task, kind, seed) in done:
                    continue
                res = run_one(task, kind, seed, **run_kwargs)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"{task:13s} {kind:11s} seed {seed}: id {res['id_acc']:.3f} | ood {res['ood_acc']:.3f} "
                      f"| {res['train_time_s']:.0f} s", flush=True)
    return load_results(path)


def _mean_std(values):
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def summarize(results, metric="ood_acc"):
    """{(tarea, codificación): (media, desviación, nº de semillas)} de una métrica."""
    groups = {}
    for r in results:
        groups.setdefault((r["task"], r["pos_encoding"]), []).append(r[metric])
    return {key: (*_mean_std(vals), len(vals)) for key, vals in groups.items()}


def format_table(results):
    """Tabla de texto: por tarea y codificación, 'ID / OOD' como media ± desviación entre semillas."""
    ids, oods = summarize(results, "id_acc"), summarize(results, "ood_acc")
    tasks = sorted({t for t, _ in ids})
    kinds = ordered_labels({kk for _, kk in ids})
    lines = ["tarea".ljust(14) + "".join(k.center(24) for k in kinds), "(ID / OOD, media ± desv.)"]
    for t in tasks:
        row = t.ljust(14)
        for k in kinds:
            if (t, k) in ids:
                (im, isd, _), (om, osd, _) = ids[(t, k)], oods[(t, k)]
                row += f"{im:.2f}±{isd:.2f} / {om:.2f}±{osd:.2f}".center(24)
            else:
                row += "-".center(24)
        lines.append(row)
    return "\n".join(lines)


def rank_summary(results, metric="ood_acc"):
    """Ranking por tarea según la métrica (1 = mejor; los empates comparten rango) y
    media del rango inverso (MRR) de cada codificación, como en el artículo."""
    stats = summarize(results, metric)
    by_task = {}
    for (task, kind), (mean, _, _) in stats.items():
        by_task.setdefault(task, {})[kind] = mean
    reciprocal = {}
    for task, means in by_task.items():
        for kind, mean in means.items():
            rank = 1 + sum(1 for other in means.values() if other > mean + 1e-12)
            reciprocal.setdefault(kind, []).append(1.0 / rank)
    return {kind: sum(rs) / len(rs) for kind, rs in reciprocal.items()}


def plot_length_curves(results, task, save_path=None):
    """Precisión media (banda: desviación entre semillas) frente a la longitud, por codificación."""
    import matplotlib.pyplot as plt
    rows = [r for r in results if r["task"] == task]
    if not rows:
        raise ValueError(f"No hay resultados de la tarea {task!r}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for kind in ordered_labels({r["pos_encoding"] for r in rows}):
        runs = [r["acc_by_length"] for r in rows if r["pos_encoding"] == kind]
        lengths = sorted(runs[0])
        stats = [_mean_std([run[n] for run in runs]) for n in lengths]
        mean = [m for m, _ in stats]
        ax.plot(lengths, mean, label=kind)
        ax.fill_between(lengths, [m - s for m, s in stats], [min(1.0, m + s) for m, s in stats], alpha=0.15)
    ax.axvline(rows[0]["train_max_len"] + 0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Longitud n")
    ax.set_ylabel("Exact match")
    ax.set_title(f"{task} (línea: fin de las longitudes de entrenamiento)", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


#####################################
# Autotest: python train_eval.py selftest
#####################################
def run_selftest():
    import contextlib
    import io
    import tempfile

    outcomes = []

    def check(name, cond, detail=""):
        outcomes.append(bool(cond))
        print(f"[{'OK' if cond else 'FALLO'}] {name} {detail}".rstrip())

    tiny = dict(train_max_len=4, eval_max_len=8, steps=6, batch_size=8, emb_dim=32, n_heads=4,
                n_layers=2, n_eval=8, device="cpu")

    # Humo: todas las codificaciones entrenan y evalúan hasta 2L sin errores
    for kind in PE_KINDS:
        res = run_one("copy", kind, 0, **tiny)
        accs = list(res["acc_by_length"].values())
        check(f"'{kind}': entrena y evalúa las longitudes 1..8 (incluida la APE aprendida)",
              sorted(res["acc_by_length"]) == list(range(1, 9)) and all(0.0 <= a <= 1.0 for a in accs)
              and math.isfinite(res["id_acc"]) and math.isfinite(res["ood_acc"]))

    # Híbridos RoPE/NoPE por capas
    res = run_one("copy", "rope_nope2", 0, **tiny)
    check("'rope_nope2' (híbrido RoPE/NoPE): entrena y evalúa las longitudes 1..8",
          res["pos_encoding"] == "rope_nope2" and sorted(res["acc_by_length"]) == list(range(1, 9))
          and math.isfinite(res["id_acc"]))
    cfg_h = make_model_config("copy", "rope_nope4", 5, 10, n_layers=4)
    check("make_model_config traduce rope_nope4 a RoPE con nope_every=4",
          cfg_h["pos_encoding"] == "rope" and cfg_h["nope_every"] == 4)
    check("make_model_config sin híbrido no añade nope_every (las variantes antiguas no cambian)",
          "nope_every" not in make_model_config("copy", "rope", 5, 10, n_layers=4))
    try:
        make_model_config("copy", "rope_nope5", 5, 10, n_layers=4)
        check("Un híbrido sin ninguna capa NoPE (N > capas) lanza error claro", False)
    except ValueError:
        check("Un híbrido sin ninguna capa NoPE (N > capas) lanza error claro", True)

    # Reproducibilidad
    a = run_one("sort", "rope", 3, **{**tiny, "steps": 20})
    b = run_one("sort", "rope", 3, **{**tiny, "steps": 20})
    c = run_one("sort", "rope", 4, **{**tiny, "steps": 20})
    check("Misma semilla -> misma curva de pérdida y mismas precisiones",
          a["loss_curve"] == b["loss_curve"] and a["acc_by_length"] == b["acc_by_length"])
    check("Otra semilla -> otra curva de pérdida", a["loss_curve"] != c["loss_curve"])

    # El flujo de datos no depende de la codificación (misma semilla)
    first_batches = []
    for _ in range(2):
        rng = random.Random(5)
        first_batches.append(sample_batch("copy", 8, 1, 4, rng))
    check("El flujo de datos depende solo de la semilla",
          all(torch.equal(x, y) for x, y in zip(*first_batches)))

    # Aprendizaje real: copiar con longitudes 1..5 debe aprenderse en las longitudes vistas
    t0 = time.time()
    learn = run_one("copy", "sinusoidal", 0, train_max_len=5, eval_max_len=10, steps=1200, batch_size=64,
                    lr=2e-3, emb_dim=64, n_heads=4, n_layers=2, n_eval=40, device="cpu")
    check("El pipeline aprende: copy con longitudes 1..5 -> precisión en longitudes vistas >= 0.9",
          learn["id_acc"] >= 0.9, f"(id {learn['id_acc']:.2f}, ood {learn['ood_acc']:.2f}, {time.time() - t0:.0f} s)")
    check("La pérdida de entrenamiento baja durante el entrenamiento",
          learn["loss_curve"][-1][1] < 0.5 * learn["loss_curve"][0][1])

    # Experimentos reanudables, tabla, ranking y gráficas
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            r1 = run_experiment(["copy"], ["nope", "alibi", "rope_nope2"], [0, 1], tmp, **tiny)
            lines_before = len((Path(tmp) / "results.jsonl").read_text().splitlines())
            r2 = run_experiment(["copy"], ["nope", "alibi", "rope_nope2"], [0, 1], tmp, **tiny)
            lines_after = len((Path(tmp) / "results.jsonl").read_text().splitlines())
        check("run_experiment: reanudable (la segunda llamada no repite ejecuciones)",
              len(r1) == 6 and lines_before == lines_after == 6 and len(r2) == 6)
        check("load_results: las claves de longitud vuelven a ser enteros",
              all(isinstance(n, int) for n in r2[0]["acc_by_length"]))
        table = format_table(r2)
        check("format_table incluye la tarea, las codificaciones y el híbrido (al final)",
              "copy" in table and "alibi" in table and table.index("rope_nope2") > table.index("alibi"))
        fig = plot_length_curves(r2, "copy", save_path=Path(tmp) / "copy.png")
        import matplotlib.pyplot as plt
        plt.close(fig)
        check("plot_length_curves guarda la figura", (Path(tmp) / "copy.png").exists())

    fake = [{"task": t, "pos_encoding": k, "seed": s, "ood_acc": v, "id_acc": 1.0}
            for t, vals in {"a": {"x": 0.9, "y": 0.5, "z": 0.1}, "b": {"x": 0.2, "y": 0.8, "z": 0.8}}.items()
            for k, v in vals.items() for s in (0, 1)]
    mrr = rank_summary(fake)
    check("rank_summary: MRR con empates (x: 1 y 1/3, y: 1/2 y 1, z: 1/3 y 1)",
          abs(mrr["x"] - (1 + 1 / 3) / 2) < 1e-9 and abs(mrr["y"] - (1 / 2 + 1) / 2) < 1e-9
          and abs(mrr["z"] - (1 / 3 + 1) / 2) < 1e-9)

    ok = all(outcomes)
    print(f"\n{sum(outcomes)}/{len(outcomes)} comprobaciones correctas" + ("" if ok else " (hay fallos)"))
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description="Banco de pruebas de generalización de longitud")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="Comprobaciones internas")

    r = sub.add_parser("run", help="Entrena y evalúa todas las combinaciones (reanudable)")
    r.add_argument("--tasks", nargs="+", default=["copy", "reverse", "sum_mod"], choices=list(TASKS))
    r.add_argument("--pe", nargs="+", default=["learned", "sinusoidal", "nope", "rope", "alibi"], type=pe_label_type,
                   help=f"Codificaciones: {', '.join(PE_KINDS)} o híbridos rope_nopeN (p. ej. rope_nope4)")
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--train-max-len", type=int, default=20)
    r.add_argument("--eval-max-len", type=int, default=40)
    r.add_argument("--steps", type=int, default=3000)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--emb-dim", type=int, default=128)
    r.add_argument("--n-heads", type=int, default=4)
    r.add_argument("--n-layers", type=int, default=4)
    r.add_argument("--n-eval", type=int, default=50, help="Muestras de evaluación por longitud")
    r.add_argument("--device", default=None)
    r.add_argument("--out-dir", default="runs/length")

    s = sub.add_parser("summarize", help="Tabla, ranking y gráficas de unos resultados")
    s.add_argument("--dir", default="runs/length")

    args = p.parse_args(argv)
    if args.command == "selftest":
        raise SystemExit(0 if run_selftest() else 1)
    if args.command == "run":
        run_experiment(args.tasks, args.pe, args.seeds, args.out_dir, train_max_len=args.train_max_len,
                       eval_max_len=args.eval_max_len, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                       emb_dim=args.emb_dim, n_heads=args.n_heads, n_layers=args.n_layers,
                       n_eval=args.n_eval, device=args.device)
        args.dir = args.out_dir
    results = load_results(Path(args.dir) / "results.jsonl")
    if not results:
        raise SystemExit(f"No hay resultados en {args.dir}/results.jsonl")
    print("\n" + format_table(results))
    print("\nMRR por codificación (ranking por precisión OOD, 1 = mejor):")
    for kind, value in sorted(rank_summary(results).items(), key=lambda kv: -kv[1]):
        print(f"  {kind:11s} {value:.2f}")
    import matplotlib.pyplot as plt
    for task in sorted({r["task"] for r in results}):
        plt.close(plot_length_curves(results, task, save_path=Path(args.dir) / f"curves_{task}.png"))
    print(f"\nGráficas guardadas en {args.dir}/curves_<tarea>.png")


if __name__ == "__main__":
    main()
