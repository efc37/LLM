# analyze.py
#
# Agrega los resultados de results/*.json (3 semillas por configuración) y genera:
#   results/summary.json    cifras agregadas (media y desviación entre semillas)
#   results/summary.md      tablas en markdown, listas para la memoria
#   results/fig_val_ce.png  curvas de val CE: denso vs MoE vs denso ancho
#   results/fig_expert_load.png  carga del experto más usado por capa (ablación de L_aux)
#
#   python scripts/analyze.py

import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
SEEDS = [1, 2, 3]

LABELS = {
    "dense": "Denso (libro)",
    "moe_e4k1": "MoE E=4 k=1",
    "moe_e4k2": "MoE E=4 k=2",
    "dense_wide": "Denso ancho (FFN 4x)",
    "moe_e4k1_aux0": "MoE E=4 k=1, coef. aux 0",
    "moe_e4k1_aux0.1": "MoE E=4 k=1, coef. aux 0.1",
    "moe_e4k1_z": "MoE E=4 k=1, z-loss 0.001",
}

# Paleta categórica de referencia (slots 1-4 en orden fijo), validada con el
# validador de la guía de visualización: pasa CVD y visión normal; aqua y amarillo
# tienen contraste < 3:1, por eso las líneas llevan etiqueta directa y marcador.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(name):
    return [json.load(open(os.path.join(RES, f"{name}_seed{s}.json"), encoding="utf-8"))
            for s in SEEDS]


def mean_std(values):
    return st.mean(values), st.stdev(values)


def summarize(name, runs):
    out = {"label": LABELS[name], "params_total": runs[0]["params_total"],
           "params_active": runs[0]["params_active"]}
    for key, getter in [
        ("val_ce", lambda r: r["final_val_ce"]),
        ("train_ce", lambda r: r["history"][-1]["train_ce"]),
        ("tokens_per_s", lambda r: r["train_tokens_per_s"]),
        ("train_time_min", lambda r: r["train_time_s"] / 60),
    ]:
        m, s = mean_std([getter(r) for r in runs])
        out[key] = {"mean": m, "std": s, "per_seed": [getter(r) for r in runs]}
    if "expert_fraction" in runs[0]["history"][-1] and runs[0]["config"]["model"]["moe"]["n_experts"] > 1:
        # Carga del experto más usado en cada capa (últimos 100 pasos); 1/E = equilibrio perfecto
        max_load = [[max(row) for row in r["history"][-1]["expert_fraction"]] for r in runs]
        out["max_expert_load_per_layer"] = max_load
        out["max_expert_load"] = mean_std([max(layers) for layers in max_load])
        out["train_aux"] = mean_std([r["history"][-1]["train_aux"] for r in runs])
    return out


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_val_ce(all_runs, path):
    names = ["dense", "moe_e4k1", "moe_e4k2", "dense_wide"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200, facecolor=SURFACE)
    style_axes(ax)
    for name, color, marker in zip(names, COLORS, markers):
        runs = all_runs[name]
        steps = [h["step"] for h in runs[0]["history"]]
        per_step = list(zip(*[[h["val_ce"] for h in r["history"]] for r in runs]))
        mean = [st.mean(v) for v in per_step]
        ax.fill_between(steps, [min(v) for v in per_step], [max(v) for v in per_step],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=1.5, label=LABELS[name],
                marker=marker, markersize=3, markevery=4)
    ax.set_xlabel("Paso de entrenamiento", fontsize=8, color=TEXT_2)
    ax.set_ylabel("Val CE (nats/carácter)", fontsize=8, color=TEXT_2)
    ax.set_ylim(1.5, 2.1)
    ax.set_xlim(0, 2050)
    leg = ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    for text in leg.get_texts():
        text.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_expert_load(summary, path):
    names = ["moe_e4k1_aux0", "moe_e4k1", "moe_e4k1_aux0.1"]
    coefs = ["coef. 0", "coef. 0.01", "coef. 0.1"]
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200, facecolor=SURFACE)
    style_axes(ax)
    layers = [1, 2, 3, 4]
    for i, (name, coef, color, marker) in enumerate(zip(names, coefs, COLORS, markers)):
        per_seed = summary[name]["max_expert_load_per_layer"]   # (semillas, capas)
        offset = (i - 1) * 0.12
        for seed_vals in per_seed:
            ax.scatter([l + offset for l in layers], seed_vals, color=color, s=10, alpha=0.5,
                       linewidths=0, marker=marker)
        mean = [st.mean(col) for col in zip(*per_seed)]
        ax.plot([l + offset for l in layers], mean, color=color, linewidth=1.5,
                marker=marker, markersize=4, label=coef)
    ax.axhline(0.25, color=TEXT_2, linewidth=0.8, linestyle="--")
    ax.text(4.68, 0.26, "1/E", fontsize=7, color=TEXT_2, va="bottom", ha="right")
    ax.set_xticks(layers)
    ax.set_xlabel("Capa", fontsize=8, color=TEXT_2)
    ax.set_ylabel("Carga del experto más usado", fontsize=8, color=TEXT_2)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(0.6, 4.7)
    leg = ax.legend(fontsize=6.5, frameon=False, loc="upper right", title="L_aux", title_fontsize=6.5)
    for text in leg.get_texts():
        text.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def markdown(summary, bench):
    lines = ["## Tiny Shakespeare (char-level), 2000 pasos, 3 semillas (media ± desviación)", "",
             "| Modelo | Parám. totales | Parám. activos | Val CE | Train CE | tok/s | Tiempo (min) | Carga máx. experto | L_aux |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        load = s.get("max_expert_load")
        aux = s.get("train_aux")
        lines.append(
            f"| {s['label']} | {s['params_total']:,} | {s['params_active']:,} "
            f"| {s['val_ce']['mean']:.4f} ± {s['val_ce']['std']:.4f} "
            f"| {s['train_ce']['mean']:.4f} ± {s['train_ce']['std']:.4f} "
            f"| {s['tokens_per_s']['mean']:,.0f} ± {s['tokens_per_s']['std']:,.0f} "
            f"| {s['train_time_min']['mean']:.1f} "
            f"| {f'{load[0]:.3f} ± {load[1]:.3f}' if load else '-'} "
            f"| {f'{aux[0]:.3f} ± {aux[1]:.3f}' if aux else '-'} |")
    lines += ["", "Carga máx. experto: fracción de asignaciones del experto más usado en la peor capa "
              "(últimos 100 pasos); 0.25 = equilibrio perfecto con E=4.", "",
              "## Despacho: forward+backward de una capa (B=32, T=128, D=128), mediana de 30 repeticiones", "",
              "| Capa | Despacho | ms | tok/s |", "|---|---|---|---|"]
    for r in bench["runs"]:
        lines.append(f"| {r['layer']} | {r['dispatch']} | {r['median_ms']:.2f} ± {r['stdev_ms']:.2f} "
                     f"| {r['tokens_per_s']:,.0f} |")
    lines += ["", f"torch {bench['setup']['torch_version']}, {bench['setup']['torch_threads']} hilos, CPU."]
    return "\n".join(lines) + "\n"


def main():
    all_runs = {name: load(name) for name in LABELS}
    summary = {name: summarize(name, runs) for name, runs in all_runs.items()}
    with open(os.path.join(RES, "dispatch_bench.json"), encoding="utf-8") as f:
        bench = json.load(f)

    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as f:
        f.write(markdown(summary, bench))
    plot_val_ce(all_runs, os.path.join(RES, "fig_val_ce.png"))
    plot_expert_load(summary, os.path.join(RES, "fig_expert_load.png"))
    print(markdown(summary, bench))


if __name__ == "__main__":
    main()
