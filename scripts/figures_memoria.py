# figures_memoria.py
#
# Figura de la memoria (memoria/figuras/fig_principal.pdf), generada a partir de los
# resultados guardados, sin reentrenar:
#   (a) perplejidad frente a la longitud de contexto para cada codificación posicional,
#       con FFN denso (discontinua) y MoE E=4 k=1 (continua): runs/context_moe/results.jsonl
#   (b) carga del experto más usado (peor capa) por tramo de posición en los modelos MoE:
#       ¿cambia el reparto del router en posiciones no vistas? (mismo fichero)
#   (c) carga del experto más usado por capa según el coeficiente de L_aux:
#       results/summary.json
#
#   python scripts/figures_memoria.py

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "memoria" / "figuras"

# Paleta categórica de referencia validada (guía de visualización): slots 1-4 para las
# codificaciones y 5-7 para la ablación, para no reutilizar colores con otro significado.
PE_STYLE = {
    "learned": ("APE aprendida", "#2a78d6", "o"),
    "nope": ("NoPE", "#eb6834", "s"),
    "alibi": ("ALiBi", "#1baf7a", "^"),
    "t5": ("T5", "#eda100", "D"),
}
AUX_STYLE = [("moe_e4k1_aux0", r"$\alpha=0$", "#e87ba4", "o"),
             ("moe_e4k1", r"$\alpha=0.01$", "#008300", "s"),
             ("moe_e4k1_aux0.1", r"$\alpha=0.1$", "#6250d6", "^")]
TEXT_2, GRID = "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=7)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_context(ax):
    rows = [json.loads(line) for line in (ROOT / "runs" / "context_moe" / "results.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    for pe, (label, color, marker) in PE_STYLE.items():
        for ffn, ls, fill in (("dense", "--", "white"), ("moe_e4k1", "-", color)):
            rs = [r for r in rows if r["pos_encoding"] == pe and r["ffn"] == ffn]
            if not rs:
                continue
            ctxs = sorted(int(c) for c in rs[0]["ppl"]["last"])
            mean = [st.mean(r["ppl"]["last"][str(c)] for r in rs) for c in ctxs]
            ax.plot(ctxs, mean, ls=ls, color=color, lw=1.3, marker=marker, ms=3.5,
                    mfc=fill, mec=color, mew=0.9)
    ax.axvline(128, color=TEXT_2, lw=0.7, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(["32", "64", "128", "256", "512", "1024"])
    ax.set_yticks([5, 7, 10, 15, 20, 30])
    ax.set_yticklabels(["5", "7", "10", "15", "20", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Longitud de contexto en evaluación", color=TEXT_2)
    ax.set_ylabel("Perplejidad (últimos 32 tokens)", color=TEXT_2)
    ax.set_title("(a) Codificación posicional × FFN (P1)", fontsize=8, color="#0b0b0b")


def panel_position(ax):
    rows = [json.loads(line) for line in (ROOT / "runs" / "context_moe" / "results.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    for pe, (label, color, marker) in PE_STYLE.items():
        rs = [r for r in rows if r["pos_encoding"] == pe and "routing" in r]
        if not rs:
            continue
        n_buckets = len(rs[0]["routing"]["layers"][0]["fractions"])
        bucket = rs[0]["routing"]["bucket"]
        centers = [bucket * (b + 0.5) for b in range(n_buckets)]
        # carga del experto más usado en la peor capa de cada tramo, media de semillas
        load = [st.mean(max(max(layer["fractions"][b]) for layer in r["routing"]["layers"]) for r in rs)
                for b in range(n_buckets)]
        ax.plot(centers, load, color=color, lw=1.3, marker=marker, ms=3.5, label=label)
    ax.axhline(0.25, color=TEXT_2, lw=0.7, ls="--")
    ax.axvline(128, color=TEXT_2, lw=0.7, ls=":")
    ax.text(1000, 0.235, "1/E", fontsize=7, color=TEXT_2, ha="right", va="top")
    ax.set_xlim(0, 1024)
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Posición en la ventana", color=TEXT_2)
    ax.set_ylabel("Carga máx. (peor capa)", color=TEXT_2)
    ax.set_title("(b) Router MoE por posición (P1)", fontsize=8, color="#0b0b0b")


def panel_load(ax):
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    layers = [1, 2, 3, 4]
    for i, (name, label, color, marker) in enumerate(AUX_STYLE):
        per_seed = summary[name]["max_expert_load_per_layer"]     # (semillas, capas)
        offset = (i - 1) * 0.12
        for seed_vals in per_seed:
            ax.scatter([l + offset for l in layers], seed_vals, color=color, s=8, alpha=0.45,
                       linewidths=0, marker=marker)
        ax.plot([l + offset for l in layers], [st.mean(c) for c in zip(*per_seed)], color=color,
                lw=1.3, marker=marker, ms=4, label=label)
    ax.axhline(0.25, color=TEXT_2, lw=0.7, ls="--")
    ax.text(4.62, 0.262, "1/E", fontsize=7, color=TEXT_2, ha="right", va="bottom")
    ax.set_xticks(layers)
    ax.set_xlim(0.6, 4.7)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Capa", color=TEXT_2)
    ax.set_ylabel("Carga del experto más usado", color=TEXT_2)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", title=r"coef. de $\mathcal{L}_{aux}$",
              title_fontsize=6.5)
    ax.set_title("(c) Balanceo de carga (P2)", fontsize=8, color="#0b0b0b")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.3, 2.5), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    for ax in (a, b, c):
        style(ax)
    panel_context(a)
    panel_position(b)
    panel_load(c)
    # Leyenda común de (a) y (b): color = codificación; trazo = FFN denso (discontinuo) o MoE (continuo)
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.3, ms=3.5, label=l) for l, c, m in PE_STYLE.values()]
    handles += [plt.Line2D([], [], color=TEXT_2, ls="--", lw=1.1, marker="o", mfc="white", ms=3.5, label="FFN denso"),
                plt.Line2D([], [], color=TEXT_2, ls="-", lw=1.1, marker="o", ms=3.5, label="MoE $E{=}4$, $k{=}1$")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 1.0), ncol=6, fontsize=6.8,
               frameon=False, handlelength=2.0, columnspacing=1.1)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_principal.pdf")
    fig.savefig(OUT / "fig_principal.png", dpi=200)
    plt.close(fig)
    print(f"Figura en {OUT / 'fig_principal.pdf'}")


if __name__ == "__main__":
    main()
