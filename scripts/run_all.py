# run_all.py
#
# Lanza toda la batería de experimentos de la ampliación MoE, en secuencia (para
# que las medidas de tokens/s no se pisen entre sí). Se puede relanzar: salta las
# ejecuciones cuyo JSON ya existe en results/ (usa --force para repetirlas).
#
#   python scripts/run_all.py

import argparse
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CONFIGS = [
    "dense",            # modelo base del libro
    "moe_e4k1",         # MoE, mismo cómputo por token que el denso (Switch, k=1)
    "moe_e4k2",         # MoE, 2 expertos activos por token (Mixtral)
    "dense_wide",       # denso con los mismos parámetros totales que E=4
    "moe_e4k1_aux0",    # ablación: sin pérdida de balanceo
    "moe_e4k1_aux0.1",  # ablación: pérdida de balanceo 10 veces mayor
]
SEEDS = [1, 2, 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Se recorre por semilla para tener cuanto antes una semilla completa de todo
    for seed in SEEDS:
        for name in CONFIGS:
            out = os.path.join(ROOT, "results", f"{name}_seed{seed}.json")
            if os.path.exists(out) and not args.force:
                print(f"[saltado] {name} seed {seed}")
                continue
            print(f"[ejecutando] {name} seed {seed}", flush=True)
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_shakespeare.py"),
                            "--config", os.path.join(ROOT, "configs", f"{name}.json"),
                            "--seed", str(seed), "--save-model"], check=True)

    print("[ejecutando] benchmark de despachos", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "bench_dispatch.py")], check=True)


if __name__ == "__main__":
    main()
