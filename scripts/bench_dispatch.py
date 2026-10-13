# bench_dispatch.py
#
# Mide el tiempo de forward+backward de una capa feed-forward densa frente a la
# capa MoE con los tres despachos (dense, loop, sorted). Guarda results/dispatch_bench.json.
#
#   python scripts/bench_dispatch.py

import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from model_base import FeedForward  # noqa: E402
from moe import MoEFeedForward  # noqa: E402

D, B, T = 128, 32, 128          # mismas dimensiones que los experimentos de Tiny Shakespeare
WARMUP, REPS = 5, 30


def time_fwd_bwd(module, x):
    for _ in range(WARMUP):
        module(x).sum().backward()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        module(x).sum().backward()
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(times), "stdev_ms": statistics.stdev(times),
            "tokens_per_s": B * T / (statistics.median(times) / 1000)}


def main():
    torch.manual_seed(0)
    x = torch.randn(B, T, D)
    results = {"setup": {"emb_dim": D, "batch": B, "tokens": T, "warmup": WARMUP, "reps": REPS,
                         "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__},
               "runs": []}

    r = time_fwd_bwd(FeedForward({"emb_dim": D}), x)
    results["runs"].append({"layer": "FFN denso", "dispatch": "-", **r})
    print(f"FFN denso            {r['median_ms']:7.2f} ms")

    for E, k in [(4, 1), (4, 2), (8, 1), (8, 2)]:
        moe = MoEFeedForward({"emb_dim": D, "moe": {"n_experts": E, "top_k": k}})
        for dispatch in ["dense", "loop", "sorted"]:
            moe.dispatch = dispatch
            r = time_fwd_bwd(moe, x)
            results["runs"].append({"layer": f"MoE E={E} k={k}", "dispatch": dispatch, **r})
            print(f"MoE E={E} k={k} {dispatch:6s}  {r['median_ms']:7.2f} ms")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "dispatch_bench.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
