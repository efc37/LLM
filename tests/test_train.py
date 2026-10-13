# Tests del script de entrenamiento con un texto sintético (no descarga nada).
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_shakespeare import CharTokenizer, train  # noqa: E402

TEXT = "To be, or not to be, that is the question.\n" * 50


def tiny_config(moe=None):
    model = {"context_length": 16, "emb_dim": 32, "n_heads": 4, "n_layers": 2,
             "drop_rate": 0.0, "qkv_bias": False}
    if moe:
        model["moe"] = moe
    train_cfg = {"steps": 6, "batch_size": 4, "lr": 1e-3, "weight_decay": 0.1,
                 "eval_every": 3, "eval_batches": 2}
    return {"name": "tiny", "model": model, "train": train_cfg}


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer(TEXT)
    assert tok.decode(tok.encode("not to be")) == "not to be"
    assert tok.vocab_size == len(set(TEXT))


@pytest.mark.parametrize("moe", [None, {"n_experts": 4, "top_k": 2, "aux_loss_coef": 0.01}])
def test_train_runs_and_is_reproducible(moe):
    r1 = train(tiny_config(moe), seed=7, text=TEXT, verbose=False, prompt="To")
    r2 = train(tiny_config(moe), seed=7, text=TEXT, verbose=False, prompt="To")

    assert [h["step"] for h in r1["history"]] == [3, 6]
    # Misma semilla ⇒ mismas pérdidas y misma muestra de texto
    assert [h["val_ce"] for h in r1["history"]] == [h["val_ce"] for h in r2["history"]]
    assert r1["sample"] == r2["sample"]

    if moe is None:
        assert r1["params_total"] == r1["params_active"]
        assert all(h["train_aux"] == 0.0 for h in r1["history"])
    else:
        assert r1["params_active"] < r1["params_total"]
        fractions = r1["history"][-1]["expert_fraction"]   # (capas MoE, E)
        assert len(fractions) == 2 and len(fractions[0]) == 4
        assert all(abs(sum(row) - 1.0) < 1e-6 for row in fractions)
