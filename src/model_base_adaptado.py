# model_base.py
#
# Modelo base (GPT-2) de "Build a Large Language Model (From Scratch)".
# Consolida en un solo módulo el código de los capítulos 2 a 6 y lo deja listo
# para ejecutarse de principio a fin y para servir de referencia al evaluar las
# ampliaciones (misma configuración, mismos datos, mismas métricas).
#
# Origen del código:
#   Sebastian Raschka, Apache License 2.0
#   https://github.com/rasbt/LLMs-from-scratch
#   - ch04/01_main-chapter-code/gpt.py                (caps. 2-4)
#   - ch05/01_main-chapter-code/gpt_train.py          (cap. 5, entrenamiento)
#   - ch05/01_main-chapter-code/gpt_generate.py       (cap. 5, generación y pesos)
#   - ch06/01_main-chapter-code/gpt_class_finetune.py (cap. 6, clasificación)
#
# Secciones:
#   0. Configuraciones
#   1. Cap. 2: tokenizadores y datos
#   2. Cap. 3: atención multi-cabeza causal
#   3. Cap. 4: arquitectura GPT y generación voraz
#   4. Cap. 5: entrenamiento, generación con temperatura/top-k, pesos de GPT-2
#   5. Utilidades de experimentación: checkpoints, gráficas, métricas, pretrain()
#   6. Cap. 6: fine-tuning para clasificación (dataset de spam)
#   7. Autotest y línea de comandos
#
# La lógica de los modelos es la del repositorio. Lo que se ha añadido o
# cambiado respecto al código original, para poder reutilizarlo con ampliaciones:
#   - CharTokenizer: tokenizador de caracteres (vocabulario pequeño) con la misma
#     interfaz mínima que tiktoken. Modelos diminutos + BPE de GPT-2 (50257
#     tokens) hacen que la capa de embeddings domine los parámetros.
#   - train_model_simple y generate_and_print_sample aceptan `context_size`.
#     El original lo lee de `model.pos_emb`, que no existirá en variantes sin
#     embeddings posicionales aprendidos (RoPE, NoPE, ALiBi...).
#   - Dataloaders con semilla propia: el orden de los datos es idéntico entre
#     variantes del modelo aunque consuman distinta cantidad de números
#     aleatorios al inicializarse.
#   - Pérdida: calc_loss_batch ignora los objetivos con valor -100 (comportamiento
#     por defecto de cross_entropy), útil para entrenar solo sobre la respuesta.
#   - Pesos de GPT-2 desde Hugging Face (safetensors), sin TensorFlow.
#   - Funciones del cap. 6 renombradas con sufijo _cls para no chocar con las del 5.
#
# Uso rápido (desde la terminal):
#   python model_base.py selftest                      # comprobaciones sin descargas
#   python model_base.py pretrain --data tinyshakespeare --tokenizer char --out-dir runs/base
#   python model_base.py generate --ckpt runs/base/model.pt --prompt "ROMEO:"
#   python model_base.py classify --out-dir runs/spam  # fine-tuning del cap. 6

import argparse
import json
import math
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#####################################
# 0. Configuraciones
#####################################

# Configuración de GPT-2 small (124M) tal y como aparece en el cap. 4
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Tamaño del vocabulario (BPE de GPT-2)
    "context_length": 1024,  # Longitud máxima de contexto
    "emb_dim": 768,          # Dimensión de los embeddings
    "n_heads": 12,           # Número de cabezas de atención
    "n_layers": 12,          # Número de bloques transformer
    "drop_rate": 0.1,        # Dropout que se usa en el entrenamiento
    "qkv_bias": False        # Sesgo en las proyecciones Q, K, V
}

# Tamaños de GPT-2 cuyos pesos se pueden cargar en el cap. 5
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def get_gpt2_config(model_name="gpt2-small (124M)"):
    """Configuración para cargar los pesos preentrenados de OpenAI (cap. 5).

    Los pesos originales usan context_length=1024 y qkv_bias=True; el dropout
    se desactiva porque solo se va a hacer inferencia o fine-tuning ligero.
    """
    cfg = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    cfg.update(MODEL_CONFIGS[model_name])
    return cfg


def get_small_config(vocab_size, context_length=128, emb_dim=128, n_heads=4,
                     n_layers=4, drop_rate=0.1, qkv_bias=False):
    """Configuración diminuta para experimentos que convergen en minutos."""
    return {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "drop_rate": drop_rate,
        "qkv_bias": qkv_bias,
    }


def get_device(name=None):
    """Dispositivo explícito (p. ej. "cpu", "cuda") o el mejor disponible."""
    if isinstance(name, torch.device):
        return name
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#####################################
# 1. Capítulo 2: tokenizadores y datos
#####################################
class CharTokenizer:
    """Tokenizador a nivel de carácter con la interfaz mínima de tiktoken.

    Sirve para modelos diminutos (vocabulario de decenas de tokens) y para tareas
    sintéticas. Se construye a partir de un texto: CharTokenizer.from_text(txt).
    """

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def n_vocab(self):
        return len(self.chars)

    def encode(self, text, allowed_special=None):
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f"Carácter fuera del vocabulario: {e}") from None

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def get_tokenizer(name, text=None):
    """"gpt2" -> BPE de GPT-2 (tiktoken); "char" -> CharTokenizer (requiere text)."""
    if name == "gpt2":
        return tiktoken.get_encoding("gpt2")
    if name == "char":
        if text is None:
            raise ValueError("El tokenizador 'char' necesita el texto para crear el vocabulario")
        return CharTokenizer.from_text(text)
    raise ValueError(f"Tokenizador desconocido: {name}")


def tokenizer_to_dict(tokenizer):
    """Serializa el tokenizador (para guardarlo dentro de un checkpoint)."""
    if isinstance(tokenizer, CharTokenizer):
        return {"type": "char", "chars": "".join(tokenizer.chars)}
    return {"type": "gpt2"}


def tokenizer_from_dict(d):
    if d is None:
        return None
    if d["type"] == "char":
        return CharTokenizer(d["chars"])
    return tiktoken.get_encoding("gpt2")


# Textos de ejemplo que se pueden descargar por nombre
DATA_URLS = {
    # Texto corto del cap. 2 y 5 del libro (~20 KB)
    "verdict": "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
    # Tiny Shakespeare (~1,1 MB), sugerido en el enunciado para validar
    "tinyshakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def download_file(url, dest, timeout=60):
    """Descarga `url` en `dest` (no hace nada si ya existe)."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "model_base.py"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)
    return dest


def load_text(source, data_dir="data"):
    """Lee un texto: `source` es un nombre de DATA_URLS (se descarga) o una ruta."""
    if source in DATA_URLS:
        path = download_file(DATA_URLS[source], Path(data_dir) / f"{source}.txt")
    else:
        path = Path(source)
    return path.read_text(encoding="utf-8")


class GPTDatasetV1(Dataset):
    """Trocea un texto en ventanas deslizantes (entrada, objetivo desplazado 1)."""

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokeniza todo el texto
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Ventana deslizante: secuencias de max_length con solapamiento (stride)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """Versión original del libro (siempre usa el BPE de GPT-2)."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers)
    return dataloader


def create_dataloader(txt, tokenizer, batch_size=4, max_length=256, stride=128,
                      shuffle=True, drop_last=True, num_workers=0, generator=None):
    """Como create_dataloader_v1 pero con tokenizador y generador aleatorio propios."""
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, generator=generator)


def make_train_val_loaders(text, tokenizer, context_length, batch_size,
                           train_ratio=0.9, stride=None, seed=123):
    """Divide el texto en entrenamiento/validación (por caracteres, como el libro).

    El orden de los lotes de entrenamiento depende solo de `seed`, no del resto
    del programa: así se compara el mismo flujo de datos entre modelos.
    """
    stride = context_length if stride is None else stride
    split_idx = int(train_ratio * len(text))
    train_loader = create_dataloader(
        text[:split_idx], tokenizer, batch_size=batch_size, max_length=context_length,
        stride=stride, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    val_loader = create_dataloader(
        text[split_idx:], tokenizer, batch_size=batch_size, max_length=context_length,
        stride=stride, shuffle=False, drop_last=False)
    if len(train_loader) == 0:
        raise ValueError("No hay suficientes tokens para entrenamiento: reduce context_length o batch_size")
    if len(val_loader) == 0:
        raise ValueError("No hay suficientes tokens para validación: reduce context_length o train_ratio")
    return train_loader, val_loader


#####################################
# 2. Capítulo 3: atención multi-cabeza causal
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimensión por cabeza

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Combina la salida de las cabezas
        self.dropout = nn.Dropout(dropout)
        # Máscara causal: 1 por encima de la diagonal (posiciones futuras)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Separa las cabezas: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Puntuaciones de atención: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # Máscara causal recortada a la longitud actual y pasada a booleano
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Concatena las cabezas: d_out = num_heads * head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# 3. Capítulo 4: arquitectura GPT
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Aproximación con tanh de GELU
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Bloque pre-norm: LayerNorm -> atención -> residual; LayerNorm -> FFN -> residual."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Conexión residual del bloque de atención
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # (batch_size, num_tokens, emb_dim)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Conexión residual del bloque feed-forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # posiciones absolutas aprendidas
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # (batch_size, num_tokens, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)    # (batch_size, num_tokens, vocab_size)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generación voraz (greedy). idx tiene forma (B, T)."""
    for _ in range(max_new_tokens):
        # Recorta el contexto si supera el máximo soportado
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Solo interesa el último paso: (batch, n_token, vocab) -> (batch, vocab)
        logits = logits[:, -1, :]

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)                # (batch, n_tokens+1)

    return idx


#####################################
# 4. Capítulo 5: preentrenamiento, generación y carga de pesos
#####################################
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)  # añade la dimensión de batch


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # quita la dimensión de batch
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """Entropía cruzada media. Los objetivos con valor -100 se ignoran (por defecto
    en cross_entropy), lo que permite enmascarar la parte del prompt si se quiere."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context,
                              context_size=None, max_new_tokens=50):
    """Genera y muestra un texto de ejemplo.

    `context_size` es opcional: si no se indica se toma de `model.pos_emb` como en
    el libro. Los modelos sin `pos_emb` deben pasarlo explícitamente.
    """
    model.eval()
    if context_size is None:
        context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=max_new_tokens, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, context_size=None):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reinicia los gradientes del batch anterior
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()        # Calcula los gradientes
            optimizer.step()       # Actualiza los pesos
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Muestra un texto generado al final de cada época (si hay prompt)
        if start_context:
            generate_and_print_sample(model, tokenizer, device, start_context,
                                      context_size=context_size)

    return train_losses, val_losses, track_tokens_seen


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Generación con temperatura, top-k y parada opcional por token EOS."""
    if eos_id is not None and idx.shape[0] != 1:
        raise ValueError("EOS stopping currently supports batch size 1 only")

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Top-k: deja solo los k logits mayores
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Temperatura: si es > 0 se muestrea; si no, se toma el máximo
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values  # estabilidad numérica
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:  # Parada temprana si aparece el token de fin
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """Copia los pesos de GPT-2 (dict `params` de OpenAI) al GPTModel.

    Requiere qkv_bias=True en la configuración. `params` se obtiene con
    `load_gpt2_params` (más abajo, sin TensorFlow) o con `download_and_load_gpt2`
    de ch05/01_main-chapter-code/gpt_download.py del repositorio original.
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # En GPT-2 Q, K y V vienen concatenadas en una sola matriz c_attn
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # GPT-2 comparte los pesos de la capa de salida con los embeddings de tokens
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# --- Pesos de GPT-2 desde Hugging Face (safetensors), sin TensorFlow ---------
HF_GPT2_REPOS = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl",
}


def hf_state_to_gpt2_params(state):
    """Convierte un state dict de GPT-2 de Hugging Face (nombre -> array numpy) al
    diccionario `params` que espera `load_weights_into_gpt`.

    Las matrices de las capas Conv1D de Hugging Face ya tienen la orientación
    (entrada, salida) de los pesos originales de OpenAI, así que no se transponen.
    Acepta nombres con o sin el prefijo "transformer.".
    """
    def get(name):
        for key in (name, "transformer." + name):
            if key in state:
                return np.asarray(state[key])
        sample = ", ".join(list(state.keys())[:5])
        raise KeyError(f"No se encontró '{name}' en los pesos. Claves de ejemplo: {sample}")

    n_layers = 0
    while any(k in state for k in (f"h.{n_layers}.ln_1.weight", f"transformer.h.{n_layers}.ln_1.weight")):
        n_layers += 1
    if n_layers == 0:
        get("h.0.ln_1.weight")  # lanza el error informativo

    blocks = []
    for i in range(n_layers):
        p = f"h.{i}."
        blocks.append({
            "attn": {
                "c_attn": {"w": get(p + "attn.c_attn.weight"), "b": get(p + "attn.c_attn.bias")},
                "c_proj": {"w": get(p + "attn.c_proj.weight"), "b": get(p + "attn.c_proj.bias")},
            },
            "mlp": {
                "c_fc": {"w": get(p + "mlp.c_fc.weight"), "b": get(p + "mlp.c_fc.bias")},
                "c_proj": {"w": get(p + "mlp.c_proj.weight"), "b": get(p + "mlp.c_proj.bias")},
            },
            "ln_1": {"g": get(p + "ln_1.weight"), "b": get(p + "ln_1.bias")},
            "ln_2": {"g": get(p + "ln_2.weight"), "b": get(p + "ln_2.bias")},
        })
    return {
        "wpe": get("wpe.weight"),
        "wte": get("wte.weight"),
        "blocks": blocks,
        "g": get("ln_f.weight"),
        "b": get("ln_f.bias"),
    }


def load_gpt2_params(model_name="gpt2-small (124M)", models_dir="gpt2_weights"):
    """Descarga (si hace falta) los pesos de GPT-2 de Hugging Face y devuelve `params`.

    Necesita el paquete `safetensors` y acceso a huggingface.co.
    """
    from safetensors.numpy import load_file

    repo = HF_GPT2_REPOS[model_name]
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
    dest = Path(models_dir) / repo.split("/")[-1] / "model.safetensors"
    if not dest.exists():
        print(f"Descargando pesos de {model_name} desde {url} ...")
    download_file(url, dest, timeout=120)
    return hf_state_to_gpt2_params(load_file(str(dest)))


def load_pretrained_gpt2(model_name="gpt2-small (124M)", models_dir="gpt2_weights"):
    """Crea un GPTModel con los pesos preentrenados de GPT-2. Devuelve (model, cfg)."""
    cfg = get_gpt2_config(model_name)
    model = GPTModel(cfg)
    load_weights_into_gpt(model, load_gpt2_params(model_name, models_dir))
    model.eval()
    return model, cfg


#####################################
# 5. Utilidades de experimentación
#####################################
def save_checkpoint(path, model, cfg, tokenizer=None, extra=None):
    """Guarda pesos + configuración + tokenizador. `extra` debe contener solo
    tipos simples (números, cadenas, listas, diccionarios)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": dict(cfg),
        "tokenizer": tokenizer_to_dict(tokenizer) if tokenizer is not None else None,
        "extra": extra,
    }, path)


def load_checkpoint(path, model_factory=None, device="cpu"):
    """Carga un checkpoint. Devuelve (model, cfg, tokenizer)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = (model_factory or GPTModel)(ckpt["cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, ckpt["cfg"], tokenizer_from_dict(ckpt["tokenizer"])


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path=None):
    """Pérdida de entrenamiento y validación frente a épocas (y tokens vistos)."""
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()  # segundo eje x con los tokens vistos
    ax2.plot(tokens_seen, train_losses, alpha=0)  # curva invisible para alinear las marcas
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss", save_path=None):
    """Igual que plot_losses para el cap. 6 (pérdida o precisión frente a ejemplos vistos)."""
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def calc_perplexity(loss):
    """Perplejidad a partir de una entropía cruzada media (en nats)."""
    return math.exp(loss)


def benchmark_generation(model, vocab_size, context_size, device=None, prompt_len=32,
                         new_tokens=64, batch_size=1, generate_fn=None, n_runs=3):
    """Mide tokens generados por segundo (y memoria máxima en GPU, si la hay).

    `generate_fn(model, idx, max_new_tokens, context_size)` permite medir variantes
    con otra forma de generar (por defecto, generate_text_simple).
    """
    generate_fn = generate_fn or generate_text_simple
    device = get_device(device) if device is not None else next(model.parameters()).device
    model.eval()
    idx = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    generate_fn(model, idx, 2, context_size)  # calentamiento
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        generate_fn(model, idx, new_tokens, context_size)
        if is_cuda:
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    return {
        "tokens_per_s": batch_size * new_tokens / min(times),
        "seconds_per_run": sum(times) / len(times),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if is_cuda else None,
    }


def pretrain(cfg, text, tokenizer, model_factory=None, num_epochs=10, batch_size=2,
             learning_rate=5e-4, weight_decay=0.1, train_ratio=0.9, stride=None,
             eval_freq=5, eval_iter=1, start_context="Every effort moves you",
             seed=123, device=None, out_dir=None):
    """Preentrena un modelo (cap. 5) y devuelve un diccionario con resultados.

    Para evaluar una ampliación basta con llamar a pretrain() con el mismo `cfg`,
    texto, tokenizador y semilla, pasando su clase en `model_factory`.
    """
    device = get_device(device)
    torch.manual_seed(seed)
    model = (model_factory or GPTModel)(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, val_loader = make_train_val_loaders(
        text, tokenizer, cfg["context_length"], batch_size, train_ratio, stride, seed)

    t0 = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=start_context, tokenizer=tokenizer, context_size=cfg["context_length"])
    elapsed = time.time() - t0

    # Evaluación final sobre todos los lotes
    final_train, final_val = evaluate_model(model, train_loader, val_loader, device, eval_iter=None)
    train_tokens = num_epochs * len(train_loader) * batch_size * cfg["context_length"]
    results = {
        "train_losses": train_losses, "val_losses": val_losses, "tokens_seen": tokens_seen,
        "final_train_loss": final_train, "final_val_loss": final_val,
        "val_perplexity": calc_perplexity(final_val),
        "n_params": count_parameters(model),
        "train_time_s": elapsed, "train_tokens_per_s": train_tokens / elapsed,
        "seed": seed, "cfg": dict(cfg),
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out_dir / "model.pt", model, cfg, tokenizer)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if train_losses:
            epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
            fig = plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses,
                              save_path=out_dir / "loss.png")
            import matplotlib.pyplot as plt
            plt.close(fig)

    results["model"] = model
    return results


#####################################
# 6. Capítulo 6: fine-tuning para clasificación
#####################################
class SpamDataset(Dataset):
    """Dataset de clasificación (CSV con columnas "Text" y "Label"), con padding."""

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        import pandas as pd  # solo hace falta para el cap. 6

        self.data = pd.read_csv(csv_file)

        # Tokeniza los textos de antemano
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Trunca las secuencias que superen max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Rellena con pad_token_id hasta la longitud máxima
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


def adapt_for_classification(model, emb_dim, num_classes=2):
    """Prepara el modelo para clasificar (cap. 6).

    Congela todo, sustituye la cabeza de salida por una lineal de num_classes
    salidas y vuelve a habilitar el entrenamiento del último bloque
    transformer y de la normalización final.
    """
    for param in model.parameters():
        param.requires_grad = False

    model.out_head = torch.nn.Linear(in_features=emb_dim, out_features=num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    return model


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # logits del último token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch_cls(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # logits del último token
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader_cls(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch_cls(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model_cls(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_cls(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader_cls(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch_cls(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]  # ahora se cuentan ejemplos, no tokens
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_cls(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Precisión al final de cada época
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# --- Datos del cap. 6: SMS Spam Collection -----------------------------------
SPAM_URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    # Copia alternativa que usa el repositorio del libro
    "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip",
]


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    data_file_path = Path(data_file_path)
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Descarga el zip
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    download_file(url, zip_path)

    # Descomprime
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Añade la extensión .tsv
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    import pandas as pd

    # Cuenta las instancias de "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Submuestrea aleatoriamente "ham" para igualar el número de "spam"
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combina el subconjunto de "ham" con "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    # Mezcla todo el DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calcula los índices de corte
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Divide el DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def prepare_spam_data(data_dir="sms_spam_collection"):
    """Descarga el dataset, lo equilibra y lo divide 70/10/20 en train/validation/test.csv.

    Devuelve las tres rutas. Etiquetas: 0 = "ham" (no spam), 1 = "spam".
    """
    import pandas as pd

    data_dir = Path(data_dir)
    train_csv, val_csv, test_csv = (data_dir / f"{n}.csv" for n in ("train", "validation", "test"))
    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        return train_csv, val_csv, test_csv

    data_file_path = data_dir / "SMSSpamCollection.tsv"
    last_error = None
    for url in SPAM_URLS:
        try:
            download_and_unzip_spam_data(url, data_dir / "sms_spam_collection.zip", data_dir, data_file_path)
            break
        except Exception as e:  # prueba la siguiente URL
            last_error = e
            print(f"Falló la descarga desde {url}: {e}")
    else:
        raise RuntimeError(f"No se pudo descargar el dataset de spam: {last_error}")

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv(train_csv, index=None)
    validation_df.to_csv(val_csv, index=None)
    test_df.to_csv(test_csv, index=None)
    return train_csv, val_csv, test_csv


def finetune_classifier(model, train_csv, val_csv, test_csv, tokenizer, device=None,
                        num_epochs=5, batch_size=8, learning_rate=5e-5, weight_decay=0.1,
                        eval_freq=50, eval_iter=5, pad_token_id=50256, context_size=None,
                        seed=123, out_dir=None):
    """Fine-tuning para clasificación (cap. 6) de un modelo ya adaptado con
    adapt_for_classification. Devuelve un diccionario con pérdidas y precisiones."""
    device = get_device(device)
    model.to(device)
    if context_size is None:
        context_size = model.pos_emb.weight.shape[0]

    train_dataset = SpamDataset(train_csv, tokenizer, max_length=None, pad_token_id=pad_token_id)
    if train_dataset.max_length > context_size:
        print(f"Textos más largos que el contexto ({train_dataset.max_length} > {context_size}): se truncan")
        train_dataset = SpamDataset(train_csv, tokenizer, max_length=context_size, pad_token_id=pad_token_id)
    val_dataset = SpamDataset(val_csv, tokenizer, max_length=train_dataset.max_length, pad_token_id=pad_token_id)
    test_dataset = SpamDataset(test_csv, tokenizer, max_length=train_dataset.max_length, pad_token_id=pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    t0 = time.time()
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter)
    elapsed = time.time() - t0

    results = {
        "train_losses": train_losses, "val_losses": val_losses,
        "train_accs": train_accs, "val_accs": val_accs, "examples_seen": examples_seen,
        "train_accuracy": calc_accuracy_loader(train_loader, model, device),
        "val_accuracy": calc_accuracy_loader(val_loader, model, device),
        "test_accuracy": calc_accuracy_loader(test_loader, model, device),
        "max_length": train_dataset.max_length,
        "trainable_params": count_parameters(model, trainable_only=True),
        "train_time_s": elapsed,
    }
    print(f"Training accuracy: {results['train_accuracy']*100:.2f}%")
    print(f"Validation accuracy: {results['val_accuracy']*100:.2f}%")
    print(f"Test accuracy: {results['test_accuracy']*100:.2f}%")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        import matplotlib.pyplot as plt
        for values, label in (((train_losses, val_losses), "loss"), ((train_accs, val_accs), "accuracy")):
            epochs_tensor = torch.linspace(0, num_epochs, len(values[0]))
            seen_tensor = torch.linspace(0, examples_seen, len(values[0]))
            fig = plot_values(epochs_tensor, seen_tensor, values[0], values[1], label=label,
                              save_path=out_dir / f"{label}.png")
            plt.close(fig)
    return results


def classify_text(text, model, tokenizer, device=None, max_length=None,
                  pad_token_id=50256, context_size=None):
    """Clasifica un texto con un modelo ajustado (cap. 6). Devuelve "spam" o "not spam"."""
    device = get_device(device) if device is not None else next(model.parameters()).device
    model.eval()
    if context_size is None:
        context_size = model.pos_emb.weight.shape[0]
    input_ids = tokenizer.encode(text)
    limit = context_size if max_length is None else min(max_length, context_size)
    input_ids = input_ids[:limit]
    if max_length is not None:
        input_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # añade la dimensión de batch
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # logits del último token
    return "spam" if torch.argmax(logits, dim=-1).item() == 1 else "not spam"


#####################################
# 7. Autotest y línea de comandos
#####################################
def run_selftest():
    """Comprobaciones que no necesitan descargas ni GPU. Devuelve True si todo pasa."""
    import contextlib
    import io
    import tempfile

    outcomes = []

    def check(name, cond, detail=""):
        outcomes.append(bool(cond))
        print(f"[{'OK' if cond else 'FALLO'}] {name} {detail}".rstrip())

    def quiet():
        return contextlib.redirect_stdout(io.StringIO())

    torch.manual_seed(123)

    # 1) Arquitectura: parámetros, formas y causalidad
    n = count_parameters(GPTModel(GPT_CONFIG_124M))
    check("GPTModel con GPT_CONFIG_124M tiene 163.009.536 parámetros", n == 163_009_536, f"({n:,})")

    cfg = get_small_config(vocab_size=100, context_length=16, emb_dim=32, n_heads=4,
                           n_layers=2, drop_rate=0.0)
    model = GPTModel(cfg).eval()
    x = torch.randint(0, 100, (3, 10))
    logits = model(x)
    check("Forma de los logits (batch, tokens, vocab)", tuple(logits.shape) == (3, 10, 100))
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % 100
    check("Causalidad: cambiar el último token no altera los anteriores",
          torch.allclose(logits[:, :-1], model(x2)[:, :-1], atol=1e-6))

    # 2) Piezas frente a PyTorch
    att = MultiHeadAttention(32, 32, 16, 0.0, 4).eval()
    h = torch.randn(2, 10, 32)
    b, t, _ = h.shape
    q = att.W_query(h).view(b, t, 4, 8).transpose(1, 2)
    k = att.W_key(h).view(b, t, 4, 8).transpose(1, 2)
    v = att.W_value(h).view(b, t, 4, 8).transpose(1, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = att.out_proj(ref.transpose(1, 2).reshape(b, t, 32))
    check("MultiHeadAttention == scaled_dot_product_attention de PyTorch",
          torch.allclose(att(h), ref, atol=1e-5))
    z = torch.randn(4, 32)
    check("LayerNorm == torch.nn.functional.layer_norm",
          torch.allclose(LayerNorm(32)(z), torch.nn.functional.layer_norm(z, (32,), eps=1e-5), atol=1e-5))
    check("GELU == gelu(approximate='tanh')",
          torch.allclose(GELU()(z), torch.nn.functional.gelu(z, approximate="tanh"), atol=1e-5))

    # 3) Generación y pérdida enmascarada
    out = generate_text_simple(model, x[:1, :4], 5, cfg["context_length"])
    out_t = generate(model, x[:1, :4], 5, cfg["context_length"], temperature=1.0, top_k=5)
    check("generate_text_simple / generate producen 5 tokens nuevos",
          out.shape[1] == 9 and out_t.shape[1] == 9)
    targets = x.clone()
    masked = targets.clone()
    masked[:, :5] = -100
    full_logits = model(x)
    manual = torch.nn.functional.cross_entropy(full_logits[:, 5:].reshape(-1, 100), targets[:, 5:].reshape(-1))
    check("calc_loss_batch ignora los objetivos con valor -100",
          torch.allclose(calc_loss_batch(x, masked, model, "cpu"), manual, atol=1e-6))

    # 4) Preentrenamiento extremo a extremo con tokenizador de caracteres
    text = "abcde fghij " * 200
    tok = CharTokenizer.from_text(text)
    check("CharTokenizer: decode(encode(t)) == t", tok.decode(tok.encode(text[:50])) == text[:50])
    pcfg = get_small_config(tok.n_vocab, context_length=16, emb_dim=32, n_heads=4, n_layers=2, drop_rate=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        kwargs = dict(num_epochs=15, batch_size=8, learning_rate=3e-3, eval_freq=20, eval_iter=2,
                      start_context="abcde", device="cpu", seed=7)
        with quiet():
            res1 = pretrain(pcfg, text, tok, out_dir=tmp, **kwargs)
            res2 = pretrain(pcfg, text, tok, **kwargs)
        check("pretrain(): la pérdida de validación baja de ln(vocab) a < 0.2",
              res1["final_val_loss"] < 0.2,
              f"(val {res1['final_val_loss']:.3f}, ppl {res1['val_perplexity']:.2f})")
        check("pretrain(): reproducible con la misma semilla",
              abs(res1["final_val_loss"] - res2["final_val_loss"]) < 1e-6)
        ckpt_model, ckpt_cfg, ckpt_tok = load_checkpoint(Path(tmp) / "model.pt")
        a = res1["model"].eval()(x[:, :8] % tok.n_vocab)
        c = ckpt_model.eval()(x[:, :8] % tok.n_vocab)
        check("Checkpoint: guardar y cargar da los mismos logits", torch.allclose(a, c, atol=1e-6)
              and ckpt_tok.chars == tok.chars)
        check("pretrain(out_dir) crea model.pt, metrics.json y loss.png",
              all((Path(tmp) / f).exists() for f in ("model.pt", "metrics.json", "loss.png")))
    bench = benchmark_generation(model, 100, cfg["context_length"], device="cpu",
                                 prompt_len=4, new_tokens=8, n_runs=1)
    check("benchmark_generation devuelve tokens/s > 0", bench["tokens_per_s"] > 0)

    # 5) Carga de pesos de GPT-2: conversión desde el formato de Hugging Face
    rng = np.random.default_rng(0)
    d, layers, vocab, ctx = 32, 2, 50, 16
    r = lambda *s: rng.standard_normal(s).astype(np.float32)
    state = {"wte.weight": r(vocab, d), "wpe.weight": r(ctx, d), "ln_f.weight": r(d), "ln_f.bias": r(d)}
    for i in range(layers):
        p = f"h.{i}."
        state.update({
            p + "ln_1.weight": r(d), p + "ln_1.bias": r(d), p + "ln_2.weight": r(d), p + "ln_2.bias": r(d),
            p + "attn.c_attn.weight": r(d, 3 * d), p + "attn.c_attn.bias": r(3 * d),
            p + "attn.c_proj.weight": r(d, d), p + "attn.c_proj.bias": r(d),
            p + "mlp.c_fc.weight": r(d, 4 * d), p + "mlp.c_fc.bias": r(4 * d),
            p + "mlp.c_proj.weight": r(4 * d, d), p + "mlp.c_proj.bias": r(d),
        })
    gcfg = get_small_config(vocab, ctx, d, 4, layers, 0.0, qkv_bias=True)
    for prefix in ("", "transformer."):
        gpt = GPTModel(gcfg)
        load_weights_into_gpt(gpt, hf_state_to_gpt2_params({prefix + k: v for k, v in state.items()}))
        q_w = state["h.1.attn.c_attn.weight"][:, :d].T
        check(f"Conversión de pesos de Hugging Face (prefijo '{prefix}')",
              np.allclose(gpt.trf_blocks[1].att.W_query.weight.detach().numpy(), q_w)
              and np.allclose(gpt.out_head.weight.detach().numpy(), state["wte.weight"]))
    try:
        from safetensors.numpy import load_file, save_file
        with tempfile.TemporaryDirectory() as tmp:
            save_file(state, str(Path(tmp) / "m.safetensors"))
            params = hf_state_to_gpt2_params(load_file(str(Path(tmp) / "m.safetensors")))
            check("Conversión de pesos desde un archivo safetensors", len(params["blocks"]) == layers)
    except ImportError:
        print("[--] safetensors no instalado: se omite la prueba del archivo")

    # 5b) Cap. 6 con un dataset sintético (sin descargas)
    try:
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            spam = [f"FREE prize win now {i}" for i in range(40)]
            ham = [f"ok see you at {i}" for i in range(40)]
            df = pd.DataFrame({"Text": spam + ham, "Label": [1] * 40 + [0] * 40}).sample(frac=1, random_state=0)
            ctok = CharTokenizer.from_text("".join(df["Text"]))
            paths = []
            for name, part in (("train", df[:56]), ("val", df[56:68]), ("test", df[68:])):
                path = Path(tmp) / f"{name}.csv"
                part.to_csv(path, index=None)
                paths.append(path)
            ccfg = get_small_config(ctok.n_vocab, 64, 32, 4, 2, 0.0)
            torch.manual_seed(1)
            cmodel = adapt_for_classification(GPTModel(ccfg), ccfg["emb_dim"], 2)
            with quiet():
                cres = finetune_classifier(cmodel, *paths, ctok, device="cpu", num_epochs=10, batch_size=4,
                                           learning_rate=1e-3, eval_freq=10, eval_iter=3, pad_token_id=0,
                                           context_size=64)
            check("Cap. 6: el clasificador aprende un dataset separable", cres["test_accuracy"] >= 0.9,
                  f"(test acc {cres['test_accuracy']:.2f})")
            label = classify_text("FREE prize win now 5", cmodel, ctok, device="cpu", max_length=cres["max_length"],
                                  pad_token_id=0, context_size=64)
            check("classify_text devuelve 'spam' o 'not spam'", label in ("spam", "not spam"), f"({label})")
    except ImportError:
        print("[--] pandas no instalado: se omite la prueba del cap. 6")

    ok = all(outcomes)
    print(f"\n{sum(outcomes)}/{len(outcomes)} comprobaciones correctas" + ("" if ok else " (hay fallos)"))
    return ok


# Hiperparámetros por defecto de cada preajuste de `pretrain`
PRESETS = {
    # Modelo diminuto: converge en minutos con Tiny Shakespeare a nivel de carácter
    "small": dict(context_length=128, emb_dim=128, n_heads=4, n_layers=4, drop_rate=0.1,
                  batch_size=32, learning_rate=1e-3, num_epochs=5, eval_freq=50, eval_iter=5),
    # GPT-2 small con contexto 256, como en gpt_train.py del libro
    "gpt2": dict(context_length=256, emb_dim=768, n_heads=12, n_layers=12, drop_rate=0.1,
                 batch_size=2, learning_rate=5e-4, num_epochs=10, eval_freq=5, eval_iter=1),
}


def demo():
    """Prueba mínima del cap. 4: modelo sin entrenar generando texto sin sentido."""
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # desactiva el dropout

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_tensor = text_to_token_ids(start_context, tokenizer)

    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output length:", len(out[0]))
    print("Output text:", token_ids_to_text(out, tokenizer))
    print(f"Parámetros totales: {count_parameters(model):,}")


def _cmd_pretrain(args):
    text = load_text(args.data, args.data_dir)
    if args.max_chars:
        text = text[:args.max_chars]
    tokenizer = get_tokenizer(args.tokenizer, text)

    hp = dict(PRESETS[args.preset])
    for key in hp:
        value = getattr(args, key, None)
        if value is not None:
            hp[key] = value
    cfg = {"vocab_size": tokenizer.n_vocab, "qkv_bias": False}
    cfg.update({k: hp[k] for k in ("context_length", "emb_dim", "n_heads", "n_layers", "drop_rate")})

    prompt = args.prompt
    try:
        tokenizer.encode(prompt)
    except ValueError:
        prompt = text[:10]
    print(f"Texto: {len(text):,} caracteres | tokenizador: {args.tokenizer} (vocab {cfg['vocab_size']}) | "
          f"dispositivo: {get_device(args.device)}")
    res = pretrain(cfg, text, tokenizer, num_epochs=hp["num_epochs"], batch_size=hp["batch_size"],
                   learning_rate=hp["learning_rate"], weight_decay=args.weight_decay,
                   eval_freq=hp["eval_freq"], eval_iter=hp["eval_iter"], start_context=prompt,
                   seed=args.seed, device=args.device, out_dir=args.out_dir)
    print(f"\nParámetros: {res['n_params']:,} | val loss: {res['final_val_loss']:.3f} | "
          f"perplejidad: {res['val_perplexity']:.2f} | tiempo: {res['train_time_s']:.0f} s")
    if args.out_dir:
        print(f"Resultados guardados en {args.out_dir}/ (model.pt, metrics.json, loss.png)")


def _cmd_generate(args):
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    model, cfg, tokenizer = load_checkpoint(args.ckpt, device=device)
    model.eval()
    idx = text_to_token_ids(args.prompt, tokenizer).to(device)
    out = generate(model, idx, args.max_new_tokens, cfg["context_length"],
                   temperature=args.temperature, top_k=args.top_k)
    print(token_ids_to_text(out.cpu(), tokenizer))


def _cmd_classify(args):
    device = get_device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_csv, val_csv, test_csv = prepare_spam_data(args.data_dir)
    if args.test_mode:
        # Modelo diminuto sin pesos preentrenados, solo para comprobar el flujo
        cfg = get_small_config(50257, context_length=120, emb_dim=12, n_heads=2, n_layers=1, drop_rate=0.0)
        torch.manual_seed(args.seed)
        model = GPTModel(cfg)
    else:
        model, cfg = load_pretrained_gpt2(args.model, args.models_dir)
    model = adapt_for_classification(model, cfg["emb_dim"], num_classes=2)
    res = finetune_classifier(model, train_csv, val_csv, test_csv, tokenizer, device=device,
                              num_epochs=args.epochs, batch_size=args.batch_size,
                              learning_rate=args.learning_rate, seed=args.seed, out_dir=args.out_dir,
                              context_size=cfg["context_length"])
    for text in ("You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
                 "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"):
        print(f"{classify_text(text, model, tokenizer, device, res['max_length'], context_size=cfg['context_length'])!r:>12} <- {text[:60]}...")
    if args.out_dir:
        torch.save(model.state_dict(), Path(args.out_dir) / "classifier.pth")


def build_parser():
    p = argparse.ArgumentParser(description="Modelo base GPT (caps. 2-6 de LLMs-from-scratch)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="Genera texto con un modelo sin entrenar (cap. 4)")
    sub.add_parser("selftest", help="Comprobaciones internas sin descargas")

    pt = sub.add_parser("pretrain", help="Preentrena un modelo desde cero (cap. 5)")
    pt.add_argument("--data", default="tinyshakespeare",
                    help="'verdict', 'tinyshakespeare' o ruta a un .txt (por defecto: tinyshakespeare)")
    pt.add_argument("--data-dir", default="data", help="Carpeta donde se guardan las descargas")
    pt.add_argument("--max-chars", type=int, default=None, help="Usa solo los primeros N caracteres")
    pt.add_argument("--tokenizer", choices=["char", "gpt2"], default="char")
    pt.add_argument("--preset", choices=list(PRESETS), default="small")
    for name, typ in (("context_length", int), ("emb_dim", int), ("n_heads", int), ("n_layers", int),
                      ("drop_rate", float), ("batch_size", int), ("learning_rate", float),
                      ("num_epochs", int), ("eval_freq", int), ("eval_iter", int)):
        pt.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None,
                        help="Sustituye el valor del preajuste")
    pt.add_argument("--epochs", dest="num_epochs", type=int, default=None, help="Alias de --num-epochs")
    pt.add_argument("--lr", dest="learning_rate", type=float, default=None, help="Alias de --learning-rate")
    pt.add_argument("--weight-decay", type=float, default=0.1)
    pt.add_argument("--prompt", default="Every effort moves you")
    pt.add_argument("--seed", type=int, default=123)
    pt.add_argument("--device", default=None)
    pt.add_argument("--out-dir", default=None, help="Guarda model.pt, metrics.json y loss.png")

    g = sub.add_parser("generate", help="Genera texto con un checkpoint")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--prompt", default="Every effort moves you")
    g.add_argument("--max-new-tokens", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--seed", type=int, default=123)
    g.add_argument("--device", default=None)

    c = sub.add_parser("classify", help="Fine-tuning para clasificar spam con GPT-2 (cap. 6)")
    c.add_argument("--model", choices=list(MODEL_CONFIGS), default="gpt2-small (124M)")
    c.add_argument("--models-dir", default="gpt2_weights")
    c.add_argument("--data-dir", default="sms_spam_collection")
    c.add_argument("--epochs", type=int, default=5)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--learning-rate", type=float, default=5e-5)
    c.add_argument("--seed", type=int, default=123)
    c.add_argument("--device", default=None)
    c.add_argument("--out-dir", default=None)
    c.add_argument("--test-mode", action="store_true",
                   help="Modelo diminuto sin pesos preentrenados para probar el flujo rápidamente")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        demo()
    elif args.command == "selftest":
        raise SystemExit(0 if run_selftest() else 1)
    elif args.command == "pretrain":
        _cmd_pretrain(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "classify":
        _cmd_classify(args)


if __name__ == "__main__":
    run_selftest()
