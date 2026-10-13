# tasks.py
#
# Tareas sintéticas para medir la generalización de longitud de las codificaciones
# posicionales (Kazemnejad et al., 2023, arXiv:2305.19466): se entrena con entradas
# de longitud 1..L y se evalúa con longitudes mayores, que el modelo no ha visto.
#
# Cada ejemplo es una secuencia  <bos> prompt <sep> respuesta <eos>  de un modelo de
# solo decodificador. La pérdida se calcula solo sobre la respuesta y el <eos>
# (los objetivos del prompt valen -100, que model_base.calc_loss_batch ignora).
#
# El parámetro `n` es el tamaño del problema ("longitud"):
#   copy        copiar n dígitos
#   reverse     invertir n dígitos
#   sort        ordenar n dígitos de menor a mayor
#   sum_mod     suma de n dígitos módulo 10 (respuesta de 1 dígito)
#   parity      paridad de n bits (respuesta de 1 dígito)
#   addition    suma de dos números de n dígitos
#   assoc_recall  n pares clave-valor y una consulta: devolver el valor de la clave
#                 (tarea de memoria asociativa que sugiere el enunciado de la práctica)
#
# Las muestras de evaluación son deterministas: para una tarea y una longitud dadas son
# SIEMPRE las mismas, para todos los modelos y semillas de entrenamiento.
#
#   python tasks.py selftest        # comprobaciones
#   python tasks.py show copy 5     # un ejemplo legible

import random
from dataclasses import dataclass
from typing import Callable

import torch

IGNORE_INDEX = -100   # valor que cross_entropy ignora (lo usa model_base.calc_loss_batch)
NUM_KEYS = 64         # claves distintas de assoc_recall (n máximo = 64)


#####################################
# Vocabulario
#####################################
class TaskVocab:
    """Vocabulario único para todas las tareas: especiales, dígitos, '+' y claves."""

    def __init__(self, num_keys=NUM_KEYS):
        specials = ["<pad>", "<bos>", "<sep>", "<eos>", "<query>"]
        self.tokens = (specials + [str(d) for d in range(10)] + ["+"]
                       + [f"K{i}" for i in range(num_keys)])
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.sep_id, self.eos_id, self.query_id = range(5)
        self.plus_id = self.stoi["+"]
        self.num_keys = num_keys

    @property
    def size(self):
        return len(self.tokens)

    def digit_id(self, d):
        return self.stoi[str(d)]

    def digit_value(self, token_id):
        return int(self.tokens[token_id])

    def key_id(self, i):
        return self.stoi[f"K{i}"]

    def decode(self, ids):
        return " ".join(self.tokens[int(i)] for i in ids)


VOCAB = TaskVocab()


#####################################
# Tareas
#####################################
@dataclass(frozen=True)
class Task:
    name: str
    description: str
    sample: Callable            # (n, rng, vocab) -> (prompt_ids, answer_ids)
    prompt_len: Callable        # n -> longitud del prompt (sin <bos> ni <sep>)
    max_answer_len: Callable    # n -> longitud máxima de la respuesta (sin <eos>)


def _digits(rng, n, v):
    return [v.digit_id(rng.randrange(10)) for _ in range(n)]


def _sample_copy(n, rng, v):
    d = _digits(rng, n, v)
    return d, list(d)


def _sample_reverse(n, rng, v):
    d = _digits(rng, n, v)
    return d, d[::-1]


def _sample_sort(n, rng, v):
    d = _digits(rng, n, v)
    return d, sorted(d)  # los ids de los dígitos están ordenados como sus valores


def _sample_sum_mod(n, rng, v):
    d = _digits(rng, n, v)
    return d, [v.digit_id(sum(v.digit_value(i) for i in d) % 10)]


def _sample_parity(n, rng, v):
    bits = [rng.randrange(2) for _ in range(n)]
    return [v.digit_id(b) for b in bits], [v.digit_id(sum(bits) % 2)]


def _number(rng, n):
    """n dígitos sin cero a la izquierda (salvo el número de 1 dígito)."""
    first = rng.randrange(1, 10) if n > 1 else rng.randrange(10)
    return [first] + [rng.randrange(10) for _ in range(n - 1)]


def _sample_addition(n, rng, v):
    a, b = _number(rng, n), _number(rng, n)
    total = int("".join(map(str, a))) + int("".join(map(str, b)))
    prompt = [v.digit_id(x) for x in a] + [v.plus_id] + [v.digit_id(x) for x in b]
    return prompt, [v.digit_id(int(c)) for c in str(total)]


def _sample_assoc_recall(n, rng, v):
    if n > v.num_keys:
        raise ValueError(f"assoc_recall admite como mucho n={v.num_keys} pares")
    keys = rng.sample(range(v.num_keys), n)
    values = [rng.randrange(10) for _ in range(n)]
    q = rng.randrange(n)
    prompt = []
    for key, value in zip(keys, values):
        prompt += [v.key_id(key), v.digit_id(value)]
    prompt += [v.query_id, v.key_id(keys[q])]
    return prompt, [v.digit_id(values[q])]


TASKS = {t.name: t for t in [
    Task("copy", "Copiar n dígitos", _sample_copy, lambda n: n, lambda n: n),
    Task("reverse", "Invertir n dígitos", _sample_reverse, lambda n: n, lambda n: n),
    Task("sort", "Ordenar n dígitos de menor a mayor", _sample_sort, lambda n: n, lambda n: n),
    Task("sum_mod", "Suma de n dígitos módulo 10", _sample_sum_mod, lambda n: n, lambda n: 1),
    Task("parity", "Paridad de n bits", _sample_parity, lambda n: n, lambda n: 1),
    Task("addition", "Suma de dos números de n dígitos", _sample_addition,
         lambda n: 2 * n + 1, lambda n: n + 1),
    Task("assoc_recall", "Memoria asociativa con n pares clave-valor", _sample_assoc_recall,
         lambda n: 2 * n + 2, lambda n: 1),
]}


def max_total_length(task_name, n):
    """Longitud máxima de la secuencia completa <bos> prompt <sep> respuesta <eos>.
    Sirve para dimensionar la tabla de la APE aprendida."""
    t = TASKS[task_name]
    return 1 + t.prompt_len(n) + 1 + t.max_answer_len(n) + 1


#####################################
# Ejemplos, lotes y conjuntos de evaluación
#####################################
def encode_example(prompt, answer, v=VOCAB):
    """Devuelve (secuencia, índice del <sep>)."""
    seq = [v.bos_id] + prompt + [v.sep_id] + answer + [v.eos_id]
    return seq, len(prompt) + 1


def collate(examples, v=VOCAB):
    """Agrupa ejemplos [(secuencia, índice_sep)] en (entradas, objetivos) con relleno.

    El modelo predice seq[t+1] a partir de seq[:t+1]; solo cuentan los objetivos que
    son parte de la respuesta o el <eos> (t >= índice del <sep>). El resto vale -100.
    El relleno va a la derecha, así que no afecta a las posiciones anteriores.
    """
    width = max(len(seq) for seq, _ in examples) - 1
    inputs = torch.full((len(examples), width), v.pad_id, dtype=torch.long)
    targets = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    for i, (seq, sep) in enumerate(examples):
        inputs[i, :len(seq) - 1] = torch.tensor(seq[:-1])
        targets[i, sep:len(seq) - 1] = torch.tensor(seq[1:][sep:])
    return inputs, targets


def sample_batch(task_name, batch_size, min_len, max_len, rng, v=VOCAB):
    """Lote de entrenamiento con longitudes uniformes en [min_len, max_len]."""
    task = TASKS[task_name]
    examples = []
    for _ in range(batch_size):
        n = rng.randint(min_len, max_len)
        prompt, answer = task.sample(n, rng, v)
        examples.append(encode_example(prompt, answer, v))
    return collate(examples, v)


def make_eval_set(task_name, n, num_samples, seed=0, v=VOCAB):
    """Prompts (con <bos> y <sep>, todos de la misma longitud) y respuestas esperadas.
    Determinista: depende solo de (seed, tarea, n)."""
    rng = random.Random(f"eval-{seed}-{task_name}-{n}")
    task = TASKS[task_name]
    prompts, answers = [], []
    for _ in range(num_samples):
        prompt, answer = task.sample(n, rng, v)
        prompts.append([v.bos_id] + prompt + [v.sep_id])
        answers.append(answer)
    return torch.tensor(prompts, dtype=torch.long), answers


def check_generation(new_tokens, answers, v=VOCAB):
    """Exact match: la generación (tokens tras el <sep>) debe ser la respuesta esperada
    seguida de <eos>. Devuelve una lista de booleanos."""
    results = []
    for row, answer in zip(new_tokens.tolist(), answers):
        results.append(v.eos_id in row and row[:row.index(v.eos_id)] == answer)
    return results


def show_example(task_name, n, seed=0, v=VOCAB):
    """Ejemplo legible (prompt -> respuesta), útil para la memoria."""
    prompts, answers = make_eval_set(task_name, n, 1, seed, v)
    return f"{v.decode(prompts[0])} -> {v.decode(answers[0])} {v.tokens[v.eos_id]}"


#####################################
# Autotest: python tasks.py selftest
#####################################
def run_selftest():
    outcomes = []

    def check(name, cond, detail=""):
        outcomes.append(bool(cond))
        print(f"[{'OK' if cond else 'FALLO'}] {name} {detail}".rstrip())

    v = VOCAB
    rng = random.Random(0)
    check("Vocabulario: ids especiales 0..4 y tamaño", (v.pad_id, v.bos_id, v.sep_id, v.eos_id, v.query_id) == (0, 1, 2, 3, 4)
          and v.size == 5 + 10 + 1 + NUM_KEYS, f"(tamaño {v.size})")

    # Corrección de las respuestas, comprobadas con código independiente
    ok = {name: True for name in TASKS}
    for n in (1, 2, 5, 17, 40):
        for _ in range(30):
            digits = lambda ids: [v.digit_value(i) for i in ids]
            p, a = TASKS["copy"].sample(n, rng, v)
            ok["copy"] &= digits(a) == digits(p)
            p, a = TASKS["reverse"].sample(n, rng, v)
            ok["reverse"] &= digits(a) == digits(p)[::-1]
            p, a = TASKS["sort"].sample(n, rng, v)
            ok["sort"] &= digits(a) == sorted(digits(p)) and sorted(a) == a
            p, a = TASKS["sum_mod"].sample(n, rng, v)
            ok["sum_mod"] &= digits(a) == [sum(digits(p)) % 10]
            p, a = TASKS["parity"].sample(n, rng, v)
            ok["parity"] &= digits(a) == [sum(digits(p)) % 2] and set(digits(p)) <= {0, 1}
            p, a = TASKS["addition"].sample(n, rng, v)
            plus = p.index(v.plus_id)
            x = int("".join(map(str, digits(p[:plus]))))
            y = int("".join(map(str, digits(p[plus + 1:]))))
            ok["addition"] &= int("".join(map(str, digits(a)))) == x + y and plus == n and len(p) - plus - 1 == n
            ok["addition"] &= len(a) <= n + 1 and (n == 1 or (p[0] != v.digit_id(0) and p[plus + 1] != v.digit_id(0)))
            if n <= NUM_KEYS:
                p, a = TASKS["assoc_recall"].sample(n, rng, v)
                pairs = {p[2 * i]: p[2 * i + 1] for i in range(n)}
                ok["assoc_recall"] &= len(pairs) == n and a == [pairs[p[-1]]] and p[-2] == v.query_id
    for name, passed in ok.items():
        check(f"Tarea '{name}': respuestas correctas (n = 1, 2, 5, 17, 40)", passed)

    check("Longitudes: prompt_len y max_answer_len coinciden con las muestras",
          all(len(TASKS[t].sample(n, rng, v)[0]) == TASKS[t].prompt_len(n)
              and len(TASKS[t].sample(n, rng, v)[1]) <= TASKS[t].max_answer_len(n)
              for t in TASKS for n in (1, 4, 9)))
    try:
        TASKS["assoc_recall"].sample(NUM_KEYS + 1, rng, v)
        check("assoc_recall: error claro si n supera el número de claves", False)
    except ValueError:
        check("assoc_recall: error claro si n supera el número de claves", True)

    # Lotes: alineación de entradas y objetivos, y enmascarado del prompt
    inputs, targets = sample_batch("addition", 16, 1, 6, random.Random(1), v)
    aligned, masked = True, True
    for row_in, row_tg in zip(inputs, targets):
        seq_len = int((row_in != v.pad_id).sum()) + 1  # longitud de la secuencia completa
        sep = int((row_in == v.sep_id).nonzero()[0])
        valid = (row_tg != IGNORE_INDEX).nonzero().flatten().tolist()
        masked &= valid == list(range(sep, seq_len - 1))
        aligned &= all(row_tg[t] == (row_in[t + 1] if t + 1 < seq_len - 1 else v.eos_id) for t in valid)
        aligned &= int(row_tg[seq_len - 2]) == v.eos_id
    check("Lotes: solo la respuesta y el <eos> tienen objetivo (el prompt vale -100)", masked)
    check("Lotes: objetivo[t] == secuencia[t+1] y el último es <eos>", aligned)
    check("Lotes: el relleno (pad) no tiene objetivo",
          bool(((inputs == v.pad_id) & (targets != IGNORE_INDEX)).sum() == 0))
    lens = [int((r != v.pad_id).sum()) + 1 for r in sample_batch("copy", 200, 3, 7, random.Random(2), v)[0]]
    check("sample_batch: longitudes de copy en [3, 7] (secuencia = 2n + 3)",
          min(lens) == 2 * 3 + 3 and max(lens) == 2 * 7 + 3)

    # Evaluación determinista y exact match
    p1, a1 = make_eval_set("sort", 12, 20, seed=0)
    p2, a2 = make_eval_set("sort", 12, 20, seed=0)
    p3, _ = make_eval_set("sort", 12, 20, seed=1)
    check("Conjunto de evaluación: determinista y dependiente de la semilla",
          torch.equal(p1, p2) and a1 == a2 and not torch.equal(p1, p3))
    check("Conjunto de evaluación: prompts de igual longitud con <bos> ... <sep>",
          p1.shape == (20, 14) and bool((p1[:, 0] == v.bos_id).all()) and bool((p1[:, -1] == v.sep_id).all()))
    width = 13  # 12 dígitos de respuesta + <eos>
    good = torch.tensor([a + [v.eos_id] for a in a1[:3]])
    wrong_digit = good[0].clone()
    wrong_digit[0] = v.digit_id((v.digit_value(int(wrong_digit[0])) + 1) % 10)
    no_eos = torch.tensor([a1[0] + [v.digit_id(0)]])
    early_eos = torch.tensor([a1[0][:-1] + [v.eos_id, v.eos_id]])
    check("check_generation: exige la respuesta exacta seguida de <eos>",
          good.shape[1] == width and check_generation(good, a1[:3]) == [True] * 3
          and check_generation(wrong_digit.unsqueeze(0), a1[:1]) == [False]
          and check_generation(no_eos, a1[:1]) == [False]
          and check_generation(early_eos, a1[:1]) == [False])
    check("max_total_length coincide con la secuencia real",
          all(max_total_length(t, n) >= len(encode_example(*TASKS[t].sample(n, rng, v), v)[0])
              for t in TASKS for n in (1, 7, 20))
          and max_total_length("copy", 20) == 43 and max_total_length("addition", 5) == 1 + 11 + 1 + 6 + 1)

    ok_all = all(outcomes)
    print(f"\n{sum(outcomes)}/{len(outcomes)} comprobaciones correctas" + ("" if ok_all else " (hay fallos)"))
    return ok_all


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(0 if run_selftest() else 1)
    if len(sys.argv) > 2 and sys.argv[1] == "show":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        print(show_example(sys.argv[2], n))
        raise SystemExit(0)
    print("Uso: python tasks.py selftest | python tasks.py show <tarea> [n]")
    for t in TASKS.values():
        print(f"  {t.name:13s} {t.description}")
