# pe_labels.py
#
# Etiquetas de codificación posicional para los scripts de experimentos
# (train_eval.py y context_ppl.py). Añade al conjunto de positional.PE_KINDS las
# variantes híbridas RoPE/NoPE por capas:
#
#   "rope_nopeN"   RoPE en todas las capas salvo las capas N, 2N, 3N, ... (contando desde 1),
#                  que no llevan ninguna codificación posicional (NoPE).
#                  Con 4 capas: rope_nope4 -> solo la capa 4 sin RoPE (1 de cada 4);
#                               rope_nope2 -> capas 2 y 4 sin RoPE (la mitad).
#
# El híbrido ya estaba implementado en positional.py / model_ext.py (opción nope_every);
# este módulo solo traduce la etiqueta a la configuración:
#   parse_pe_label("rope_nope4")  ->  ("rope", {"nope_every": 4})
#   parse_pe_label("alibi")       ->  ("alibi", {})
#
# Las etiquetas antiguas ("learned", "rope", ...) se interpretan igual que antes, así que
# los resultados ya obtenidos siguen siendo válidos y no hace falta repetirlos.
#
#   python pe_labels.py selftest

import argparse

from positional import PE_KINDS

HYBRID_PREFIX = "rope_nope"


def parse_pe_label(label):
    """Devuelve (pos_encoding, opciones_extra_de_la_configuración)."""
    label = label.lower()
    if label in PE_KINDS:
        return label, {}
    if label.startswith(HYBRID_PREFIX):
        suffix = label[len(HYBRID_PREFIX):]
        if suffix.isdigit() and int(suffix) >= 1:
            return "rope", {"nope_every": int(suffix)}
    raise ValueError(f"Codificación posicional desconocida: {label!r}. Opciones: {', '.join(PE_KINDS)} "
                     f"o {HYBRID_PREFIX}N con N >= 1 (por ejemplo, {HYBRID_PREFIX}4)")


def pe_label_type(label):
    """Para usar como `type=` en argparse."""
    try:
        parse_pe_label(label)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return label


def nope_layers(label, n_layers):
    """Capas (contando desde 1) sin codificación posicional en una etiqueta híbrida."""
    every = parse_pe_label(label)[1].get("nope_every")
    return [i for i in range(1, n_layers + 1) if every and i % every == 0]


def check_layers(label, n_layers):
    """Error claro si el híbrido no deja ninguna capa NoPE con ese número de capas."""
    extra = parse_pe_label(label)[1]
    if extra and not nope_layers(label, n_layers):
        raise ValueError(f"{label} con {n_layers} capas no tiene ninguna capa NoPE "
                         f"(sería igual que 'rope'). Usa N <= {n_layers}.")


def _hybrid_n(label):
    try:
        return parse_pe_label(label)[1].get("nope_every")
    except ValueError:
        return None


def ordered_labels(labels):
    """Orden de presentación: primero las de PE_KINDS, luego los híbridos (menos NoPE
    primero) y al final cualquier etiqueta no reconocida."""
    labels = set(labels)
    base = [k for k in PE_KINDS if k in labels]
    hybrids = sorted((l for l in labels if l not in PE_KINDS and _hybrid_n(l)), key=lambda l: -_hybrid_n(l))
    others = sorted(l for l in labels if l not in PE_KINDS and not _hybrid_n(l))
    return base + hybrids + others


def run_selftest():
    outcomes = []

    def check(name, cond, detail=""):
        outcomes.append(bool(cond))
        print(f"[{'OK' if cond else 'FALLO'}] {name} {detail}".rstrip())

    check("Las etiquetas de PE_KINDS se interpretan igual que antes",
          all(parse_pe_label(k) == (k, {}) for k in PE_KINDS))
    check("rope_nope4 -> ('rope', {'nope_every': 4}) y no distingue mayúsculas",
          parse_pe_label("rope_nope4") == ("rope", {"nope_every": 4}) and parse_pe_label("ROPE_NOPE2")[1] == {"nope_every": 2})
    for bad in ("rope_nope", "rope_nope0", "rope_nopeX", "rope_nope-1", "xyz", "nope_rope4"):
        try:
            parse_pe_label(bad)
            check(f"Etiqueta inválida rechazada: {bad!r}", False)
        except ValueError:
            check(f"Etiqueta inválida rechazada: {bad!r}", True)
    try:
        pe_label_type("xyz")
        check("pe_label_type lanza ArgumentTypeError (para argparse)", False)
    except argparse.ArgumentTypeError:
        check("pe_label_type lanza ArgumentTypeError (para argparse)", True)
    check("pe_label_type devuelve la etiqueta si es válida", pe_label_type("rope_nope4") == "rope_nope4")
    check("nope_layers: rope_nope4 con 4 capas -> [4]; rope_nope2 -> [2, 4]; rope_nope3 con 8 capas -> [3, 6]",
          nope_layers("rope_nope4", 4) == [4] and nope_layers("rope_nope2", 4) == [2, 4]
          and nope_layers("rope_nope3", 8) == [3, 6] and nope_layers("rope", 4) == [])
    check("check_layers acepta N <= capas y rechaza N > capas",
          check_layers("rope_nope4", 4) is None and check_layers("rope", 4) is None
          and _raises(lambda: check_layers("rope_nope5", 4)))
    check("ordered_labels: PE_KINDS en su orden, luego híbridos con menos NoPE primero",
          ordered_labels(["rope_nope2", "rope", "t5", "rope_nope4", "learned", "zzz"])
          == ["learned", "rope", "t5", "rope_nope4", "rope_nope2", "zzz"])

    ok = all(outcomes)
    print(f"\n{sum(outcomes)}/{len(outcomes)} comprobaciones correctas" + ("" if ok else " (hay fallos)"))
    return ok


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(0 if run_selftest() else 1)
    print("Uso: python pe_labels.py selftest")
    print(f"Etiquetas: {', '.join(PE_KINDS)} y {HYBRID_PREFIX}N (híbrido RoPE/NoPE; N = una capa de cada N sin RoPE)")
