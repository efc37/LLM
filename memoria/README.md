# Memoria (formato ACL)

- `memoria.tex`: el texto. Máximo 4 páginas sin contar las referencias.
- `referencias.bib`: la bibliografía.
- `acl.sty`, `acl_natbib.bst`: estilo oficial de ACL, sin modificar (https://github.com/acl-org/acl-style-files).
- `figuras/fig_principal.pdf`: la figura 1. Se genera con `python scripts/figures_memoria.py` desde la raíz del repo, a partir de los resultados guardados.
- `memoria.pdf`: la versión compilada que se entrega.

## Compilar

Cualquiera de estas opciones:
- **Tectonic** (compilador LaTeX autocontenido, sin instalación): `tectonic memoria.tex`
- **Overleaf:** subir toda la carpeta `memoria/` y compilar `memoria.tex` con pdfLaTeX.
- **TeX Live / MiKTeX:** `pdflatex memoria && bibtex memoria && pdflatex memoria && pdflatex memoria`

## De dónde sale cada cifra

| Elemento | Origen |
|---|---|
| Tabla 1 (codificaciones, perplejidad) | `runs/context_ppl_all/results.jsonl` y `runs/context_hybrid/results.jsonl` (del equipo) |
| Tabla 2 (tareas sintéticas) | `runs/length/results.jsonl` (del equipo) |
| Tabla 3 (validación) | `tests/` (`pytest`), las autocomprobaciones `src/*.py selftest` y `src/validate_rope_hf.py` |
| Tabla 4 (MoE) | `results/summary.md` y `results/router_stats.md` (`scripts/run_all.py`) |
| Tabla 5 y figura 1a/1b (combinación) | `runs/context_moe/summary.md` (`scripts/context_ppl_moe.py`) |
| Figura 1c | `results/summary.json` |

Antes de entregar hay que completar los nombres que faltan en la línea de autores (aparecen en rojo).
