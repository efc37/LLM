# CLAUDE.md — Práctica 1 de PLN I (Universitat d'Alacant)

## Contexto
Práctica grupal (4 personas) de Procesamiento del Lenguaje Natural I, Grado en Ingeniería en IA.
- **Base:** código de los capítulos 2–6 del libro *Build a Large Language Model (From Scratch)* de Sebastian Raschka (tokenización, atención, GPT, preentrenamiento, clasificación). Ya está en este repo.
- **Objetivo:** ampliar ese modelo con técnicas de LLMs modernos que NO estén en el código base, elegidas de la galería de arquitecturas de Raschka (hasta 3 ampliaciones según complejidad).
- **Restricción del profesor:** RoPE está descartado (lo ha hecho toda la clase).
- **Entrega:** solo un PDF por Moodle. **Fecha límite: 22 de octubre de 2026.**
- **Memoria:** formato ACL, máximo 4 páginas sin contar referencias (plantilla LaTeX recomendada).
- **Examen de prácticas:** habrá preguntas sobre la implementación base. Quien trabaje aquí debe ENTENDER el código, no solo tenerlo.

## Evaluación (lo que importa)
- La nota depende de la ambición y complejidad técnica. Un cambio trivial (p. ej. solo cambiar una función de activación) no aprueba por sí solo.
- Máxima nota: cambios que reformulen la interacción entre varios módulos, con transformaciones tensoriales no triviales y una batería empírica exhaustiva.
- Es obligatorio aportar evidencia experimental rigurosa de que la ampliación está bien implementada y produce el efecto esperado.

## Vías de validación aceptadas
1. Si la técnica existe en PyTorch/librerías estándar: comparar numéricamente las salidas con una misma entrada (con `torch.allclose` y tolerancia).
2. Optimizaciones estructurales: comparativa reproducible de memoria o tokens/s frente al modelo base.
3. Entrenamiento desde cero: modelos muy pequeños que converjan en minutos (p. ej. Tiny Shakespeare).
4. Modelo grande (GPT-2): fine-tuning acotado con LoRA o few-shot in-context learning.
5. Tareas sintéticas, p. ej. associative recall (`A: 3, B: 7, C: 1, ... QUERY B ->` debe dar alta probabilidad a `7`).

## Estado del proyecto (rellenar / verificar)
- Ampliaciones elegidas por el equipo: a 13-oct-2026 no hay ninguna empezada en el repo (solo `src/ampliacion_X.py` y `experimentos/0X_.ipynb` vacíos como esqueleto). Base hecha: `src/model_base.py` (caps. 2–6) y notebooks `base/ch02–ch06` (ch02–ch03 comentados).
- Reparto de tareas entre miembros: _(pendiente de confirmar con el equipo)_
- Mi parte como colaborador: **Mixture of Experts (MoE)** en la rama `feat/moe`. Ver bitácora abajo. Extra opcional (solo si MoE está cerrado ~19-oct y nadie lo ha cogido): GQA + KV cache + atención de ventana deslizante.

## Bitácora de la ampliación MoE (rama `feat/moe`)
Plan aprobado el 13-oct-2026. Todo el código nuevo va en `src/moe.py` (no se toca `src/model_base.py`, que es compartido); tests en `tests/test_moe.py` (`python -m pytest tests/test_moe.py -q`). Se activa por config: si `cfg` no tiene la clave `"moe"`, es el modelo base.

```python
cfg["moe"] = {"n_experts": 4, "top_k": 2, "aux_loss_coef": 0.01,
              "expert_hidden": 4*emb_dim, "normalize_topk": None, "dispatch": "sorted"}
```
Notación: B batch, T tokens, D emb_dim, E expertos, k expertos activos por token, N = B·T.

**Estado de los bloques**
| # | Bloque | Estado | Commit |
|---|---|---|---|
| 0 | Entorno: `pip install tiktoken matplotlib pytest`; `python src/model_base.py` reproduce el texto del libro | hecho | — |
| 1 | `TopKRouter` | hecho | 6116e99 |
| 2 | `Expert`, `MoEFeedForward` con despacho denso de referencia, opción `normalize_topk` | hecho | f3f8ecf |
| 3 | Despachos `loop` y `sorted` (eficiente, por defecto) | hecho | 1caeca1 |
| 4 | Pérdida auxiliar de balanceo `load_balancing_loss` | hecho | e8d59b7 |
| 5 | `MoETransformerBlock` + `GPTMoEModel` + `build_model` + `upcycle_from_dense` | hecho | ver git log |
| 6 | Tiny Shakespeare (char-level) + `scripts/train_shakespeare.py`, resultados en `results/` | pendiente | |
| 7 | `scripts/run_all.py`: denso vs MoE, ablación coef. aux {0, 0.01, 0.1}, tokens/s, uso de expertos, especialización | pendiente | |
| 8 | Sección de la memoria | pendiente | |

**Bloque 1 — Router top-k.** `gate = Linear(D, E, bias=False)`: x (N,D) → logits (N,E) → softmax `probs` (N,E) → `topk` → `indices` (N,k) y `weights` (N,k). `probs` se guarda para la pérdida auxiliar. `topk` no es derivable en los índices: el router aprende a través de los pesos (y de la pérdida auxiliar).

**Bloque 2 — Capa MoE y referencia densa.**
- `normalize_topk`: con k>1 se renormalizan los k pesos para que sumen 1 (Mixtral); con k=1 se usa la probabilidad tal cual (Switch). Motivo (descubierto y testeado): con k=1 renormalizar da peso = p/p = 1 constante ⇒ el router no recibe gradiente de la pérdida principal (medido: 19.6 sin renormalizar vs 2.3e-6 renormalizado, que es solo redondeo).
- `Expert`: mismo FFN del libro (Linear D→H, GELU, Linear H→D) con H configurable; conserva `self.layers` para poder copiar pesos de `FeedForward` con `load_state_dict`.
- `MoEFeedForward.forward`: (B,T,D) → (N,D) → router → despacho → (B,T,D). Guarda `last_probs` (N,E) y `last_indices` (N,k).
- Despacho `dense` (referencia, coste E·FFN): `all_out` (N,E,D) con todos los expertos sobre todos los tokens; `gates` (N,E) = `scatter` de `weights` en `indices` (0 fuera del top-k); salida = `(gates.unsqueeze(-1) * all_out).sum(1)`.
- Tests: E=1,k=1 con pesos de `FeedForward` ⇒ `allclose` (atol 1e-6), con y sin renormalizar; k=E ⇒ igual a la mezcla Σ pₑ·Eₑ(x) calculada token a token; shapes.

**Bloque 3 — Despacho eficiente.**
- `sorted`: aplanar asignaciones `flat_experts`, `flat_weights`, `token_ids = arange(N).repeat_interleave(k)` (todos (N·k,)) → `argsort(flat_experts, stable=True)` → `bincount` da `counts` (E,), que suma N·k → `x_flat[sorted_tokens]` (N·k,D) → `split(counts)` en E bloques contiguos → cada experto procesa su bloque → `cat` → × peso → `index_add_` en (N,D). Se usa `index_add_` (y no asignación) porque con k≥2 un token recibe k resultados que hay que sumar.
- `loop`: por experto, `torch.where(indices == e)` da (tokens, posición en el top-k) y se acumula con `index_add_` (como Mixtral en Hugging Face).
- Tests: salida, gradiente de la entrada y de todos los parámetros iguales a `dense` (atol 1e-5), 3 semillas × {E4k1, E8k2, E8k2 sin normalizar, E4k4}; caso con expertos sin tokens.
- Tiempo orientativo en CPU (4096 tokens, D=128, fwd+bwd, NO guardado, solo indicativo): FFN denso 25.8 ms; E4k1 dense/loop/sorted 129.0/30.8/28.1 ms; E8k2 232.6/50.1/48.1 ms. El coste escala con k, no con E. La medición reproducible irá en el bloque 7.

**Bloque 4 — Pérdida auxiliar (Switch Transformer).** `L_aux = E · Σᵢ fᵢ·Pᵢ`; fᵢ = fracción de las N·k asignaciones al experto i (de `bincount`, NO derivable), Pᵢ = `probs.mean(0)` (derivable; por aquí entra el gradiente). Vale 1 con reparto uniforme y E con colapso total. Si fᵢ = 1/E, L = Σ Pᵢ = 1 sean cuales sean las Pᵢ.
- Tests: reparto perfecto ⇒ 1; colapso ⇒ E; router uniforme ⇒ 1; optimizando solo L_aux un router sesgado pasa de >90 % de tokens en un experto a <35 % (≈25 % = 1/E), 3 semillas.
- Hallazgos (útiles para la memoria): (1) si el router está saturado (P₀ = 1.0) el gradiente del softmax desaparece (~3e-3) y L_aux no lo recupera (se queda en 4 = E) → motivación de la router z-loss; (2) con lr grande (0.5) la carga salta en bloque a otro experto en vez de repartirse (fᵢ es discontinua y un router lineal mueve juntos los tokens parecidos); con lr ≤ 0.1 se reparte bien.

**Bloque 5 — Integración en el GPT.**
- `MoETransformerBlock(TransformerBlock)`: solo cambia `self.ff` por `MoEFeedForward`; el forward del libro (pre-norm, residuales, dropout) se hereda.
- `GPTMoEModel(GPTModel)`: reconstruye `trf_blocks`; con `every_n_layers = n` solo uno de cada n bloques es MoE (el n-ésimo, como Switch/GShard). `moe_layers()`, `aux_loss()` = **media** de L_aux sobre capas MoE (así el coeficiente no depende de la profundidad y 1 = equilibrado), `expert_stats()` = asignaciones (L_moe, E) del último forward.
- `build_model(cfg)`: `GPTModel` si no hay `"moe"` en cfg, `GPTMoEModel` si lo hay.
- `upcycle_from_dense(moe_model, dense_model)` (*sparse upcycling*, Komatsuzaki et al. 2023): copia todos los pesos y replica el FFN denso en cada experto (renombrando `trf_blocks.i.ff.experts.j.layers.X` → `trf_blocks.i.ff.layers.X`); el router mantiene su inicialización. Con expertos idénticos y pesos que suman 1, Σ wⱼ·FFN(x) = FFN(x).
- Tests (de extremo a extremo, logits `allclose` atol 1e-5 con `GPTModel`): E=1; E=4 k=2; E=4 k=1 renormalizado; E=4 k=2 alternando capas con despacho `loop`. Control negativo: E=4 k=1 sin renormalizar (Switch) NO coincide (salida p·FFN(x)). También: alternancia de bloques, shapes de `expert_stats` (suma por capa = N·k), y el gradiente de CE + 0.01·L_aux llega al router de todas las capas.
- Escala (cálculo analítico para GPT-2 124M, 163.0M parámetros con la cabeza sin compartir; FFN por capa 4.72M): E=4 k=1 → 333.1M totales / 163.0M activos por token; E=8 k=2 → 559.8M / 219.8M. Es la idea central del MoE: más parámetros sin más cómputo por token.

**Preguntas de repaso (tipo examen)**
- E=8, k=2, batch 4×64: shapes de `probs`, `indices`, `weights`. ¿Y de `flat_experts`, `x_sorted`, `counts`? ¿Cuánto suma `counts`?
- ¿Por qué `topk` no deja pasar gradiente por los índices? ¿Por dónde aprende el router?
- ¿Qué pasa con k=1 si se renormalizan los pesos? ¿Cómo lo resuelve Switch?
- ¿Por qué en el despacho denso hace falta `gates.unsqueeze(-1)`?
- ¿Por qué `index_add_` y no `out[sorted_tokens] = y_sorted`?
- ¿Por qué con k=1 el MoE cuesta casi lo mismo que el FFN denso con 4× parámetros?
- ¿Por qué el gradiente de L_aux pasa por Pᵢ y no por fᵢ? ¿Cuánto vale L_aux si fᵢ = 1/E?
- ¿Por qué un router saturado no se recupera con L_aux? ¿Qué hace la z-loss?
- En `MoETransformerBlock`, ¿dónde se aplica el MoE respecto a `norm2` y a la residual? ¿Cambia algo del flujo del bloque del libro?
- Con expertos idénticos (upcycling), ¿por qué E=4 k=2 reproduce el modelo denso y E=4 k=1 estilo Switch no?
- GPT-2 124M con E=4 k=1: ¿por qué los parámetros totales casi se duplican pero los activos por token no cambian?

## Cómo quiero que trabajes
1. **Primero explora, no escribas.** Al empezar: resume la estructura del repo, qué está hecho, qué falta, y revisa `git log` y las ramas. No toques nada todavía.
2. **Plan antes de código.** Propón un plan concreto y espera mi OK antes de implementar. Si hay varias opciones, compáralas brevemente y recomienda una.
3. **Pasos pequeños.** Implementa de una pieza en una (un módulo, una función), explicando en pocas líneas qué hace, por qué, y las formas de los tensores (shapes) en cada paso importante.
4. **Prueba cada paso.** Cada cambio lleva un test o un script de comprobación (equivalencia numérica con PyTorch, comprobación de shapes, o un caso mínimo). No des nada por bueno sin ejecutarlo.
5. **Mantén el comportamiento base intacto.** Las ampliaciones deben activarse por configuración (flag o clase aparte) para poder comparar siempre contra el modelo base.
6. **Experimentos reproducibles.** Fija semillas, guarda configuraciones y resultados (métricas, tiempos, memoria) en ficheros, y que los scripts se puedan relanzar con un solo comando. Los resultados deben poder citarse tal cual en la memoria.
7. **No inventes resultados.** Si algo no se ha ejecutado, dilo. Las cifras de la memoria salen solo de ejecuciones reales.
8. **Modelos pequeños y rápidos.** No hay cómputo masivo: prioriza experimentos que acaben en minutos.

## Reglas de colaboración (git)
- Trabaja siempre en una rama propia (`feat/<nombre-ampliacion>`), nunca directamente en `main`.
- No reescribas ni reformatees ficheros que no sean de mi parte; hay más gente tocando el repo.
- Commits pequeños y con mensaje claro. Antes de integrar, avísame de posibles conflictos.
- No borres ni hagas `push --force` sin que yo lo confirme.

## Aprendizaje (importante)
- Al terminar cada bloque, hazme 2–3 preguntas cortas sobre lo que acabamos de implementar, tipo examen, para comprobar que lo he entendido.
- Si te pido que me expliques una parte del código base (caps. 2–6), hazlo de lo concreto a lo general y señala qué es lo que suele preguntarse (shapes, máscara causal, escalado por √d, residuales, LayerNorm, pérdida, etc.).

## Memoria (4 páginas, ACL)
Cuando toque, ayúdame a estructurarla: introducción breve, descripción de la ampliación (con la formulación y los shapes), metodología de validación, resultados (tablas/gráficas comparando con el modelo base), conclusiones y limitaciones. Concisa y con cifras reales.
