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
- Ampliaciones elegidas por el equipo (actualizado 16-oct-2026, commit `c257b2d` de aos35 en `main`, 13-oct): **codificaciones posicionales y generalización de longitud** (Kazemnejad et al., 2023, arXiv:2305.19466; PDF `NoPE.pdf` en la raíz).
  - `src/positional.py`: variantes intercambiables `learned` (la del libro), `sinusoidal`, `nope` (sin codificación), `rope`, `alibi`, `t5` (sesgo relativo); híbrido **RoPE/NoPE por capas** (`nope_every=N`: las capas N, 2N… sin codificación; etiquetas `rope_nopeN` en `src/pe_labels.py`).
  - `src/model_base_adaptado.py`: copia ampliada del base (CharTokenizer, `pretrain`, dataloaders con semilla, pérdida que ignora -100). `src/model_ext.py`: `GPTModelExt`/`TransformerBlockExt` (su bloque tiene `self.ff = FeedForward(cfg)` como el del libro → el MoE se puede enchufar ahí).
  - Experimentos: `src/tasks.py` + `src/train_eval.py` (tareas sintéticas copy/reverse/sort/sum_mod/parity/addition/assoc_recall, entrenar con longitudes cortas y evaluar más largas) y `src/context_ppl.py` (perplejidad en Tiny Shakespeare con contextos de 32 a 1024 entrenando con 128). Resultados en `runs/`. `src/validate_rope_hf.py` valida RoPE contra Hugging Face (requiere `transformers`).
  - Su configuración pequeña coincide casi con la mía (emb 128, 4 capas, 4 cabezas, ctx 128, batch 32, lr 1e-3, wd 0.1; ellos drop 0.1, yo 0.0) y usan el mismo `tinyshakespeare.txt` (mismo SHA-1).
  - Ojo: el profesor descartó RoPE; ellos lo usan como variante y en el híbrido. Confirmar con el equipo que el enfoque (NoPE/ALiBi/T5/híbrido y generalización de longitud) es lo que presentan.
- Integración: el 16-oct se fusionó `main` en `feat/moe` en local (merge `31ebcc2`, sin conflictos: ningún fichero en común; `model_base.py` solo cambió comentarios). Tras la fusión: mis 61 tests y sus 101 autocomprobaciones (`positional`, `model_ext`, `pe_labels`, `tasks`) pasan.
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
| 6 | Tiny Shakespeare (char-level) + `scripts/train_shakespeare.py` + `configs/` + `calc_loss_batch_moe` | hecho | ver git log |
| 7 | `scripts/run_all.py`: denso vs MoE, ablación coef. aux {0, 0.01, 0.1}, tokens/s, uso de expertos, especialización | hecho | b0dac07, 5374c23 |
| 8 | Sección de la memoria | pendiente | |
| 9 | Unir MoE al modelo del equipo (`GPTModelExt`): MoE + codificaciones posicionales | propuesto (16-oct) | |

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

**Bloque 6 — Entrenamiento en Tiny Shakespeare.**
- `calc_loss_batch_moe(x, y, model, device, aux_loss_coef)` en `src/moe.py`: como `calc_loss_batch` del cap. 5 pero devuelve (CE + coef·L_aux, CE, L_aux); L_aux = 0 en el modelo base.
- `scripts/train_shakespeare.py --config configs/X.json --seed s [--steps n]`: descarga Tiny Shakespeare a `data/` (ignorado en git), tokenizador por carácter (vocab 65; con BPE de 50257 el embedding dominaría un modelo tan pequeño), 90/10 train/val, lotes de trozos contiguos aleatorios con un `torch.Generator` sembrado, AdamW. Evalúa la val CE sobre 20 lotes fijos (semilla 1234, iguales para todos los modelos). Guarda en `results/<nombre>_seed<s>.json`: configuración, parámetros totales/activos, tokens/s (solo pasos de entrenamiento), historial (train/val CE, L_aux, fracción de uso por capa y experto) y una muestra generada con `generate` del libro.
- Configs: `configs/dense.json`, `moe_e4k1.json`, `moe_e4k2.json` (emb 128, 4 capas, 4 cabezas, ctx 128, batch 32, lr 1e-3, wd 0.1, 2000 pasos, coef aux 0.01).
- Tests (`tests/test_train.py`): tokenizador ida y vuelta; entrenamiento corto con texto sintético para base y MoE; misma semilla ⇒ mismas pérdidas y misma muestra (reproducibilidad); fracciones de uso suman 1.
- Prueba de humo (100 pasos, seed 1, NO es resultado de la memoria): denso 824,832 parámetros, 18.0k tok/s, val CE 2.506; MoE E4k2 2,407,424 totales / 1,353,728 activos, 11.7k tok/s, val CE 2.490, L_aux 1.03.

**Bloque 7 — Batería de experimentos (ejecutada el 13-oct, ~3 h 20 min en CPU, 8 hilos).** Comando: `python scripts/run_all.py` (salta lo ya hecho; log en `results/run_all.log`). 6 configuraciones × semillas {1, 2, 3} = 18 entrenamientos de 2000 pasos (≈8.2M tokens cada uno), en secuencia para no falsear los tokens/s, y al final `scripts/bench_dispatch.py`.
| Config | Qué es | Pregunta que responde | Parámetros totales / activos |
|---|---|---|---|
| `dense` | GPT del libro (FFN 128→512→128) | referencia | 824,832 / 824,832 |
| `moe_e4k1` | MoE 4 expertos, k=1 (Switch), coef aux 0.01 | ¿mejora el MoE a igual cómputo por token? | 2,407,424 / 826,880 |
| `moe_e4k2` | MoE 4 expertos, k=2 (Mixtral), coef 0.01 | ¿cuánto aporta un segundo experto activo? | 2,407,424 / 1,353,728 |
| `dense_wide` | denso con FFN 4× más ancho (hidden 2048) | ¿el MoE se acerca a un denso con los mismos parámetros totales pero 4× más cómputo en el FFN? | 2,404,352 / 2,404,352 |
| `moe_e4k1_aux0` | como e4k1 sin pérdida de balanceo | ¿colapsa el router sin L_aux? | igual que e4k1 |
| `moe_e4k1_aux0.1` | como e4k1 con coef 0.1 | ¿un balanceo fuerte perjudica la CE? | igual que e4k1 |
- `dense_wide` se implementa como MoE de 1 experto con hidden 2048 (probado equivalente a un FFN denso en el bloque 5; el router no recibe gradiente porque el peso es siempre 1). Su tokens/s incluye el pequeño sobrecoste del despacho.
- Velocidades medidas en pruebas cortas: dense 18.0k, e4k1 17.2k, e4k2 11.7k, dense_wide 6.3k tok/s.
- `bench_dispatch.py`: forward+backward de FFN denso vs MoE {E4k1, E4k2, E8k1, E8k2} × despachos {dense, loop, sorted}, 4096 tokens, D=128, mediana de 30 repeticiones → `results/dispatch_bench.json`.
- Los pesos se guardan en `checkpoints/` (no versionado) para el análisis de especialización (qué caracteres van a cada experto), que se hará después.
- Mientras corre, no usar el PC para cosas pesadas (falsea los tokens/s).
- Análisis: `scripts/analyze.py` → `results/summary.{json,md}`, `results/fig_val_ce.png`, `results/fig_expert_load.png`; `scripts/specialization.py --name X` → `results/specialization_X.{json,md}`. `run_all.py` los lanza al final. Regenerarlos da ficheros idénticos byte a byte (comprobado con `git status`).

**Resultados del bloque 7 (cifras reales, `results/summary.md`; media ± desviación, 3 semillas, 2000 pasos)**
| Modelo | Parám. totales / activos | Val CE | tok/s entrenando | Tiempo | Carga máx. experto (peor capa) |
|---|---|---|---|---|---|
| Denso (libro) | 824,832 / 824,832 | 1.6449 ± 0.0077 | 18,636 | 7.3 min | — |
| MoE E=4 k=1 (coef 0.01) | 2,407,424 / 826,880 | 1.5953 ± 0.0119 | 17,585 | 7.8 min | 0.341 ± 0.053 |
| MoE E=4 k=2 (coef 0.01) | 2,407,424 / 1,353,728 | 1.5807 ± 0.0064 | 11,920 | 11.5 min | 0.307 ± 0.004 |
| Denso ancho (FFN 4×) | 2,404,352 / 2,404,352 | 1.5760 ± 0.0039 | 6,390 | 21.4 min | — |
| MoE E=4 k=1, coef 0 | 2,407,424 / 826,880 | 1.6051 ± 0.0064 | 17,492 | 7.8 min | 0.654 ± 0.080 |
| MoE E=4 k=1, coef 0.1 | 2,407,424 / 826,880 | 1.5935 ± 0.0044 | 17,865 | 7.6 min | 0.263 ± 0.004 |

Conclusiones (y cuánto se pueden afirmar):
1. **MoE k=1 vs denso con el mismo cómputo por token:** −0.050 de val CE con solo un 5.6 % menos de tok/s. Es robusto: las 3 semillas del MoE (1.582, 1.601, 1.604) son mejores que las 3 del denso (1.641, 1.654, 1.640).
2. **Frente al denso ancho (mismos parámetros totales):** el MoE k=1 recupera el 72 % de la mejora del denso ancho ((1.6449−1.5953)/(1.6449−1.5760)) y el k=2 el 93 %, entrenando 2.75× y 1.86× más rápido respectivamente. La diferencia entre k=2 y el denso ancho (0.005) es del orden de la desviación: no se puede afirmar que sean distintos.
3. **Ablación de L_aux:** sin pérdida de balanceo el router se desequilibra (el experto más usado llega al 65 % ± 8 % de los tokens en la peor capa, y hasta un 74 % en una capa; L_aux 1.40) pero NO colapsa del todo. Con 0.01 queda en 34 % y con 0.1 en 26 % (casi el 25 % ideal). La val CE sin L_aux es algo peor (1.605 vs 1.595), pero la diferencia es comparable a la desviación: es una tendencia, no un resultado concluyente. Un coef. de 0.1 no perjudica la CE (1.5935).
4. **Despacho** (`results/dispatch_bench.json`, una capa, 4096 tokens, fwd+bwd): `sorted` cuesta lo mismo que el FFN denso con k=1 (24.07 vs 23.56 ms, +2 %) y escala con k, no con E (E=8 k=1: 25.90 ms). `sorted` gana a `loop` en todas las configuraciones, y más con más expertos (E=8 k=1: 25.90 vs 28.08 ms, −8 %). La referencia densa cuesta E veces más (99 ms con E=4, 203 ms con E=8).
5. **Especialización** (`results/specialization_moe_e4k1.md`): la información mutua normalizada entre la clase del carácter de entrada y el experto crece con la profundidad (capa 1: 0.08–0.22; capa 4: 0.37–0.41, en las 3 semillas). En la capa 4 (semilla 1), un experto recibe el 100 % de los saltos de línea, el 82 % de las mayúsculas y el 86 % de la puntuación ("fronteras": fin de verso, nombres de personaje); vocales y consonantes van a otros expertos. Matiz: el router ve el estado contextual, no solo el carácter, así que es una correlación, no una regla.
- **Limitaciones:** los modelos siguen mejorando a los 2000 pasos (la mejor val CE es la del último paso), así que son comparaciones a presupuesto fijo, no a convergencia; solo 3 semillas; un único corpus pequeño y tokens por carácter; tiempos en CPU (en GPU el despacho se comporta distinto).

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
- ¿Qué compara exactamente MoE k=1 vs denso, y qué compara MoE vs denso ancho? ¿Qué se mantiene igual en cada comparación?
- Sin L_aux, ¿qué le pasa a la carga de los expertos? ¿Por qué un desequilibrio así es peor aunque la CE apenas cambie? (pista: capacidad desperdiciada, y en GPU, cuellos de botella)
- ¿Por qué el despacho `sorted` cuesta casi lo mismo con E=4 y E=8 si k=1?
- ¿Qué mide la información mutua normalizada del análisis de especialización y por qué crece con la profundidad?
- ¿Por qué con 3 semillas no podemos afirmar que MoE k=2 y el denso ancho sean distintos?
- ¿Por qué se evalúa a todos los modelos sobre los mismos lotes de validación? ¿Por qué tokenizar por carácter y no con el BPE de GPT-2 aquí?
- En el entrenamiento MoE, ¿qué se suma a la CE y con qué peso? ¿Qué valor de L_aux indicaría que el router está equilibrado?

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
