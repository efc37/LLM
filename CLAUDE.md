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
- Ampliaciones elegidas por el equipo: _(pendiente: leer el repo, README, ramas y commits para averiguarlo; si no está claro, PREGUNTAR)_
- Reparto de tareas entre miembros: _(pendiente)_
- Mi parte como colaborador: _(pendiente de definir con el equipo)_

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
