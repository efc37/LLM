## ¿Responde el router a la posición? (MoE E=4 k=1, ventanas de 1024, 20 ventanas de validación)

Capa con mayor cambio de reparto; experto dominante en 512-1023. Media ± desv. típica entre semillas.

| Codificación | capa (por semilla) | corr(posición, logit) | Δ logit (896-1023 − 0-127) | prob. 0-127 → 896-1023 | top-1 0-127 → 896-1023 |
|---|---|---|---|---|---|
| learned | [4, 2, 4] | 0.22 ± 0.10 | +0.71 ± 0.38 | 0.25 → 0.33 | 0.26 → 0.35 |
| nope | [2, 2, 3] | 0.52 ± 0.27 | +0.55 ± 0.32 | 0.24 → 0.40 | 0.26 → 0.86 |
| alibi | [2, 4, 1] | 0.02 ± 0.03 | +0.06 ± 0.05 | 0.28 → 0.29 | 0.28 → 0.29 |
| t5 | [1, 2, 1] | 0.16 ± 0.15 | +0.10 ± 0.14 | 0.27 → 0.31 | 0.29 → 0.44 |
