## Tiny Shakespeare (char-level), 2000 pasos, 3 semillas (media ± desviación)

| Modelo | Parám. totales | Parám. activos | Val CE | Train CE | tok/s | Tiempo (min) | Carga máx. experto | L_aux |
|---|---|---|---|---|---|---|---|---|
| Denso (libro) | 824,832 | 824,832 | 1.6449 ± 0.0077 | 1.4619 ± 0.0049 | 18,636 ± 415 | 7.3 | - | - |
| MoE E=4 k=1 | 2,407,424 | 826,880 | 1.5953 ± 0.0119 | 1.3950 ± 0.0122 | 17,585 ± 189 | 7.8 | 0.341 ± 0.053 | 1.019 ± 0.009 |
| MoE E=4 k=2 | 2,407,424 | 1,353,728 | 1.5807 ± 0.0064 | 1.3735 ± 0.0030 | 11,920 ± 150 | 11.5 | 0.307 ± 0.004 | 1.020 ± 0.007 |
| Denso ancho (FFN 4x) | 2,404,352 | 2,404,352 | 1.5760 ± 0.0039 | 1.3710 ± 0.0025 | 6,390 ± 16 | 21.4 | - | - |
| MoE E=4 k=1, coef. aux 0 | 2,407,424 | 826,880 | 1.6051 ± 0.0064 | 1.4021 ± 0.0149 | 17,492 ± 140 | 7.8 | 0.654 ± 0.080 | 1.401 ± 0.178 |
| MoE E=4 k=1, coef. aux 0.1 | 2,407,424 | 826,880 | 1.5935 ± 0.0044 | 1.3924 ± 0.0040 | 17,865 ± 19 | 7.6 | 0.263 ± 0.004 | 1.001 ± 0.000 |
| MoE E=4 k=1, z-loss 0.001 | 2,407,424 | 826,880 | 1.5964 ± 0.0104 | 1.3964 ± 0.0136 | 15,958 ± 331 | 8.6 | 0.345 ± 0.065 | 1.018 ± 0.013 |

Carga máx. experto: fracción de asignaciones del experto más usado en la peor capa (últimos 100 pasos); 0.25 = equilibrio perfecto con E=4.

## Despacho: forward+backward de una capa (B=32, T=128, D=128), mediana de 30 repeticiones

| Capa | Despacho | ms | tok/s |
|---|---|---|---|
| FFN denso | - | 23.56 ± 0.38 | 173,840 |
| MoE E=4 k=1 | dense | 99.44 ± 3.32 | 41,191 |
| MoE E=4 k=1 | loop | 25.00 ± 0.66 | 163,834 |
| MoE E=4 k=1 | sorted | 24.07 ± 2.68 | 170,175 |
| MoE E=4 k=2 | dense | 99.34 ± 1.55 | 41,232 |
| MoE E=4 k=2 | loop | 49.82 ± 1.24 | 82,213 |
| MoE E=4 k=2 | sorted | 48.55 ± 0.84 | 84,373 |
| MoE E=8 k=1 | dense | 202.81 ± 2.62 | 20,196 |
| MoE E=8 k=1 | loop | 28.08 ± 1.02 | 145,851 |
| MoE E=8 k=1 | sorted | 25.90 ± 0.95 | 158,121 |
| MoE E=8 k=2 | dense | 202.32 ± 2.31 | 20,245 |
| MoE E=8 k=2 | loop | 50.08 ± 1.61 | 81,792 |
| MoE E=8 k=2 | sorted | 47.90 ± 1.63 | 85,503 |

torch 2.6.0+cpu, 8 hilos, CPU.
