# Evidencia de rendimiento y portabilidad

Generado el 2026-11-20 14:06 sobre el commit `0af944c` (con cambios locales sin commit).

Para regenerarlo: `python -m scripts.measure_requirements`

> Medido en una sola máquina, con la GPU compartida con el escritorio: otras aplicaciones usan parte de su memoria y de su capacidad. Los tiempos son indicativos de este equipo.

## Requisitos

| Requisito | Resultado | Dato que lo sustenta |
|---|---|---|
| RR-001 | **CUMPLE** | Máximo en frío 9.50 s (incluye la carga de modelos) y en caliente 0.41 s, frente al límite de 15 s (3 escenas sintéticas 640x480). |
| RR-002 | **CUMPLE** | 12 mapas de profundidad de entradas 640x480: forma 480x640, valores finitos y positivos, entre 1.23 m y 5.47 m. |
| RF-009 | **CUMPLE** | nvidia-smi: memoria de la GPU de 1357 MiB en reposo a 2952 MiB durante el análisis (+1595 MiB; umbral 100 MiB); utilización máxima 53 % (reposo 22 %); PyTorch reservó como máximo 871 MiB. |
| RF-010 | **CUMPLE** | GPU oculta al proceso: sí; dispositivo resuelto: CPU; escenas sintéticas completadas: sí; aviso en las advertencias del reporte: sí («No se detectó una tarjeta gráfica compatible; el análisis se ejecutará en el procesador y podría tardar más de lo habitual.»). |
| RA-002 | **CUMPLE** | Las 3 escenas sintéticas se completaron en GPU y en el entorno sin GPU, en procesos separados. |

## Entorno

| Elemento | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 |
| Controlador NVIDIA | 610.88 |
| PyTorch | 2.14.1+cu130 |
| CUDA (PyTorch) | 13.0 |
| Python | 3.13.7 |
| Sistema operativo | Windows-11-10.0.26200-SP0 |
| Procesador | Intel64 Family 6 Model 151 Stepping 5, GenuineIntel |

## GPU en reposo y durante la medición (nvidia-smi)

| Medida | Reposo (mediana de 5 lecturas) | Máximo durante la medición |
|---|---|---|
| Utilización | 22 % | 53 % |
| Memoria usada | 1357 MiB | 2952 MiB |

Aumento de memoria sobre el reposo: 1595 MiB. Memoria máxima reservada por PyTorch: 871 MiB. Lecturas cada 0.2 s (108 en total): pueden perder picos de utilización de decenas de milisegundos.

## Tiempos en GPU

Dispositivos resueltos: depth NVIDIA GeForce RTX 3070, comparison NVIDIA GeForce RTX 3070, segmentation NVIDIA GeForce RTX 3070.

### Escenas sintéticas (640x480)

| Par | Resolución | Análisis | Total (s) | Adquisición | Carga de modelos | Profundidad | Alineación | Segmentación | Comparación | Dibujo |
|---|---|---|---|---|---|---|---|---|---|---|
| sintético: desplazamiento | 640x480 | frío (con carga) | 9.50 | 0.00 | 7.60 | 0.73 | 0.07 | 0.87 | 0.22 | 0.01 |
| sintético: desplazamiento | 640x480 | caliente (media de 10; mín–máx) | 0.31 (0.27–0.40) | 0.00 | — | 0.05 | 0.07 | 0.16 | 0.03 | 0.01 |
| sintético: retiro | 640x480 | frío | 0.28 | 0.00 | — | 0.05 | 0.06 | 0.15 | 0.02 | 0.00 |
| sintético: retiro | 640x480 | caliente (media de 10; mín–máx) | 0.27 (0.25–0.30) | 0.00 | — | 0.05 | 0.06 | 0.15 | 0.01 | 0.00 |
| sintético: poca luz | 640x480 | frío | 0.28 | 0.00 | — | 0.05 | 0.06 | 0.16 | 0.01 | 0.00 |
| sintético: poca luz | 640x480 | caliente (media de 10; mín–máx) | 0.28 (0.24–0.41) | 0.00 | — | 0.05 | 0.06 | 0.16 | 0.01 | 0.00 |

### Pares de fotos reales (1920x1080, informativo)

| Par | Resolución | Análisis | Total (s) | Adquisición | Carga de modelos | Profundidad | Alineación | Segmentación | Comparación | Dibujo |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 1920x1080 | frío | 1.00 | 0.00 | — | 0.15 | 0.38 | 0.26 | 0.20 | 0.01 |
| real | 1920x1080 | caliente (media de 3; mín–máx) | 0.81 (0.79–0.82) | 0.00 | — | 0.09 | 0.32 | 0.21 | 0.17 | 0.01 |
| real_2 | 1920x1080 | frío | 0.93 | 0.00 | — | 0.10 | 0.32 | 0.23 | 0.27 | 0.01 |
| real_2 | 1920x1080 | caliente (media de 3; mín–máx) | 0.87 (0.87–0.88) | 0.00 | — | 0.08 | 0.30 | 0.21 | 0.27 | 0.01 |
| real_3 | 1920x1080 | frío | 0.94 | 0.00 | — | 0.09 | 0.29 | 0.24 | 0.32 | 0.01 |
| real_3 | 1920x1080 | caliente (media de 3; mín–máx) | 0.94 (0.90–1.03) | 0.00 | — | 0.08 | 0.31 | 0.22 | 0.32 | 0.01 |

## Tiempos en CPU (sin GPU)

Dispositivos resueltos: depth CPU, comparison CPU, segmentation CPU.
Aviso mostrado a la persona usuaria: «No se detectó una tarjeta gráfica compatible; el análisis se ejecutará en el procesador y podría tardar más de lo habitual.»

### Escenas sintéticas (640x480)

| Par | Resolución | Análisis | Total (s) | Adquisición | Carga de modelos | Profundidad | Alineación | Segmentación | Comparación | Dibujo |
|---|---|---|---|---|---|---|---|---|---|---|
| sintético: desplazamiento | 640x480 | frío (con carga) | 12.55 | 0.00 | 5.98 | 2.24 | 0.09 | 4.12 | 0.11 | 0.01 |
| sintético: desplazamiento | 640x480 | caliente (media de 3; mín–máx) | 6.32 (5.97–6.85) | 0.00 | — | 1.76 | 0.06 | 4.41 | 0.09 | 0.01 |
| sintético: retiro | 640x480 | frío | 5.91 | 0.00 | — | 1.76 | 0.06 | 4.00 | 0.08 | 0.00 |
| sintético: retiro | 640x480 | caliente (media de 3; mín–máx) | 6.18 (5.97–6.32) | 0.00 | — | 1.81 | 0.06 | 4.18 | 0.12 | 0.00 |
| sintético: poca luz | 640x480 | frío | 7.35 | 0.00 | — | 2.22 | 0.07 | 4.98 | 0.08 | 0.00 |
| sintético: poca luz | 640x480 | caliente (media de 3; mín–máx) | 7.21 (6.05–8.25) | 0.00 | — | 2.20 | 0.06 | 4.85 | 0.09 | 0.00 |

### Pares de fotos reales (1920x1080, informativo)

| Par | Resolución | Análisis | Total (s) | Adquisición | Carga de modelos | Profundidad | Alineación | Segmentación | Comparación | Dibujo |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 1920x1080 | frío | 9.24 | 0.00 | — | 2.87 | 0.34 | 5.40 | 0.62 | 0.01 |
| real_2 | 1920x1080 | frío | 8.89 | 0.00 | — | 2.88 | 0.30 | 4.96 | 0.73 | 0.01 |
| real_3 | 1920x1080 | frío | 10.24 | 0.00 | — | 3.34 | 0.36 | 5.78 | 0.74 | 0.01 |

## Observaciones

- GPU, sintético: desplazamiento: se esperaba un desplazamiento de unos 51 cm; se obtuvo: cuadro: DESPLAZAMIENTO (51.6 cm).
- GPU, sintético: retiro: se esperaba un objeto retirado; se obtuvo: cuadro: RETIRADO / DESAPARICIÓN.
- GPU, sintético: poca luz: se esperaba sin cambios, con aviso de poca luz; se obtuvo: sin cambios.
- GPU: solo el primer análisis del proceso incluye la carga de los modelos; los «frío» de los pares siguientes son el primer análisis de cada par con los modelos ya en memoria.
- CPU, sintético: desplazamiento: se esperaba un desplazamiento de unos 51 cm; se obtuvo: cuadro: DESPLAZAMIENTO (51.5 cm).
- CPU, sintético: retiro: se esperaba un objeto retirado; se obtuvo: cuadro: RETIRADO / DESAPARICIÓN.
- CPU, sintético: poca luz: se esperaba sin cambios, con aviso de poca luz; se obtuvo: sin cambios.
- CPU: solo el primer análisis del proceso incluye la carga de los modelos; los «frío» de los pares siguientes son el primer análisis de cada par con los modelos ya en memoria.
