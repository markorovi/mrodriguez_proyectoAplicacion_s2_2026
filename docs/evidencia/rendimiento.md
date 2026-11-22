# Evidencia de rendimiento y portabilidad

Generado el 2026-11-21 22:28 sobre el commit `305f284`.

Para regenerarlo: `python -m scripts.measure_requirements`

> Medido en una sola máquina, con la GPU compartida con el escritorio: otras aplicaciones usan parte de su memoria y de su capacidad. Los tiempos son indicativos de este equipo.

## Requisitos

| Requisito | Resultado | Dato que lo sustenta |
|---|---|---|
| RR-001 | **CUMPLE** | Máximo en frío 12.94 s (incluye la carga de modelos) y en caliente 0.44 s, frente al límite de 15 s (3 escenas sintéticas 640x480). |
| RR-002 | **CUMPLE** | 12 mapas de profundidad de entradas 640x480: forma 480x640, valores finitos y positivos, entre 1.23 m y 5.47 m. |
| RF-009 | **CUMPLE** | nvidia-smi: memoria de la GPU de 1285 MiB en reposo a 2836 MiB durante el análisis (+1551 MiB; umbral 100 MiB); utilización máxima 54 % (reposo 12 %); PyTorch reservó como máximo 871 MiB. |
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
| Utilización | 12 % | 54 % |
| Memoria usada | 1285 MiB | 2836 MiB |

Aumento de memoria sobre el reposo: 1551 MiB. Memoria máxima reservada por PyTorch: 871 MiB. Lecturas cada 0.2 s (132 en total): pueden perder picos de utilización de decenas de milisegundos.

## Tiempos en GPU

Dispositivos resueltos: depth NVIDIA GeForce RTX 3070, comparison NVIDIA GeForce RTX 3070, segmentation NVIDIA GeForce RTX 3070.

### Escenas sintéticas (640x480)

| Par | Resolución | Análisis | Total (s) | Adquisición | Carga de modelos | Profundidad | Alineación | Segmentación | Comparación | Dibujo |
|---|---|---|---|---|---|---|---|---|---|---|
| sintético: desplazamiento | 640x480 | frío (con carga) | 12.94 | 0.00 | 10.56 | 0.88 | 0.14 | 1.05 | 0.28 | 0.01 |
| sintético: desplazamiento | 640x480 | caliente (media de 10; mín–máx) | 0.38 (0.35–0.44) | 0.00 | — | 0.05 | 0.06 | 0.23 | 0.03 | 0.01 |
| sintético: retiro | 640x480 | frío | 0.36 | 0.00 | — | 0.05 | 0.06 | 0.23 | 0.01 | 0.00 |
| sintético: retiro | 640x480 | caliente (media de 10; mín–máx) | 0.34 (0.33–0.35) | 0.00 | — | 0.05 | 0.06 | 0.22 | 0.01 | 0.00 |
| sintético: poca luz | 640x480 | frío | 0.33 | 0.00 | — | 0.05 | 0.06 | 0.21 | 0.01 | 0.00 |
| sintético: poca luz | 640x480 | caliente (media de 10; mín–máx) | 0.34 (0.33–0.36) | 0.00 | — | 0.05 | 0.06 | 0.22 | 0.01 | 0.00 |

### Pares de fotos reales (1920x1080, informativo)

| Par | Resolución | Análisis | Total (s) | Adquisición | Carga de modelos | Profundidad | Alineación | Segmentación | Comparación | Dibujo |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 1920x1080 | frío | 0.89 | 0.00 | — | 0.14 | 0.29 | 0.27 | 0.17 | 0.02 |
| real | 1920x1080 | caliente (media de 3; mín–máx) | 0.84 (0.79–0.90) | 0.00 | — | 0.08 | 0.29 | 0.30 | 0.16 | 0.01 |
| real_2 | 1920x1080 | frío | 0.95 | 0.00 | — | 0.10 | 0.28 | 0.31 | 0.25 | 0.01 |
| real_2 | 1920x1080 | caliente (media de 3; mín–máx) | 0.86 (0.85–0.88) | 0.00 | — | 0.08 | 0.26 | 0.26 | 0.25 | 0.01 |
| real_3 | 1920x1080 | frío | 0.88 | 0.00 | — | 0.08 | 0.25 | 0.25 | 0.29 | 0.01 |
| real_3 | 1920x1080 | caliente (media de 3; mín–máx) | 0.90 (0.89–0.91) | 0.00 | — | 0.08 | 0.26 | 0.26 | 0.29 | 0.01 |

## Tiempos en CPU (sin GPU)

Dispositivos resueltos: depth CPU, comparison CPU, segmentation CPU.
Aviso mostrado a la persona usuaria: «No se detectó una tarjeta gráfica compatible; el análisis se ejecutará en el procesador y podría tardar más de lo habitual.»

### Escenas sintéticas (640x480)

| Par | Resolución | Análisis | Total (s) | Adquisición | Carga de modelos | Profundidad | Alineación | Segmentación | Comparación | Dibujo |
|---|---|---|---|---|---|---|---|---|---|---|
| sintético: desplazamiento | 640x480 | frío (con carga) | 16.59 | 0.00 | 6.84 | 2.76 | 0.10 | 6.77 | 0.11 | 0.01 |
| sintético: desplazamiento | 640x480 | caliente (media de 3; mín–máx) | 8.02 (7.73–8.53) | 0.00 | — | 1.92 | 0.06 | 5.94 | 0.09 | 0.01 |
| sintético: retiro | 640x480 | frío | 7.72 | 0.00 | — | 1.81 | 0.06 | 5.77 | 0.08 | 0.00 |
| sintético: retiro | 640x480 | caliente (media de 3; mín–máx) | 8.16 (7.79–8.43) | 0.00 | — | 1.82 | 0.06 | 6.20 | 0.08 | 0.00 |
| sintético: poca luz | 640x480 | frío | 9.15 | 0.00 | — | 2.18 | 0.07 | 6.82 | 0.09 | 0.00 |
| sintético: poca luz | 640x480 | caliente (media de 3; mín–máx) | 8.87 (8.84–8.91) | 0.00 | — | 1.99 | 0.07 | 6.72 | 0.09 | 0.00 |

### Pares de fotos reales (1920x1080, informativo)

| Par | Resolución | Análisis | Total (s) | Adquisición | Carga de modelos | Profundidad | Alineación | Segmentación | Comparación | Dibujo |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 1920x1080 | frío | 11.72 | 0.00 | — | 3.06 | 0.30 | 7.62 | 0.73 | 0.01 |
| real_2 | 1920x1080 | frío | 11.95 | 0.00 | — | 2.96 | 0.34 | 7.85 | 0.78 | 0.01 |
| real_3 | 1920x1080 | frío | 11.68 | 0.00 | — | 2.95 | 0.30 | 7.60 | 0.82 | 0.01 |

## Observaciones

- GPU, sintético: desplazamiento: se esperaba un desplazamiento de unos 51 cm; se obtuvo: cuadro: DESPLAZAMIENTO (51.6 cm).
- GPU, sintético: retiro: se esperaba un objeto retirado; se obtuvo: cuadro: RETIRADO / DESAPARICIÓN.
- GPU, sintético: poca luz: se esperaba sin cambios, con aviso de poca luz; se obtuvo: sin cambios.
- GPU: solo el primer análisis del proceso incluye la carga de los modelos; los «frío» de los pares siguientes son el primer análisis de cada par con los modelos ya en memoria.
- CPU, sintético: desplazamiento: se esperaba un desplazamiento de unos 51 cm; se obtuvo: cuadro: DESPLAZAMIENTO (51.5 cm).
- CPU, sintético: retiro: se esperaba un objeto retirado; se obtuvo: cuadro: RETIRADO / DESAPARICIÓN.
- CPU, sintético: poca luz: se esperaba sin cambios, con aviso de poca luz; se obtuvo: sin cambios.
- CPU: solo el primer análisis del proceso incluye la carga de los modelos; los «frío» de los pares siguientes son el primer análisis de cada par con los modelos ya en memoria.
