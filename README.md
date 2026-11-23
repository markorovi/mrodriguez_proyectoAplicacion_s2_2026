# Detección de cambios en escenas mediante profundidad monocular

Aplicación de escritorio en Python que detecta cambios entre dos fotografías RGB de un
mismo espacio interior, tomadas en momentos distintos ("antes" y "después") con una cámara
convencional (webcam o teléfono). No requiere sensores de profundidad dedicados (LiDAR,
RGB-D): la geometría de la escena se infiere con un modelo de visión basado en transformers.

## Descripción técnica

El sistema ejecuta un pipeline de seis etapas:

1. **Captura**: carga de imágenes desde disco o captura desde una cámara RGB conectada, con
   validación de formato, resolución, canales e iluminación.
2. **Estimación de profundidad**: un mapa de profundidad **métrica** (en metros) por imagen
   con [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), variante
   *Metric Indoor*, ejecutado con PyTorch en GPU (CUDA) o, si no hay GPU, en CPU.
3. **Alineación espacial**: homografía entre ambas vistas (SIFT + razón de Lowe + RANSAC) que
   lleva la imagen y el mapa de profundidad "después" al plano de la imagen "antes", con una
   máscara de la zona que ambas comparten.
4. **Segmentación semántica**: clase de cada píxel con SegFormer (ADE20K), usada para nombrar
   en español los objetos que cambiaron.
5. **Comparación y metrología 3D**: diferencia de profundidad con normalización de escala,
   umbral de 5 cm y verificación por apariencia; retroproyección pinhole con los intrínsecos
   `fx, fy, cx, cy` de la cámara que tomó la foto (calibración, EXIF o estimación; ver
   [Parámetros de la cámara](#parámetros-de-la-cámara)); clasificación en *retirado*,
   *desplazamiento* (con la distancia 3D en centímetros) o *aparición*.
6. **Reporte**: lista estructurada de objetos y la imagen con los cambios resaltados.

Todo el procesamiento es local; ninguna imagen se envía a servicios externos.

### Arquitectura

Arquitectura en capas con paquetes independientes. Solo la capa de orquestación invoca a los
servicios, de modo que cada modelo puede sustituirse sin modificar los demás módulos.

| Paquete | Responsabilidad |
|---|---|
| `config/` | Configuración tipada e inmutable (Pydantic): criterios de imagen, cámara, modelo, perfil calibrado opcional `fx, fy, cx, cy` y estimación de la cámara. |
| `models/` | Estructuras de datos del dominio (`ImageData`, `DepthMap`, `ValidationReport`, `AcquiredImage`, `AlignmentResult`, `SegmentationMask`, `ChangedObject`, `ComparisonReport`), historial de la sesión (`SessionHistory`) y jerarquía de excepciones. Caché local de pesos en `models/weights/`. |
| `services/capture/` | Fuentes de imagen (`ImageSource`, implementada por `FileImageLoader` y `CameraCapture`) y `ImageAcquisitionService`, que entrega imágenes ya validadas; `ExampleCatalog` lee los ejemplos precargados de `data/samples/examples.json`. |
| `services/inference/` | `BaseDepthEstimator` (contrato común), `DepthAnythingV2Estimator`, `DeviceManager` (GPU/CPU), cargador de modelos de Hugging Face y `DepthLatencyBenchmark`. |
| `services/alignment/` | `SpatialAlignmentService`: puntos de interés, emparejamiento, homografía RANSAC y deformación de imagen, profundidad y máscara de solapamiento. |
| `services/segmentation/` | `BaseSegmenter` (contrato común), `SegFormerSegmenter` (GPU/CPU), la fábrica `create_segmenter`, cargador de modelos de Hugging Face y `RegionLabeler`, que nombra en español la región de un cambio. |
| `services/comparison/` | `SpatialComparisonService`: diferencia de profundidad, regiones de cambio, retroproyección 3D y clasificación. |
| `controllers/` | `PipelineController`: orquestación de extremo a extremo (fachada única para la interfaz). |
| `utils/` | `ImageValidator`, estadísticas fotométricas, geometría pinhole, parámetros de cámara por foto (`camera_model`), calibración con tablero (`camera_calibration`), coloreado de mapas de profundidad, dibujo de los cambios y generador de escenas sintéticas. |
| `ui/` | Interfaz gráfica de escritorio (PySide6): ventana principal (`MainWindow`), hilo de análisis (`PipelineWorker`), pantalla de resultado (`ReportWidget`) con exportación, historial de la sesión (`HistoryDialog`), manual de usuario (`ManualDialog`), y paneles y diálogo de selección de imágenes. Solo usa el `PipelineController` y los modelos de dominio. |
| `scripts/` | Utilidades de línea de comandos: medición de latencia de la profundidad y del análisis completo (CPU y GPU), evidencia de los requisitos de rendimiento y portabilidad (`measure_requirements`) y calibración de la cámara. |
| `tests/` | Pruebas unitarias y de integración con `pytest`. |
| `data/samples/` | Escenas interiores sintéticas para demostración y pruebas sin conexión, tres pares de fotos reales sin metadatos (`ejemplo_01_a.jpg` a `ejemplo_03_b.jpg`) para los ejemplos precargados y una foto de exterior (`no_interior_01.png`) para probar el rechazo de exteriores. Las fotos personales originales no se versionan (ver [data/samples/README.md](data/samples/README.md)). |

Patrones aplicados:

- **Estrategia**: `BaseDepthEstimator` define `estimate(image) -> DepthMap`; cualquier modelo
  de profundidad se integra implementando `_predict`, `load` y `unload`. Del mismo modo,
  `BaseSegmenter` define `segment(image) -> SegmentationMask` y un modelo de segmentación se
  integra implementando `_predict`, `_class_names`, `load` y `unload`.
- **Método plantilla**: `estimate` y `segment` validan la entrada, cargan el modelo de forma
  perezosa, miden el tiempo, verifican la resolución de salida y traducen fallos de ejecución
  (p. ej. memoria de GPU insuficiente) a `DepthEstimationError` o `SegmentationError`.
- **Fábrica**: `create_depth_estimator` y `create_segmenter` son el único lugar que nombra la
  clase concreta de cada modelo. `PipelineController` depende solo de `BaseDepthEstimator`,
  `BaseSegmenter` y esas fábricas, así que cambiar un modelo por una versión más reciente no
  obliga a modificar el orquestador.
- **Adaptador**: `DepthAnythingV2Estimator` y `SegFormerSegmenter` adaptan el procesador y el
  modelo de Hugging Face a la interfaz uniforme del estimador y del segmentador.
- **Inyección de dependencias**: el dispositivo de video, el selector de hardware, los
  cargadores de modelos y cada servicio del pipeline son reemplazables, lo que permite probar
  cada módulo sin hardware ni descargas.
- **Fachada**: `PipelineController` expone `analyze_files(antes, después)` y
  `analyze(antes, después)`, y oculta el orden de las etapas y la vida de los modelos.
- **Observador**: ambos métodos aceptan un `progress` opcional que recibe la clave de cada etapa
  de `STAGES` al comenzar; `PipelineWorker` lo reemite como señal de Qt para la barra de progreso.

## Estado actual

**Avance 2 completado: alineación espacial, metrología 3D, segmentación y orquestación.
Interfaz de usuario final (ACT-09) y verificación de escena interior (RF-012) completadas.**

| Componente | Estado |
|---|---|
| Estructura modular, configuración, modelos de dominio | Completado |
| Carga desde disco (rutas con caracteres no ASCII, orientación EXIF, archivos dañados o truncados) | Completado |
| Captura desde cámara RGB (verificada con webcam física) | Completado |
| Validación: formato, resolución mínima 640x480, RGB de 8 bits, advertencia de baja iluminación | Completado |
| Diálogo gráfico de selección de imágenes "Antes" / "Después" con vista previa y estado | Completado |
| Estimación de profundidad métrica con Depth Anything V2 (GPU con precisión mixta / CPU) | Completado, verificada en GPU NVIDIA |
| Alineación espacial por homografía (SIFT u ORB, FLANN o fuerza bruta, RANSAC) | Completado |
| Comparación de profundidad, clasificación y metrología 3D (distancia en cm) | Completado |
| Segmentación semántica con SegFormer-B4 (ADE20K) y etiquetas en español | Completado, verificada en GPU NVIDIA |
| Fotos reales, par 1 (1920x1080, caja desplazada sobre una cama) | Verificada: 1 desplazamiento, "caja", 62.6 cm |
| Fotos reales, par 2 (taza desplazada, caja movida 5.6 cm, cubo retirado) | Verificada: los 3 cambios, sin falsos positivos; caja medida en 6.6 cm (error de 1.0 cm) |
| `PipelineController` y `ComparisonReport` con imagen anotada | Completado |
| Fotos reales, par 3 (caja girada y movida, taza colocada encima de la caja) | Verificada: 2 desplazamientos separados (caja 47.3 cm, taza 86.8 cm) |
| Ventana principal: análisis en segundo plano con progreso, reporte por objeto, zoom, fotos originales junto al resultado (RI-003), exportación (PNG + TXT/JSON) y guardado de la imagen | Completado |
| Verificación de escena interior (RF-012) con la segmentación | Completado: juzga las dos fotos originales completas, no la vista alineada. Con el modelo real (`tests/test_indoor_real_integration.py`), la foto de exterior versionada `data/samples/no_interior_01.png` se rechaza con 99.5 % de exterior tanto como "Antes" como "Después", y se aceptan la escena sintética y el par `ejemplo_01` |
| Parámetros de cámara por foto: perfil calibrado, distancia focal EXIF o estimación por campo de visión | Completado |
| Calibración con tablero de ajedrez (`scripts/calibrate_camera.py`) | Herramienta completa y verificada con una cámara virtual; falta calibrar la cámara física |
| Análisis completo en CPU (RF-010) | Medido ([docs/evidencia/rendimiento.md](docs/evidencia/rendimiento.md)): 7.72-9.15 s por par sintético de 640x480 y 11.68-11.95 s por par real de 1920x1080 (un solo análisis por par) con los modelos cargados, con aviso al usuario; mismos cambios que en GPU |
| Historial de comparaciones de la sesión (RF-014) | Completado: lista de los análisis de la sesión, del más reciente al más antiguo, con el reporte de cada uno (en memoria, máximo 20) |
| Manual de usuario integrado (RIS-001 y RIS-003): pautas para tomar las fotos, requisitos de las imágenes, lectura del resultado y avisos frecuentes | Completado |
| Ejemplos precargados (RIS-002): escenas de prueba y fotos reales sin metadatos, en Archivo > Cargar ejemplo | Completado |
| Mensajes e interfaz sin términos técnicos (RNF-001): avisos, textos de la ventana y nombre de la cámara ("Cámara N") | Completado; lo verifican `tests/test_user_messages.py` y `tests/test_ui_texts.py` |
| Pruebas | Sin integración: 394 pasan y 25 se omiten. Con `RUN_INTEGRATION=1`: 418 pasan y 1 se omite (AKAZE, requiere `opencv-contrib-python`) |

### Rendimiento medido

Variante *Small*, imagen 640x480, `python -m scripts.benchmark_depth` (medido el 2026-10-29,
commit `1415b4c`; solo profundidad, no la afecta RF-012):

| Hardware | Precisión | Inferencia estable por imagen | Profundidad por par | Carga del modelo |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GB) | float16 (mixta) | 0.024 s | 0.048 s | 5.6-7.4 s |
| NVIDIA GeForce RTX 3070 (8 GB) | float32 | 0.039 s | 0.078 s | 5.6 s |
| CPU | float32 | 0.86 s | 1.72 s | 5.8 s |

La primera inferencia tras cargar el modelo tarda más (~0.7 s en GPU) por la inicialización de
CUDA. La carga del modelo es el costo dominante, por lo que conviene mantenerlo en memoria
entre análisis. En float16 la diferencia con float32 en CPU tiene una mediana menor al 2 %.

Las cifras del análisis completo salen de la evidencia vigente,
[docs/evidencia/rendimiento.md](docs/evidencia/rendimiento.md) (`python -m scripts.measure_requirements`,
corrida sobre el commit `305f284`), que es la fuente única de estas mediciones. Cada análisis
segmenta la foto "Antes", la vista "Después" alineada y, para la verificación de escena interior,
la foto "Después" original.

Análisis completo de un par con `PipelineController` en la RTX 3070, con los modelos ya
cargados:

| Resolución | Pares | Análisis por par | Profundidad | Alineación | Segmentación | Comparación | Total |
|---|---|---|---|---|---|---|---|
| 640x480 | 3 escenas sintéticas | media de 10 | 0.05 s | 0.06 s | 0.22-0.23 s | 0.01-0.03 s | 0.34-0.38 s |
| 1920x1080 | 3 pares reales | media de 3 | 0.08 s | 0.26-0.29 s | 0.26-0.30 s | 0.16-0.29 s | 0.84-0.90 s |

El primer análisis del proceso, que incluye la carga de ambos modelos, tardó 12.94 s (10.56 s de
carga). PyTorch reservó como máximo 871 MiB en la GPU; según `nvidia-smi`, la memoria usada subió
1551 MiB sobre el reposo.

Análisis completo en CPU frente a GPU (el primer análisis del proceso incluye la carga):

| Par | Hardware | Primer análisis (carga) | Con modelos en memoria | Profundidad | Segmentación |
|---|---|---|---|---|---|
| Sintético 640x480 (desplazamiento) | CPU | 16.59 s (6.84 s) | 8.02 s (media de 3) | 1.92 s | 5.94 s |
| Sintético 640x480 (desplazamiento) | RTX 3070 | 12.94 s (10.56 s) | 0.38 s (media de 10) | 0.05 s | 0.23 s |
| Real 1920x1080 (par 2) | CPU | — | 11.95 s (un solo análisis) | 2.96 s | 7.85 s |
| Real 1920x1080 (par 2) | RTX 3070 | — | 0.86 s (media de 3) | 0.08 s | 0.26 s |

En CPU la segmentación (SegFormer-B4) es la etapa más lenta. Ambos dispositivos detectan los
mismos cambios; las distancias difieren en 0.1-0.2 cm por la precisión mixta de la GPU (51.6 cm
en GPU y 51.5 cm en CPU para la caja desplazada). La carga de los modelos depende del disco: con
los pesos ya en la caché del sistema bajó a 1.7 s en GPU (medido antes de la verificación de
interior (RF-012), commit `051e3a4`); en la primera ejecución tras encender el equipo puede tardar
varios segundos más.

## Requisitos de las imágenes de entrada

| Criterio | Valor | Efecto si no se cumple |
|---|---|---|
| Formato | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp` | Se rechaza |
| Resolución mínima | 640x480 (cualquier orientación) | Se rechaza |
| Color | RGB, 8 bits por canal (RGBA y paleta se convierten a RGB) | Se rechaza |
| Iluminación | Luminancia media >= 40 (escala 0-255) | Se procesa con advertencia |
| Escena interior | Como máximo 25 % de la foto con clases de exterior (cielo, árbol, césped, tierra, montaña, mar, edificio...) según la segmentación | El análisis se detiene antes de la comparación e indica qué foto reemplazar |

La comprobación de escena interior (RF-012) usa la segmentación de SegFormer, así que ocurre
durante el análisis, no al cargar la foto. Se juzgan las dos fotos originales completas, tal como
se tomaron, y no la vista "Después" alineada: una alineación errónea entre fotos sin relación
podría deformar un exterior hasta que pareciera una habitación. Las plantas no cuentan como
exterior, porque las macetas son comunes dentro de una habitación. Se configura con
`segmentation.check_indoor_scene` y `segmentation.max_outdoor_percent`, y no se aplica si la
segmentación está desactivada.

Para obtener resultados confiables, tome ambas fotografías desde la misma posición y con la
misma orientación de cámara, con iluminación similar.

Estos requisitos y las pautas para tomar las fotos también están en el manual de la aplicación
(menú Ayuda > Manual de usuario, F1).

## Instalación

Requiere Python 3.10 o superior (verificado con 3.13).

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Aceleración por GPU (opcional)

En Windows, `pip` instala por defecto la versión de PyTorch solo para CPU. Con una GPU NVIDIA
cuyo controlador soporte CUDA 13.0 o superior (verifíquelo con `nvidia-smi`), instale además:

```bash
pip install -r requirements-cuda.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Debe imprimir: True NVIDIA GeForce ...
```

Para controladores más antiguos, use el comando que indica el selector de
<https://pytorch.org/get-started/locally/> para su versión de CUDA.

Sin GPU la aplicación funciona igual en CPU y avisa que el proceso puede tardar más.

### Pesos del modelo

La primera ejecución descarga en `models/weights/` los pesos del modelo de profundidad (unos
95 MB para la variante *Small*) y los de SegFormer-B4
(`nvidia/segformer-b4-finetuned-ade-512-512`, unos 490 MB); los tamaños son aproximados, según la
caché local. La aplicación busca siempre primero esos archivos locales, sin conectarse a la red:
si ya están guardados, ninguna ejecución vuelve a acceder a internet. Solo si faltan, y solo con
`local_files_only=False` (el valor por defecto), los descarga una vez. Con
`local_files_only=True` nunca se accede a la red y, si faltan, se informa que los archivos del
modelo deben estar en el equipo. Para poder funcionar sin conexión, mientras carga los modelos la
aplicación desactiva la consulta automática que `transformers` hace en internet sobre la
conversión de pesos antiguos (`.bin`) al formato safetensors (variable
`DISABLE_SAFETENSORS_CONVERSION`; si ya está definida, se respeta su valor).

Revise la licencia de cada checkpoint en su ficha de Hugging Face antes de un uso comercial:
la variante *Small* se distribuye bajo Apache-2.0 y las variantes *Base* y *Large* bajo
CC-BY-NC-4.0.

## Aplicación de escritorio

```bash
python -m ui                     # con la configuración por defecto
python -m ui settings.json       # con un archivo de configuración
```

La ventana principal guía el análisis en tres pasos:

1. **Elija las dos fotos.** Un panel "Antes" y otro "Después" (detallados abajo). Las
   advertencias de las fotos, como la de baja luz, aparecen en un aviso amarillo.
2. **Analizar diferencias.** El análisis corre en un hilo de fondo (`ui/pipeline_worker.py`),
   así que la ventana no se congela; una barra muestra la etapa en curso. Si falla, la
   ventana vuelve al paso 1 con un aviso rojo y conserva las fotos para reintentar.
3. **Resultado.** A la izquierda, la imagen con los cambios marcados (rueda del ratón para
   acercar o alejar, arrastrar para moverse) y, debajo, las dos fotos originales ("Antes" y
   "Después") en miniatura. A la derecha, un aviso verde si no hubo cambios
   o una tarjeta por objeto (tipo de cambio, categoría, distancia en cm y una explicación),
   las advertencias en amarillo y los tiempos. **Exportar reporte...** guarda la imagen en PNG
   y un resumen en texto (`.txt`) o JSON (`.json`) con el mismo nombre. **Guardar imagen...**
   (también en el menú Archivo, Ctrl+S) guarda solo la imagen con los cambios marcados, en PNG
   y a su resolución original, sin importar el zoom en pantalla.

El menú y la barra de herramientas ofrecen **Nuevo análisis**, **Cargar fotos...** (una foto
llena el siguiente panel vacío; dos fotos se toman como "Antes" y "Después"), **Capturar de
cámara**, **Exportar...**, **Historial...** (Ctrl+H), **Manual de usuario** (F1) y **Acerca
de**. Los modelos se cargan en el primer análisis y se reutilizan en los siguientes; se liberan
al cerrar la ventana.

**Manual de usuario** (menú Ayuda o F1, disponible también durante un análisis) explica cómo
usar la aplicación, cómo tomar las dos fotos para que se puedan comparar (cámara fija, mismo
encuadre, iluminación parecida), los requisitos de las imágenes, cómo leer el resultado y qué
hacer ante cada aviso. Sus cifras se leen de la configuración con la que corre la aplicación.

**Historial...** abre la lista de los análisis de la sesión activa, del más reciente al más
antiguo, y muestra el reporte completo del que se elija, que también se puede exportar desde
ahí. Solo guarda los análisis que terminaron con un resultado, conserva como máximo los 20
últimos y se pierde al cerrar la aplicación, porque reside en memoria. No está disponible
mientras se analiza un par.

### Ejemplos precargados

**Archivo > Cargar ejemplo** carga con un clic un par de fotos "Antes" y "Después" en los paneles,
sin buscar archivos; el análisis se inicia, como siempre, con "Analizar diferencias". Sirve para
demostrar y verificar la aplicación sin fotos propias. Los ejemplos se declaran en
[data/samples/examples.json](data/samples/examples.json) y hoy son:

- **Escenas de prueba** generadas por computadora (640x480): una caja desplazada, una caja
  retirada y la misma habitación con poca luz (muestra el aviso de baja iluminación).
- **Fotos reales** de un espacio interior tomadas por el autor del proyecto (tres pares de
  1920x1080, `ejemplo_01_a.jpg` a `ejemplo_03_b.jpg`), versionadas sin metadatos: se les quitaron
  los datos EXIF, incluida la ubicación GPS que traían los originales.

Para agregar un par, copie las dos fotos en `data/samples/` y agregue una entrada al manifiesto
con `id` (único), `title`, `description`, `before` y `after` (solo el nombre del archivo, `.png`,
`.jpg` o `.jpeg`). Quite antes los metadatos de las fotos, no incluya personas ni datos personales
y no use nombres que empiecen por `real_` (ese prefijo lo reservan las pruebas para los pares de
validación). Las entradas inválidas se omiten sin impedir que la aplicación arranque.

Los análisis necesitan los pesos de los modelos en la caché local (`models/weights/`): si no
están, se descargan una vez en el primer análisis; después la demostración funciona sin conexión.

### Paneles de selección de imágenes

Cada panel permite **Cargar archivo...** o **Capturar** desde la cámara indicada en el selector
"Cámara". Cada imagen se valida al cargarla y muestra su miniatura con un estado:

| Estado | Significado | ¿Permite continuar? |
|---|---|---|
| VÁLIDA | Cumple todos los criterios | Sí |
| ADVERTENCIA LUZ | Imagen poco iluminada; el resultado puede no ser confiable | Sí |
| ERROR RESOLUCIÓN | Menor que 640x480 | No |
| ERROR COLOR | No es RGB de 8 bits (p. ej. escala de grises) | No |
| ERROR FORMATO / ERROR ARCHIVO | Extensión no admitida, archivo dañado o inexistente | No |
| ERROR CÁMARA | La cámara no está disponible o no entregó imagen | No |
| NO ES INTERIOR | El análisis la rechazó porque muestra una escena de exterior (se asigna al analizar, no al cargar) | No, hasta reemplazarla |

VÁLIDA y ADVERTENCIA LUZ solo indican que la foto cumple los requisitos técnicos (formato, color,
resolución e iluminación); el panel lo aclara, y avisa que la comprobación de interior ocurre al
analizar.

"Analizar diferencias" se habilita cuando ambas imágenes son utilizables. Los mismos paneles
forman `ImageSelectionDialog`, un diálogo independiente con el botón "Continuar".

Si una computadora tiene varias cámaras (por ejemplo, una cámara infrarroja o virtual además de
la webcam), la numeración depende del sistema operativo. Pruebe con 0, 1, 2... o fije el valor en
`capture.camera_index` del archivo de configuración.

## Pruebas

```bash
# Pruebas unitarias (rápidas, sin red, GPU ni cámara; la interfaz se prueba sin ventana)
pytest

# Pruebas de integración con el modelo real (descargan los pesos la primera vez)
RUN_INTEGRATION=1 pytest -m integration -v -s           # bash
$env:RUN_INTEGRATION=1; pytest -m integration -v -s     # PowerShell
```

Las pruebas de integración del pipeline ([tests/test_pipeline_integration.py](tests/test_pipeline_integration.py))
ejecutan el análisis completo con los modelos reales sobre los pares de muestra (caja movida,
retirada, sin cambios y con poca luz) y verifican que los tres pasos con PyTorch corren en la GPU.

Las pruebas de integración de hardware ([tests/test_hardware_integration.py](tests/test_hardware_integration.py))
verifican que la inferencia corre en la GPU, miden la latencia, comparan float16 en GPU con
float32 en CPU, capturan desde la webcam y calculan el mapa de profundidad coloreado de la escena
sintética `scene_before.png`.

Ninguna prueba modifica el repositorio: las imágenes que generan van al directorio temporal de
pytest ([tests/sample_outputs.py](tests/sample_outputs.py)). Solo con `UPDATE_SAMPLES=1` se
regeneran las dos muestras versionadas, `data/samples/comparison_output_sample.png` y
`data/samples/depth_output_sample.png`. Las pruebas que no encuentran su hardware o sus datos se
omiten:

- **Fotos reales**: copie cada par de fotos del mismo espacio como
  `data/samples/real_before<sufijo>.jpg` y `data/samples/real_after<sufijo>.jpg` (cualquier
  formato admitido), por ejemplo `real_before.jpg`/`real_after.jpg` y
  `real_before_2.jpg`/`real_after_2.jpg`. Cada par se prueba por separado y su resultado
  anotado se guarda como `real_comparison_output<sufijo>.png` en el directorio temporal de pytest
  (nunca en el repositorio; `.gitignore` excluye las fotos personales). El resultado esperado de cada par
  puede declararse en `data/samples/real_pairs.json` (ver [tests/real_pairs.py](tests/real_pairs.py)).
- **Webcam**: se prueban los índices 0-3; para fijar uno, defina `CAMERA_INDEX`
  (`$env:CAMERA_INDEX=1` en PowerShell).

### Medición de latencia

```bash
python -m scripts.benchmark_depth                          # GPU si está disponible
python -m scripts.benchmark_depth --device cpu             # forzar CPU
python -m scripts.benchmark_depth --full-precision         # GPU en float32
python -m scripts.benchmark_depth --variant base --repetitions 20 --image ruta/a/foto.jpg

# Análisis completo (profundidad, alineación, segmentación y comparación)
python -m scripts.benchmark_pipeline                       # CPU y GPU, pares de muestra
python -m scripts.benchmark_pipeline --devices cpu --repetitions 5
python -m scripts.benchmark_pipeline --pair antes.jpg despues.jpg
```

### Mediciones de rendimiento

La evidencia de los requisitos de rendimiento y portabilidad (RR-001, RR-002, RF-009, RF-010 y
RA-002) se genera con un solo comando:

```bash
python -m scripts.measure_requirements                  # GPU y CPU (unos 3 minutos)
python -m scripts.measure_requirements --device cuda    # solo uno de los dos
```

Mide cada dispositivo en un proceso nuevo (el primer análisis incluye la carga de los modelos),
simula un equipo sin tarjeta gráfica ocultando la GPU al proceso de CPU, registra el uso de la GPU
con `nvidia-smi` frente a su consumo en reposo y no usa la red (los pesos deben estar en
`models/weights/`). Escribe [docs/evidencia/rendimiento.md](docs/evidencia/rendimiento.md) y
`rendimiento.json`, con el resultado de cada requisito, el entorno y los tiempos por etapa.

## Uso programático

```python
from config import load_settings
from services.capture import ImageAcquisitionService
from services.inference import create_depth_estimator

settings = load_settings()                      # o load_settings("settings.json")
acquisition = ImageAcquisitionService()

before = acquisition.from_file("data/samples/scene_before.png")
for warning in before.warnings:
    print("Advertencia:", warning)

with create_depth_estimator(settings.depth) as estimator:
    if estimator.compute_device.notice:
        print(estimator.compute_device.notice)
    depth = estimator.estimate(before.image)

print(depth.shape, f"{depth.min_depth:.2f}-{depth.max_depth:.2f} m")
```

Análisis completo de un par:

```python
from PIL import Image

from controllers import PipelineController

with PipelineController(settings) as pipeline:
    report = pipeline.analyze_files("data/samples/scene_before.png",
                                    "data/samples/scene_after_moved.png")

for warning in report.warnings:
    print("Advertencia:", warning)
for obj in report.objects:
    print(obj.object_id, obj.summary, obj.position_before, obj.position_after)
Image.fromarray(report.annotated_image).save("resultado.png")
```

Los cambios se clasifican así:

| Tipo | Criterio | Datos reportados |
|---|---|---|
| `DESPLAZAMIENTO` | Una región que se alejó (lugar que quedó libre) y una que se acercó (nuevo lugar) con colores similares | Posición 3D antes y después, vector `(dX, dY, dZ)` y distancia en cm |
| `RETIRADO / DESAPARICIÓN` | Región que se alejó sin contraparte: ahora se ve el fondo | Posición 3D anterior |
| `APARICIÓN` | Región que se acercó sin contraparte: un objeto nuevo | Posición 3D actual |

Un píxel cuenta como cambio solo si su profundidad varió al menos 5 cm **y** su color también
cambió. La segunda condición es necesaria porque el modelo monocular estima cada foto por
separado y desplaza unos centímetros superficies que no cambiaron; el color de esas superficies,
en cambio, se mantiene. Los cambios globales de brillo se compensan antes de comparar colores.

Reglas adicionales, ajustadas con fotos reales:

- **Fusión**: una región sin pareja que toca la huella de otro objeto y se le parece en color se
  integra a ese objeto.
- **Zonas persistentes**: un retiro o una aparición se descarta si la segmentación ve lo mismo
  en esa zona en ambas fotos (misma clase de objeto, o el 85 % de los píxeles con la misma
  clase). Al retirar un objeto, el modelo de profundidad a veces estima unos centímetros más
  cerca la superficie que queda descubierta; sin esta regla, aparecería un objeto inexistente.
  Esas zonas tampoco se ofrecen como pareja de otro objeto en los pasos siguientes.
- **Objetos juntos**: si dos objetos terminan uno sobre otro o lado a lado (una taza sobre una
  caja), forman una sola región. Esa región se divide por color: se retroproyecta sobre ella el
  histograma de cada lugar que quedó libre y cada píxel se asigna al objeto que mejor lo explica.
  Lo mismo se hace en sentido inverso con dos objetos que estaban juntos y se separan.
- **Objetos que giran**: un lugar libre y una región nueva sin pareja que la segmentación
  reconoce como la misma clase de objeto, y de tamaño comparable, se emparejan como un
  desplazamiento aunque sus colores no coincidan (el objeto muestra otras caras).
- **Seguimiento del objeto**: la distancia de un desplazamiento se mide emparejando puntos SIFT
  del propio objeto entre ambas fotos alineadas y tomando la mediana de su movimiento 3D. Cuando
  un objeto se mueve menos que su propio tamaño, las zonas que cambian son franjas en sus bordes
  y la distancia entre ellas mediría el ancho del objeto, no su movimiento. Si menos de 5 puntos
  siguen al objeto, se usan los centroides de las regiones.
- **Nombre**: cada objeto se nombra con la vista ("antes" o "después") en la que la segmentación
  lo reconoce con mayor cobertura.
- **Reproducibilidad**: la alineación usa una semilla fija (`alignment.random_seed`), de modo que
  el mismo par produce siempre el mismo resultado.

Si las fotos no comparten suficientes puntos para alinearlas, el análisis continúa sin alinear
(suponiendo cámara fija) y lo indica en `report.warnings`; con
`alignment.fallback_to_identity = false` se lanza `AlignmentError`.

### Configuración

Todos los parámetros tienen valores por defecto. Un archivo JSON puede sobrescribir solo los
necesarios:

```json
{
  "depth": { "variant": "base", "device": "cuda", "half_precision": true },
  "camera": { "fx": 1450.0, "fy": 1450.0, "cx": 960.0, "cy": 540.0,
              "reference_width": 1920, "reference_height": 1080 },
  "camera_estimation": { "use_exif": true, "horizontal_fov_deg": 63.3 },
  "capture": { "camera_index": 1 },
  "alignment": { "detector": "sift", "matcher": "flann", "ratio_test": 0.75,
                 "ransac_reprojection_threshold": 3.0, "min_inliers": 10 },
  "comparison": { "change_threshold_m": 0.05, "appearance_gating": true },
  "segmentation": { "enabled": true, "min_label_coverage": 0.25 }
}
```

El detector `akaze` requiere `opencv-contrib-python`: desde OpenCV 5, AKAZE se distribuye en
los módulos adicionales. SIFT (por defecto) y ORB vienen incluidos en `opencv-python`.

### Parámetros de la cámara

La distancia en centímetros usa la distancia focal (`fx`, `fy`) y el centro óptico (`cx`, `cy`)
de la cámara que tomó la foto "antes". Como dependen de la cámara, la resolución y el zoom, el
sistema elige para cada foto la fuente más precisa disponible
([utils/camera_model.py](utils/camera_model.py)):

1. **Perfil calibrado** (`camera` en la configuración), si la foto tiene la misma proporción
   (por ejemplo 16:9) que la calibración; se reescala a la resolución de la foto.
2. **Distancia focal EXIF**: la equivalente a 35 mm que guardan los celulares y las cámaras
   digitales (`fx = f35 · diagonal_px / 43.27`), con píxeles cuadrados y el centro de la imagen.
3. **Estimación por campo de visión** (`camera_estimation.horizontal_fov_deg`, 63.3° por
   defecto, típico de webcams y celulares) para fotos sin datos, como las capturas de la webcam.

El reporte indica qué fuente se usó (`report.intrinsics_source`, `report.focal_length_px`), y la
pantalla de resultado y los archivos exportados lo explican en lenguaje sencillo. La
profundidad no depende de estos parámetros (el modelo la entrega en metros); influyen en el
movimiento lateral y vertical del objeto.

Para calibrar una cámara con un tablero de ajedrez:

1. Imprima un tablero de 10 x 7 cuadros (9 x 6 esquinas internas), péguelo en algo plano y mida
   con regla el lado de un cuadro.
2. Tome entre 15 y 20 fotos con la misma cámara, resolución y zoom que usará en los análisis,
   con el tablero completo en distintos ángulos, distancias y posiciones (incluidos los bordes).
   Con la webcam, el propio script puede tomarlas: cada Enter guarda una foto.
3. Calibre y guarde el perfil en un archivo de configuración:

```bash
python -m scripts.calibrate_camera fotos_tablero --capture 15 --camera 0   # opcional: tomar las fotos
python -m scripts.calibrate_camera fotos_tablero --square-mm 25 --output settings.json
python -m ui settings.json
```

El script informa el error de calibración (menos de 0.5 px es bueno) y las fotos en las que no
encontró el tablero. El perfil se aplica a todas las fotos con su misma proporción, así que
calibre la cámara con la que realmente toma las fotos que analiza.

## Estructura del proyecto

```
|-- config/              Configuración y parámetros (Pydantic)
|-- models/              Estructuras de datos del dominio y caché de pesos
|-- services/
|   |-- capture/         Carga desde disco y captura desde cámara
|   |-- inference/       Estimación de profundidad (Depth Anything V2) y selección de hardware
|   |-- alignment/       Homografía RANSAC y deformación de perspectiva
|   |-- segmentation/    Segmentación semántica (SegFormer) y nombres de objetos
|   `-- comparison/      Diferencia de profundidad, metrología 3D y clasificación
|-- controllers/         Orquestación del pipeline (PipelineController)
|-- utils/               Validación de imágenes, visualización y utilidades numéricas
|-- ui/                  Interfaz gráfica (ventana principal, resultado y selección de imágenes)
|-- scripts/             Medición de latencia (profundidad y análisis completo), evidencia de
|                        rendimiento (measure_requirements) y calibración
|-- tests/               Pruebas con pytest
|-- data/samples/        Imágenes de ejemplo y resultados de muestra
|-- docs/evidencia/      Evidencia de rendimiento y portabilidad (rendimiento.md y .json)
|-- Documents/           Documentos del proyecto en PDF (avances, diseño, requerimientos, plan)
|-- pyproject.toml       Configuración de pytest
|-- requirements.txt
`-- requirements-cuda.txt  PyTorch con soporte CUDA (opcional)
```

## Próximos pasos

1. **Escena interior**: calibración del umbral del 25 % con casos límite (habitaciones con
   ventanales, balcones, patios techados).
2. **Validación con más fotos reales**: calibrar con el tablero la cámara que toma las fotos,
   y más pares (objetos retirados, varios cambios, cámara en mano) con distancias medidas con
   cinta, para cuantificar el error de cada fuente de parámetros.
3. **Captura desde la cámara en segundo plano**: hoy se ejecuta en el hilo de la interfaz y la
   ventana puede pausarse mientras la cámara se estabiliza.
