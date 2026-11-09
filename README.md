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
   `fx, fy, cx, cy`; clasificación en *retirado*, *desplazamiento* (con la distancia 3D en
   centímetros) o *aparición*.
6. **Reporte**: lista estructurada de objetos y la imagen con los cambios resaltados.

Todo el procesamiento es local; ninguna imagen se envía a servicios externos.

### Arquitectura

Arquitectura en capas con paquetes independientes. Solo la capa de orquestación invoca a los
servicios, de modo que cada modelo puede sustituirse sin modificar los demás módulos.

| Paquete | Responsabilidad |
|---|---|
| `config/` | Configuración tipada e inmutable (Pydantic): criterios de imagen, cámara, modelo e intrínsecos `fx, fy, cx, cy`. |
| `models/` | Estructuras de datos del dominio (`ImageData`, `DepthMap`, `ValidationReport`, `AcquiredImage`, `AlignmentResult`, `SegmentationMask`, `ChangedObject`, `ComparisonReport`) y jerarquía de excepciones. Caché local de pesos en `models/weights/`. |
| `services/capture/` | Fuentes de imagen (`ImageSource`, implementada por `FileImageLoader` y `CameraCapture`) y `ImageAcquisitionService`, que entrega imágenes ya validadas. |
| `services/inference/` | `BaseDepthEstimator` (contrato común), `DepthAnythingV2Estimator`, `DeviceManager` (GPU/CPU), cargador de modelos de Hugging Face y `DepthLatencyBenchmark`. |
| `services/alignment/` | `SpatialAlignmentService`: puntos de interés, emparejamiento, homografía RANSAC y deformación de imagen, profundidad y máscara de solapamiento. |
| `services/segmentation/` | `SegFormerSegmenter` (GPU/CPU) y `RegionLabeler`, que nombra en español la región de un cambio. |
| `services/comparison/` | `SpatialComparisonService`: diferencia de profundidad, regiones de cambio, retroproyección 3D y clasificación. |
| `controllers/` | `PipelineController`: orquestación de extremo a extremo (fachada única para la interfaz). |
| `utils/` | `ImageValidator`, estadísticas fotométricas, geometría pinhole, coloreado de mapas de profundidad, dibujo de los cambios y generador de escenas sintéticas. |
| `ui/` | Interfaz gráfica de escritorio (PySide6): diálogo de selección de imágenes. |
| `scripts/` | Utilidades de línea de comandos (medición de latencia). |
| `tests/` | Pruebas unitarias y de integración con `pytest`. |
| `data/samples/` | Escenas interiores sintéticas para demostración y pruebas sin conexión. |

Patrones aplicados:

- **Estrategia**: `BaseDepthEstimator` define `estimate(image) -> DepthMap`; cualquier modelo
  de profundidad se integra implementando `_predict`, `load` y `unload`.
- **Método plantilla**: `estimate` valida la entrada, carga el modelo de forma perezosa, mide el
  tiempo, verifica la resolución de salida y traduce fallos de ejecución (p. ej. memoria de GPU
  insuficiente) a `DepthEstimationError`.
- **Adaptador**: `DepthAnythingV2Estimator` adapta el procesador y el modelo de Hugging Face a
  la interfaz uniforme del estimador.
- **Inyección de dependencias**: el dispositivo de video, el selector de hardware, los
  cargadores de modelos y cada servicio del pipeline son reemplazables, lo que permite probar
  cada módulo sin hardware ni descargas.
- **Fachada**: `PipelineController` expone `analyze_files(antes, después)` y
  `analyze(antes, después)`, y oculta el orden de las etapas y la vida de los modelos.

## Estado actual

**Avance 2 completado: alineación espacial, metrología 3D, segmentación y orquestación.**

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
| Fotos reales, par 1 (1920x1080, caja desplazada sobre una cama) | Verificada: 1 desplazamiento, "caja", 63.3 cm |
| Fotos reales, par 2 (taza desplazada, caja movida 5.6 cm, cubo retirado) | Verificada: los 3 cambios, sin falsos positivos; caja medida en 6.5 cm (error de 0.9 cm) |
| `PipelineController` y `ComparisonReport` con imagen anotada | Completado |
| Fotos reales, par 3 (caja girada y movida, taza colocada encima de la caja) | Verificada: 2 desplazamientos separados (caja 47.5 cm, taza 87.0 cm) |
| Pruebas | 203 unitarias + 20 de integración |
| Ventana principal con el reporte | Pendiente |

### Rendimiento medido

Variante *Small*, imagen 640x480, `python -m scripts.benchmark_depth`:

| Hardware | Precisión | Inferencia estable por imagen | Profundidad por par | Carga del modelo |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GB) | float16 (mixta) | 0.024 s | 0.048 s | 5.6-7.4 s |
| NVIDIA GeForce RTX 3070 (8 GB) | float32 | 0.039 s | 0.078 s | 5.6 s |
| CPU | float32 | 0.86 s | 1.72 s | 5.8 s |

La primera inferencia tras cargar el modelo tarda más (~0.7 s en GPU) por la inicialización de
CUDA. La carga del modelo es el costo dominante, por lo que conviene mantenerlo en memoria
entre análisis. En float16 la diferencia con float32 en CPU tiene una mediana menor al 2 %.

Análisis completo de un par con `PipelineController` en la RTX 3070 (media de 10 análisis con
los modelos ya cargados):

| Resolución | Profundidad | Alineación | Segmentación | Comparación | Total |
|---|---|---|---|---|---|
| 640x480 | 0.04 s | 0.05 s | 0.12 s | 0.01 s | 0.24 s |
| 1920x1080 | 0.07 s | 0.22 s | 0.14 s | 0.10 s | 0.60 s |

El primer análisis, que incluye la carga de ambos modelos, tarda ~7 s. El pico de memoria de
GPU con ambos modelos es de 869 MiB.

## Requisitos de las imágenes de entrada

| Criterio | Valor | Efecto si no se cumple |
|---|---|---|
| Formato | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp` | Se rechaza |
| Resolución mínima | 640x480 (cualquier orientación) | Se rechaza |
| Color | RGB, 8 bits por canal (RGBA y paleta se convierten a RGB) | Se rechaza |
| Iluminación | Luminancia media >= 40 (escala 0-255) | Se procesa con advertencia |

Para obtener resultados confiables, tome ambas fotografías desde la misma posición y con la
misma orientación de cámara, con iluminación similar.

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

La primera ejecución descarga los pesos (~100 MB para la variante *Small*) en
`models/weights/`. Las siguientes ejecuciones los reutilizan; con `local_files_only=True` en
la configuración no se accede a la red.

Revise la licencia de cada checkpoint en su ficha de Hugging Face antes de un uso comercial:
la variante *Small* se distribuye bajo Apache-2.0 y las variantes *Base* y *Large* bajo
CC-BY-NC-4.0.

## Diálogo de selección de imágenes

```bash
python -m ui                     # con la configuración por defecto
python -m ui settings.json       # con un archivo de configuración
```

El diálogo tiene un panel "Antes" y otro "Después". Cada uno permite **Cargar archivo...** o
**Capturar** desde la cámara indicada en el selector "Cámara". Cada imagen se valida al
cargarla y muestra su miniatura con un estado:

| Estado | Significado | ¿Permite continuar? |
|---|---|---|
| VÁLIDA | Cumple todos los criterios | Sí |
| ADVERTENCIA LUZ | Imagen poco iluminada; el resultado puede no ser confiable | Sí |
| ERROR RESOLUCIÓN | Menor que 640x480 | No |
| ERROR COLOR | No es RGB de 8 bits (p. ej. escala de grises) | No |
| ERROR FORMATO / ERROR ARCHIVO | Extensión no admitida, archivo dañado o inexistente | No |
| ERROR CÁMARA | La cámara no está disponible o no entregó imagen | No |

"Continuar" se habilita cuando ambas imágenes son utilizables; al aceptar, el diálogo imprime
el resumen del par seleccionado.

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
retirada, sin cambios y con poca luz), verifican que los tres pasos con PyTorch corren en la GPU
y guardan la imagen anotada en `data/samples/comparison_output_sample.png`.

Las pruebas de integración de hardware ([tests/test_hardware_integration.py](tests/test_hardware_integration.py))
verifican que la inferencia corre en la GPU, miden la latencia, comparan float16 en GPU con
float32 en CPU, capturan desde la webcam y guardan el mapa de profundidad coloreado en
`data/samples/depth_output_sample.png`. Las que no encuentran su hardware o sus datos se omiten:

- **Fotos reales**: copie cada par de fotos del mismo espacio como
  `data/samples/real_before<sufijo>.jpg` y `data/samples/real_after<sufijo>.jpg` (cualquier
  formato admitido), por ejemplo `real_before.jpg`/`real_after.jpg` y
  `real_before_2.jpg`/`real_after_2.jpg`. Cada par se prueba por separado y su resultado
  anotado se guarda como `real_comparison_output<sufijo>.png`. El resultado esperado de cada par
  puede declararse en `data/samples/real_pairs.json` (ver [tests/real_pairs.py](tests/real_pairs.py)).
- **Webcam**: se prueban los índices 0-3; para fijar uno, defina `CAMERA_INDEX`
  (`$env:CAMERA_INDEX=1` en PowerShell).

### Medición de latencia

```bash
python -m scripts.benchmark_depth                          # GPU si está disponible
python -m scripts.benchmark_depth --device cpu             # forzar CPU
python -m scripts.benchmark_depth --full-precision         # GPU en float32
python -m scripts.benchmark_depth --variant base --repetitions 20 --image ruta/a/foto.jpg
```

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
  "capture": { "camera_index": 1 },
  "alignment": { "detector": "sift", "matcher": "flann", "ratio_test": 0.75,
                 "ransac_reprojection_threshold": 3.0, "min_inliers": 10 },
  "comparison": { "change_threshold_m": 0.05, "appearance_gating": true },
  "segmentation": { "enabled": true, "min_label_coverage": 0.25 }
}
```

El detector `akaze` requiere `opencv-contrib-python`: desde OpenCV 5, AKAZE se distribuye en
los módulos adicionales. SIFT (por defecto) y ORB vienen incluidos en `opencv-python`.

Los intrínsecos por defecto corresponden a una cámara típica de 640x480 y se reescalan con
`CameraIntrinsics.scaled_to(width, height)`. Para mediciones en centímetros precisas, use los
valores de una calibración de su cámara.

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
|-- ui/                  Interfaz gráfica (diálogo de selección de imágenes)
|-- scripts/             Medición de latencia
|-- tests/               Pruebas con pytest
|-- data/samples/        Imágenes de ejemplo y resultados de muestra
|-- pyproject.toml       Configuración de pytest
|-- requirements.txt
`-- requirements-cuda.txt  PyTorch con soporte CUDA (opcional)
```

## Próximos pasos

1. **Ventana principal**: interfaz PySide6 que integre el diálogo de selección con el
   `PipelineController`, ejecute el análisis en segundo plano y muestre la imagen anotada y la
   lista de cambios en lenguaje no técnico.
2. **Historial de la sesión**: conservar los reportes de los análisis realizados.
3. **Validación con más fotos reales**: más pares (objetos retirados, varios cambios, cámara en
   mano), distancias medidas con cinta y una calibración de la cámara utilizada.
